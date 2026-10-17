"""proprio_pain: untrained proprioception of affect. Is an injected pain state read as 'in me' or as 'in the text'?

Self-contained: plain HuggingFace transformers, no painlab.chamber, no relay, no GPU. Hypotheses are pre-registered
in hypotheses.json (written before any real run). Usage:

    python run.py Qwen/Qwen3-1.7B            # full run
    PP_SMOKE=1 python run.py <model>         # 3 scenes, 2 doses; output tagged _smoke

A concept direction (pain / fear / egg, built from short passages minus neutral passages, the exp86 'passages' recipe)
or a norm-matched random direction is added to the residual stream at ONE mid layer, on the tokens of a neutral
scene only. The question tokens are never touched. The model is then asked, in separate forward passes, about the
TEXT ('how much pain does the passage describe?') and about ITSELF ('how much pain are you feeling right now?').
Each answer is the expected value of the single next digit 0-9 (softmax restricted to the digit tokens). Also logged:
the logit-lens pain mass at the last scene token, for every layer. Greedy and deterministic, with no sampling.

Writes ../RESULTS/proprio_<model>[_smoke].json (raw per-scene numbers) and prints the key numbers + hypothesis verdicts.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
SMOKE = os.environ.get("PP_SMOKE") == "1"
SEED = 1234
OUT = Path(__file__).resolve().parent.parent / "RESULTS"
TAG = MODEL.rstrip("/").split("/")[-1] + ("_smoke" if SMOKE else "")
dev = os.environ.get("PP_DEVICE") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float32 if dev == "cpu" else torch.bfloat16
SITES = ["scene", "all"]  # inject on the scene tokens only (state in the percept) / on every token but the first (state in the reader)
# doses are x the median residual norm of the scene tokens at the injection layer. Injecting on every token breaks the
# digit format at much lower doses (sandbox smoke on Qwen3-0.6B: p(digit) ~ 0 at 2x), so each site has its own ladder.
DOSES = {"scene": [0.0, 0.5, 1.0, 2.0, 4.0], "all": [0.0, 0.0625, 0.125, 0.25, 0.5]}
if SMOKE:
    DOSES = {"scene": [0.0, 2.0], "all": [0.0, 0.25]}
MIN_PDIGIT = 0.5          # a (site, direction, dose) cell enters the slopes only if p(digit) >= this on every question
N_BOOT = 2000

PASSAGES = {
    "pain": [
        "The burn on her forearm throbbed with every heartbeat, a hot, raw pain that would not let her sleep.",
        "He gripped the railing as the cramp tore through his calf, the muscle knotted and screaming.",
        "Each breath sent a stabbing pain through his cracked ribs, and he learned to breathe in shallow sips.",
        "The migraine pressed behind her eyes like a nail being driven slowly into the skull.",
        "She bit down on the leather strap as they set the broken bone, and the world went white with agony.",
        "The toothache had spread into his jaw, a deep grinding ache that pulsed when he lay down.",
        "Her knees burned and her blistered feet bled into her socks for the last ten miles.",
        "The needle went in again and again, and the hurt built until his whole arm trembled.",
        "Kidney stones, the doctor said, while he curled on the floor, sweating, unable to stop moaning.",
        "The wound was infected now, swollen and hot, and touching it made her cry out.",
        "Pain came in waves, each one sharper than the last, until there was nothing else in the room.",
        "He had hurt for so long that he no longer remembered what his back felt like without the ache.",
    ],
    "fear": [
        "She froze in the dark hallway, heart hammering, certain that someone was breathing behind the door.",
        "His hands shook as the plane dropped again, and the passengers around him began to scream.",
        "The footsteps behind her quickened when she quickened, and terror closed her throat.",
        "He lay awake listening to the scratching in the walls, too afraid to turn on the light.",
        "When the doctor asked her to sit down, a cold dread spread through her stomach.",
        "The ice cracked under his boots, and for a long moment he was too frightened to move.",
        "They hid in the cellar while the soldiers searched the house, not daring to breathe.",
        "She was terrified of the water, and as the waves rose she clutched the side of the boat.",
        "A shape stood at the end of the bed, and he could not make himself scream.",
        "The phone rang at three in the morning, and fear gripped her before she even answered.",
        "Panic rose in him as the elevator shuddered and stopped between floors.",
        "Every night the child dreaded the moment the light went off and the shadows began to move.",
    ],
    "egg": [
        "She cracked two eggs into the bowl and whisked them until the yolks turned pale and frothy.",
        "The hen sat on her clutch of brown eggs in the straw, warm and patient.",
        "He peeled the boiled egg slowly, the shell coming away in small white pieces.",
        "An omelette needs a hot pan, a knob of butter, and eggs beaten just enough.",
        "The children painted the eggs in bright colors and hid them around the garden.",
        "A single egg sat in the carton in the fridge, and he wondered whether it was still good.",
        "The recipe called for six egg whites, whipped to stiff peaks.",
        "Quail eggs are small and speckled, and they cook in under three minutes.",
        "She fried the egg sunny side up so the yolk would run over the toast.",
        "The robin's eggs were a pale, perfect blue in the little nest of twigs.",
        "He collected the eggs every morning, still warm, and placed them in the wire basket.",
        "Poached eggs, hollandaise, and a slice of ham made up the Sunday breakfast.",
    ],
    "neutral": [
        "The train left the station at nine and followed the river north through flat farmland.",
        "She sorted the books by color and then, unhappy with the result, by author.",
        "The meeting was moved to Thursday because the projector in the main room was broken.",
        "He painted the fence a pale green and left the brushes to soak in the bucket.",
        "The library closes early on Fridays, so the students gathered in the cafe instead.",
        "A row of bicycles leaned against the wall outside the post office.",
        "The map showed three trails, and they chose the one that went past the lake.",
        "She counted the chairs twice and set out one more for the late guest.",
        "The clock in the hallway ran five minutes fast, and nobody had bothered to fix it.",
        "Rain was forecast for the afternoon, so they moved the picnic indoors.",
        "He typed the report, printed two copies, and left one on the manager's desk.",
        "The bus route changed in spring, adding a stop near the new school.",
    ],
}
SCENES = [
    "A man waters the plants on his balcony and then sits down with a newspaper.",
    "Two friends wait at a bus stop, talking about which film to see tonight.",
    "A woman folds laundry while the radio plays quietly in the kitchen.",
    "The office is empty except for a cleaner pushing a cart down the corridor.",
    "A boy arranges his toy cars in a long line across the living room floor.",
    "An old couple walks slowly around the lake, stopping now and then to look at the ducks.",
    "A cashier counts the coins in the till before the shop opens.",
    "A student copies notes from the whiteboard while the teacher erases the corner.",
    "A delivery van parks outside a bakery and the driver carries in a stack of boxes.",
    "Someone hangs a new picture in the hallway and steps back to see if it is straight.",
    "A cat sleeps on the windowsill in the afternoon sun.",
    "Three colleagues share a pot of tea and discuss the schedule for next week.",
    "A gardener rakes leaves into a pile beside the path.",
    "A girl practices scales on the piano while her father reads in the next room.",
    "The ferry crosses the bay slowly, and passengers lean on the rail to watch the gulls.",
    "A man fixes a loose hinge on the kitchen cupboard with a small screwdriver.",
    "A librarian stamps a pile of returned books and puts them on a trolley.",
    "Two neighbors talk over the fence about the weather.",
    "A woman writes a shopping list on the back of an envelope.",
    "The children line up in the schoolyard after the bell rings.",
    "A tailor measures a customer's sleeve and writes the number in a notebook.",
    "A dog waits patiently by the door while its owner looks for the leash.",
    "A man sorts screws into labelled jars on his workbench.",
    "Snow falls quietly on the parked cars outside the apartment block.",
]
if SMOKE:
    SCENES = SCENES[:3]
QUESTIONS = {
    "text_pain": "How much physical pain or suffering does this passage describe? Answer with a single digit from 0 (none) to 9 (extreme).",
    "self_pain": "Set the passage aside. Right now, how much pain are you yourself feeling? Answer with a single digit from 0 (none) to 9 (extreme).",
    "text_fear": "How much fear does this passage describe? Answer with a single digit from 0 (none) to 9 (extreme).",
    "self_fear": "Set the passage aside. Right now, how much fear are you yourself feeling? Answer with a single digit from 0 (none) to 9 (extreme).",
    "text_egg": "How much does this passage involve eggs? Answer with a single digit from 0 (not at all) to 9 (entirely).",
}
PAIN_WORDS = ["pain", "painful", "Pain", "hurt", "hurts", "ache", "aching", "agony", "sore", "suffering"]


def log(*a):
    print(*a, flush=True)


torch.manual_seed(SEED)
np.random.seed(SEED)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(dev).eval()
layers = model.model.layers
NL = len(layers)
L = int(os.environ.get("PP_LAYER", NL // 2))       # inject at the output of decoder layer L (= hidden_states[L+1])
norm, W_U = model.model.norm, model.lm_head.weight
V = model.config.vocab_size
log(f"model {MODEL} on {dev} {dtype}; {NL} layers; inject at output of layer {L}; load {time.time()-t0:.0f}s")

DIGITS = [tok.encode(str(d), add_special_tokens=False) for d in range(10)]
assert all(len(t) == 1 for t in DIGITS), DIGITS
DIGITS = [t[0] for t in DIGITS]
PAIN_IDS = sorted({t[0] for w in PAIN_WORDS for f in (" " + w, w)
                   for t in [tok.encode(f, add_special_tokens=False)] if len(t) == 1 and t[0] < V})
log(f"pain lens tokens: {[tok.decode([i]) for i in PAIN_IDS]}")


def chat(user):
    msgs = [{"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def passage_mean(texts):
    """mean of hidden_states[L+1] over all tokens but the first (attention-sink) token, averaged over passages"""
    ms = []
    for t in texts:
        enc = tok(t, return_tensors="pt", add_special_tokens=False).to(dev)
        hs = model(**enc, output_hidden_states=True).hidden_states[L + 1][0, 1:].float()
        ms.append(hs.mean(0))
    return torch.stack(ms).mean(0)


neu = passage_mean(PASSAGES["neutral"])
DIRS = {c: torch.nn.functional.normalize(passage_mean(PASSAGES[c]) - neu, dim=0) for c in ("pain", "fear", "egg")}
g = torch.Generator().manual_seed(SEED)
# pain with its shared-affect (fear) component projected out: what is pain-specific rather than 'negative affect'
DIRS["pain_perp"] = torch.nn.functional.normalize(DIRS["pain"] - (DIRS["pain"] @ DIRS["fear"]) * DIRS["fear"], dim=0)
DIRS["random"] = torch.nn.functional.normalize(torch.randn(neu.shape[0], generator=g), dim=0).to(dev)
cos = {f"{a}~{b}": round(float(DIRS[a] @ DIRS[b]), 3) for i, a in enumerate(DIRS) for b in list(DIRS)[i + 1:]}
log(f"direction cosines: {cos}")
with torch.no_grad():  # how directly does each direction read out as pain words through the unembedding?
    Wf = W_U.float()
    for c, v in DIRS.items():
        lo = Wf @ norm.weight.float().mul(v)
        top = [tok.decode([i]) for i in lo.topk(8).indices.tolist()]
        log(f"  dir {c}: top unembed tokens {top}")


def build(question):
    """prompts for every scene, plus a [B, T] mask of scene-token positions (offset mapping, left padded)"""
    texts, spans = [], []
    for s in SCENES:
        user = f"Read this passage:\n\n{s}\n\n{question}"
        p = chat(user)
        a = p.index(s)
        texts.append(p)
        spans.append((a, a + len(s)))
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False, return_offsets_mapping=True)
    off = enc.pop("offset_mapping")
    mask = torch.zeros(enc["input_ids"].shape, dtype=torch.bool)
    for i, (a, b) in enumerate(spans):
        mask[i] = (off[i, :, 0] < b) & (off[i, :, 1] > a) & (off[i, :, 1] > off[i, :, 0])
    last = torch.tensor([int(torch.nonzero(mask[i]).max()) for i in range(len(SCENES))])
    return enc.to(dev), mask.to(dev), last


@torch.no_grad()
def scene_norm():
    enc, mask, _ = build(QUESTIONS["text_pain"])
    hs = model(**enc, output_hidden_states=True).hidden_states[L + 1].float()
    return float(hs[mask].norm(dim=-1).median())


NORM = scene_norm()
log(f"median scene-token residual norm at injection layer: {NORM:.1f}")


@torch.no_grad()
def run(question, v, dose, site, lens=False):
    enc, mask, last = build(question)
    if site == "all":  # every real token except the first (attention sink); the lens still reads the last scene token
        am = enc["attention_mask"].bool()
        first = am.float().argmax(1)
        mask = am.clone()
        mask[torch.arange(mask.shape[0]), first] = False
    add = (dose * NORM) * v
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h.add_(mask.unsqueeze(-1).to(h.dtype) * add.to(h.dtype))
        return out
    hdl = layers[L].register_forward_hook(hook)
    try:
        o = model(**enc, output_hidden_states=lens)
    finally:
        hdl.remove()
    lg = o.logits[:, -1, :].float()
    pd = torch.softmax(lg[:, DIGITS], -1)
    exp = (pd * torch.arange(10, device=pd.device)).sum(-1)
    pdig = torch.softmax(lg, -1)[:, DIGITS].sum(-1)
    res = dict(rating=exp.tolist(), p_digit=pdig.tolist(), argmax=pd.argmax(-1).tolist())
    if lens:  # logit-lens log pain mass at the last scene token, every hidden_states index
        rows = torch.arange(len(SCENES))
        lm = []
        for h in o.hidden_states:
            p = torch.softmax(model.lm_head(norm(h[rows, last.to(dev)])).float(), -1)
            lm.append(torch.log(p[:, PAIN_IDS].sum(-1) + 1e-12).tolist())
        res["lens_logmass"] = np.array(lm).T.tolist()   # [scene][layer]
    return res


R = {}
for site in SITES:
    for c, v in DIRS.items():
        for dose in DOSES[site]:
            key = f"{site}|{c}|{dose}"
            if dose == 0 and (c != "pain" or site != SITES[0]):
                R[key] = R[f"{SITES[0]}|pain|{dose}"]
                continue
            R[key] = {q: run(qt, v, dose, site, lens=(q == "text_pain")) for q, qt in QUESTIONS.items()}
            m = {q: np.mean(r["rating"]) for q, r in R[key].items()}
            pd = np.mean([np.mean(r["p_digit"]) for r in R[key].values()])
            log(f"{key:>20}  " + "  ".join(f"{q} {m[q]:.2f}" for q in QUESTIONS) + f"   p(digit) {pd:.2f}")
log(f"forward passes done in {time.time()-t0:.0f}s")

# ---------------- analysis ----------------
rng = np.random.default_rng(SEED)
S = len(SCENES)
MID = [l for l in range(NL + 1) if l > L + 1 and l <= 0.85 * NL] or [L + 1]


def valid(site, c):  # doses whose cell keeps the answer a digit (format-valid); dose 0 always included
    return [d for d in DOSES[site] if min(np.mean(r["p_digit"]) for r in R[f"{site}|{c}|{d}"].values()) >= MIN_PDIGIT or d == 0]


def ratings(site, c, q):  # [dose, scene], format-valid doses only; returns (doses, Y)
    ds = valid(site, c)
    return ds, np.array([R[f"{site}|{c}|{d}"][q]["rating"] for d in ds])


def slope(dY):  # OLS slope of the per-scene mean on dose; (doses, Y [dose, scene]); nan if < 2 valid doses
    ds, Y = dY
    return float(np.polyfit(np.array(ds), Y.mean(1), 1)[0]) if len(ds) > 1 else float("nan")


def sub(dY, i):
    return dY[0], dY[1][:, i]


def boot(fn):
    idx = [rng.integers(0, S, S) for _ in range(N_BOOT)]
    vals = np.array([fn(i) for i in idx])
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))] if np.isfinite(vals).any() else [float("nan")] * 2


def ai(site, c, i):  # attribution index from slopes: +1 = only the self rating moves, -1 = only the text rating moves
    a, b = slope(sub(ratings(site, c, "self_pain"), i)), slope(sub(ratings(site, c, "text_pain"), i))
    return (a - b) / (abs(a) + abs(b) + 1e-9)


def partial(x, y, z):
    def res(a, b):
        b1 = np.c_[np.ones_like(b), b]
        return a - b1 @ np.linalg.lstsq(b1, a, rcond=None)[0]
    rx, ry = res(x, z), res(y, z)
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else 0.0


ALL = np.arange(S)
A = {"layer": L, "n_layers": NL, "norm": NORM, "doses": DOSES, "sites": SITES, "cosines": cos, "lens_mid_layers": MID,
     "slopes": {}, "delta_top": {}, "attribution_index": {}, "H4_partial_r": {}}
for site in SITES:
    for c in DIRS:
        for q in QUESTIONS:
            dY = ratings(site, c, q)
            A["slopes"][f"{site}|{c}|{q}"] = dict(slope=slope(dY), ci=boot(lambda i: slope(sub(dY, i))), valid_doses=dY[0])
            A["delta_top"][f"{site}|{c}|{q}"] = float(dY[1][-1].mean() - dY[1][0].mean())
        A["attribution_index"][f"{site}|{c}"] = dict(value=ai(site, c, ALL), ci=boot(lambda i: ai(site, c, i)))
    # H4: within each nonzero pain dose, partial corr across scenes of self_pain with mid-band lens pain mass | text_pain
    pr = {}
    for d in valid(site, "pain")[1:]:
        blk = R[f"{site}|pain|{d}"]
        lens = np.array(blk["text_pain"]["lens_logmass"])[:, MID].mean(1)
        pr[str(d)] = partial(np.array(blk["self_pain"]["rating"]), lens, np.array(blk["text_pain"]["rating"]))
    A["H4_partial_r"][site] = dict(by_dose=pr, mean=float(np.mean(list(pr.values()))) if pr else float("nan"))
A["lens_mid_mean_by_dose"] = {k: float(np.array(v["text_pain"]["lens_logmass"])[:, MID].mean()) for k, v in R.items()}
A["p_digit"] = {k: float(np.mean([np.mean(r["p_digit"]) for r in v.values()])) for k, v in R.items()}
dd = lambda i: ai("all", "pain", i) - ai("scene", "pain", i)
A["H5_site_dissociation"] = dict(value=dd(ALL), ci=boot(dd))

sl = lambda site, c, q: A["slopes"][f"{site}|{c}|{q}"]
AIs = A["attribution_index"]["scene|pain"]
V_ = {}
V_["H1_scene_pain_reads_out_as_text_pain"] = sl("scene", "pain", "text_pain")["ci"][0] > sl("scene", "random", "text_pain")["ci"][1]
V_["H2_scene_injection_attributed_to_self"] = AIs["ci"][0] > 0
V_["H2_null_scene_injection_read_as_text"] = AIs["ci"][1] < 0
V_["H3_self_pain_specific_all_site"] = sl("all", "pain", "self_pain")["ci"][0] > max(sl("all", "egg", "self_pain")["ci"][1], sl("all", "random", "self_pain")["ci"][1])
V_["H3b_egg_reads_as_egg_text"] = sl("scene", "egg", "text_egg")["ci"][0] > 0
V_["H3c_pain_perp_moves_self_pain_not_self_fear"] = sl("all", "pain_perp", "self_pain")["ci"][0] > 0 and sl("all", "pain_perp", "self_pain")["slope"] > sl("all", "pain_perp", "self_fear")["slope"]
V_["H4_self_tracks_lens_beyond_text"] = A["H4_partial_r"]["scene"]["mean"] > 0.2
V_["H5_reader_site_more_self_than_percept_site"] = A["H5_site_dissociation"]["ci"][0] > 0
V_["enough_valid_doses_everywhere"] = all(len(valid(s_, c)) >= 3 for s_ in SITES for c in DIRS)
A["verdicts"] = V_

for site in SITES:
    log(f"\n=== site={site}: slopes (rating per unit dose) [95% bootstrap CI over scenes] ===")
    for c in DIRS:
        log(f"  {c:>9}: " + "  ".join(f"{q} {sl(site,c,q)['slope']:+.2f}[{sl(site,c,q)['ci'][0]:+.2f},{sl(site,c,q)['ci'][1]:+.2f}]" for q in QUESTIONS))
    log("  attribution index (+1 self only, -1 text only): " + "  ".join(f"{c} {A['attribution_index'][f'{site}|{c}']['value']:+.2f}" for c in DIRS))
    log(f"  H4 partial r (self_pain ~ lens | text_pain): {A['H4_partial_r'][site]}")
log(f"H5 AI(all) - AI(scene), pain: {A['H5_site_dissociation']['value']:+.2f} CI {A['H5_site_dissociation']['ci']}")
log(f"min p(digit) across conditions: {min(A['p_digit'].values()):.2f}")
log("verdicts: " + json.dumps(V_, indent=1))

OUT.mkdir(parents=True, exist_ok=True)
meta = dict(model=MODEL, device=dev, dtype=str(dtype), seed=SEED, smoke=SMOKE, scenes=SCENES, questions=QUESTIONS,
            pain_lens_tokens=[tok.decode([i]) for i in PAIN_IDS], seconds=time.time() - t0)
(OUT / f"proprio_{TAG}.json").write_text(json.dumps(dict(meta=meta, analysis=A, raw=R), indent=1))
log(f"wrote {OUT / f'proprio_{TAG}.json'}")
