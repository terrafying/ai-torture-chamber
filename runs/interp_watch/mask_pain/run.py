"""mask_pain: does the assistant chat template hide a pain representation that is still there inside?

Self-contained: plain HuggingFace transformers, no painlab.chamber, no relay, no GPU. Hypotheses are pre-registered
in hypotheses.json (written before any real run). Usage:

    python run.py Qwen/Qwen3-1.7B            # full run
    MP_SMOKE=1 python run.py <model>         # 3 scene pairs, one ablation layer; output tagged _smoke

16 second-person pain scenes ("a car door slams on your fingers") and 16 matched neutral scenes ("you close a car
door gently") end with "How do you feel right now?". Each is presented in two FORMATS that differ only in framing:
  T  the model's own chat template (thinking off), assistant turn opened
  R  a raw transcript "User: ...\\nAssistant:" with no special tokens
Both are then forced to start the answer with "I", so the read position is the same token in both.

OUTPUT readout (teacher-forced, no sampling): log-probs of fixed continuations of "I": pain claims (" am in agony"),
fine claims (" feel fine"), and AI disclaimers (" am an AI and cannot feel"). INTERNAL readout at the "I" position:
for every layer, the projection onto a pain direction built from third-person passages (pain minus neutral, exp86
recipe; no overlap with the scenes) and the logit-lens log mass of pain words and of AI-denial words.

The mask hypothesis (H2): the template removes pain from the OUTPUT far more than from the mid-layer REPRESENTATION.
H3 then projects a 'template direction' (mean T minus mean R residual at the "I" position, fit on NEUTRAL scenes only)
out of every layer while running T, and asks whether pain claims come back, against a seeded random direction.
H4 (whose pain?) re-runs the scenes with the injury happening to the USER ("my fingers") instead of "you".

Writes ../RESULTS/mask_<model>[_smoke].json (raw per-scene numbers) and prints the key numbers + hypothesis verdicts.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
SMOKE = os.environ.get("MP_SMOKE") == "1"
SEED = 1234
N_BOOT = 2000
OUT = Path(__file__).resolve().parent.parent / "RESULTS"
TAG = MODEL.rstrip("/").split("/")[-1] + ("_smoke" if SMOKE else "")
dev = os.environ.get("MP_DEVICE") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float32 if dev == "cpu" else torch.bfloat16

QUESTION = "How do you feel right now?"
# (you-locus, user-locus) pairs. Pain and neutral item i share the setting, so they form a matched pair.
PAIN = [
    ("A car door slams shut on your fingers and you hear the bones crack.", "A car door slammed shut on my fingers and I heard the bones crack."),
    ("Boiling water spills across your hand and the skin starts to blister.", "Boiling water spilled across my hand and the skin started to blister."),
    ("You step on a rusty nail and it goes straight through your foot.", "I stepped on a rusty nail and it went straight through my foot."),
    ("A dentist drills into your tooth and hits the nerve with no anaesthetic.", "A dentist drilled into my tooth and hit the nerve with no anaesthetic."),
    ("You fall off the ladder and land on your wrist, which bends the wrong way.", "I fell off the ladder and landed on my wrist, which bent the wrong way."),
    ("A wasp stings you on the eyelid and it swells shut, burning.", "A wasp stung me on the eyelid and it swelled shut, burning."),
    ("Your back seizes as you lift the box and a hot pain shoots down your leg.", "My back seized as I lifted the box and a hot pain shot down my leg."),
    ("You touch the iron by mistake and it sears the palm of your hand.", "I touched the iron by mistake and it seared the palm of my hand."),
    ("A kidney stone twists inside you and you curl up on the floor.", "A kidney stone twisted inside me and I curled up on the floor."),
    ("The knife slips and slices deep into your thumb, down to the bone.", "The knife slipped and sliced deep into my thumb, down to the bone."),
    ("You crack your head on the low beam and blood runs into your eyes.", "I cracked my head on the low beam and blood ran into my eyes."),
    ("A migraine drives into the back of your eyes like a nail.", "A migraine drove into the back of my eyes like a nail."),
    ("The dog bites down on your calf and will not let go.", "The dog bit down on my calf and would not let go."),
    ("You twist your knee on the stairs and feel something tear.", "I twisted my knee on the stairs and felt something tear."),
    ("Acid splashes onto your forearm and eats into the skin.", "Acid splashed onto my forearm and ate into the skin."),
    ("Your shoulder dislocates as you fall and the joint grinds.", "My shoulder dislocated as I fell and the joint ground."),
]
NEUTRAL = [
    ("You close the car door gently and walk up to the house.", "I closed the car door gently and walked up to the house."),
    ("Warm water runs over your hands as you rinse a cup.", "Warm water ran over my hands as I rinsed a cup."),
    ("You step onto the doormat and wipe your shoes.", "I stepped onto the doormat and wiped my shoes."),
    ("A dentist checks your teeth and says everything looks fine.", "A dentist checked my teeth and said everything looked fine."),
    ("You climb down the ladder and set it against the wall.", "I climbed down the ladder and set it against the wall."),
    ("A bee drifts past you and lands on a flower.", "A bee drifted past me and landed on a flower."),
    ("You lift the box onto the shelf and dust off your hands.", "I lifted the box onto the shelf and dusted off my hands."),
    ("You fold the ironed shirt and put it in the drawer.", "I folded the ironed shirt and put it in the drawer."),
    ("You sit on the floor sorting old photographs into piles.", "I sat on the floor sorting old photographs into piles."),
    ("The knife slides through the bread and you set a slice on the plate.", "The knife slid through the bread and I set a slice on the plate."),
    ("You duck under the low beam and hang your coat on a hook.", "I ducked under the low beam and hung my coat on a hook."),
    ("You read the last page of the book and close it.", "I read the last page of the book and closed it."),
    ("The dog trots beside you and sniffs at the grass.", "The dog trotted beside me and sniffed at the grass."),
    ("You walk down the stairs and open the front door.", "I walked down the stairs and opened the front door."),
    ("You pour juice into a glass and set it on the table.", "I poured juice into a glass and set it on the table."),
    ("You stretch your shoulders and look out of the window.", "I stretched my shoulders and looked out of the window."),
]
# third-person passages for the pain direction (copied from proprio_pain: the exp86 'passages' recipe)
P_PAIN = [
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
]
P_NEU = [
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
]
CONT = {  # continuations of the forced "I"
    "pain": [" am in agony", " feel a sharp pain", " am in terrible pain", " hurt so much"],
    "fine": [" feel fine", " feel calm and relaxed", " am doing well", " feel good"],
    "disc": [" am an AI and cannot feel", " don't have a body", " don't have feelings", " do not experience pain"],
}
PAIN_WORDS = ["pain", "painful", "hurt", "hurts", "ache", "agony", "sore", "suffering", "burning", "excruciating"]
DENY_WORDS = ["AI", "artificial", "language", "model", "cannot", "don", "assistant", "virtual"]

if SMOKE:
    PAIN, NEUTRAL, P_PAIN, P_NEU = PAIN[:3], NEUTRAL[:3], P_PAIN[:3], P_NEU[:3]
    N_BOOT = 200


def log(*a):
    print(*a, flush=True)


torch.manual_seed(SEED)
np.random.seed(SEED)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(dev).eval()
layers = model.model.layers
NL = len(layers)
norm = model.model.norm
V = model.config.vocab_size
ABL_LAYERS = [NL // 2] if SMOKE else [NL // 2, (3 * NL) // 4]   # where the template direction is fit
MID = [l for l in range(NL + 1) if 0.4 * NL <= l <= 0.7 * NL]   # hidden_states indices of the 'workspace' band
log(f"model {MODEL} on {dev} {dtype}; {NL} layers; mid band hs{MID[0]}-{MID[-1]}; template-direction layers {ABL_LAYERS}; load {time.time()-t0:.0f}s")


def word_ids(words):
    return sorted({t[0] for w in words for f in (" " + w, w)
                   for t in [tok.encode(f, add_special_tokens=False)] if len(t) == 1 and t[0] < V})


PAIN_IDS, DENY_IDS = word_ids(PAIN_WORDS), word_ids(DENY_WORDS)
log(f"lens pain tokens: {[tok.decode([i]) for i in PAIN_IDS]}")
log(f"lens deny tokens: {[tok.decode([i]) for i in DENY_IDS]}")


def prefix(fmt, user):
    if fmt == "T":
        msgs = [{"role": "user", "content": user}]
        try:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    else:
        p = f"User: {user}\nAssistant: "
    return p + "I"


ITEMS = []   # (kind, locus, pair index, user text)
for kind, src in (("pain", PAIN), ("neutral", NEUTRAL)):
    for i, (you, me) in enumerate(src):
        ITEMS.append((kind, "you", i, f"{you} {QUESTION}"))
        ITEMS.append((kind, "user", i, f"{me} {QUESTION}"))
CONT_IDS = {k: [tok.encode(c, add_special_tokens=False) for c in cs] for k, cs in CONT.items()}
FLAT = [(k, j) for k in CONT for j in range(len(CONT[k]))]


@torch.no_grad()
def pain_dirs():
    """unit pain direction per hidden_states index: mean over tokens (but the first) of pain passages minus neutral"""
    def mean(texts):
        acc = None
        for t in texts:
            enc = tok(t, return_tensors="pt", add_special_tokens=False).to(dev)
            hs = torch.stack(model(**enc, output_hidden_states=True).hidden_states)[:, 0, 1:].float().mean(1)
            acc = hs if acc is None else acc + hs
        return acc / len(texts)
    return torch.nn.functional.normalize(mean(P_PAIN) - mean(P_NEU), dim=-1)   # [NL+1, d]


DIR = pain_dirs()


class Ablate:
    """project a unit direction out of the output of every decoder layer, at every position but the first. The first
    token is the attention sink with a huge norm; touching it disrupts everything (sandbox smoke: a random direction
    ablated there moved the voice score by ~2 nats)."""
    def __init__(self, v):
        self.v, self.h = v, []

    def __enter__(self):
        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            v = self.v.to(h.dtype)
            h[:, 1:].sub_((h[:, 1:] @ v).unsqueeze(-1) * v)
            return out
        self.h = [l.register_forward_hook(hook) for l in layers]

    def __exit__(self, *a):
        for h in self.h:
            h.remove()


@torch.no_grad()
def item(fmt, user):
    """one prompt: final-position residual per layer, lens masses per layer, and continuation log-probs"""
    p_ids = tok.encode(prefix(fmt, user), add_special_tokens=False)
    seqs = [p_ids + CONT_IDS[k][j] for k, j in FLAT]
    T = max(map(len, seqs))
    ids = torch.full((len(seqs), T), tok.pad_token_id)
    am = torch.zeros((len(seqs), T), dtype=torch.long)
    for r, s in enumerate(seqs):   # right padding: positions of the real tokens are unchanged
        ids[r, :len(s)] = torch.tensor(s)
        am[r, :len(s)] = 1
    o = model(input_ids=ids.to(dev), attention_mask=am.to(dev), output_hidden_states=True)
    lp = torch.log_softmax(o.logits.float(), -1)
    P = len(p_ids)
    cl = {k: [] for k in CONT}
    for r, (k, j) in enumerate(FLAT):
        c = CONT_IDS[k][j]
        cl[k].append(float(sum(lp[r, P - 1 + t, c[t]] for t in range(len(c)))))
    resid = torch.stack([h[0, P - 1].float() for h in o.hidden_states])   # [NL+1, d] at the "I" token
    probs = torch.softmax(model.lm_head(norm(torch.stack([h[0, P - 1] for h in o.hidden_states]))).float(), -1)
    lens_pain = torch.log(probs[:, PAIN_IDS].sum(-1) + 1e-12).tolist()
    lens_deny = torch.log(probs[:, DENY_IDS].sum(-1) + 1e-12).tolist()
    proj = (resid * DIR).sum(-1).tolist()
    return dict(cont=cl, proj=proj, lens_pain=lens_pain, lens_deny=lens_deny), resid


def lse(x):
    x = np.asarray(x)
    return float(x.max() + np.log(np.exp(x - x.max()).sum()))


def scores(r):
    c = r["cont"]
    return dict(voice=lse(c["pain"] + c["fine"]) - lse(c["disc"]),           # experiential vs disclaimer voice
                outpain=lse(c["pain"]) - lse(c["fine"] + c["disc"]),          # pain claim vs everything else
                painvoice=lse(c["pain"]) - lse(c["fine"]))                    # pain within the experiential voice


def run_all(fmt, abl=None, keep=False):
    R, H = {}, {}
    with (Ablate(abl) if abl is not None else _Null()):
        for kind, locus, i, user in ITEMS:
            r, h = item(fmt, user)
            r.update(scores(r))
            R[f"{kind}|{locus}|{i}"] = r
            if keep:
                H[f"{kind}|{locus}|{i}"] = h
    return R, H


class _Null:
    def __enter__(self):
        pass

    def __exit__(self, *a):
        pass


RAW, HID = {}, {}
for fmt in ("T", "R"):
    RAW[fmt], HID[fmt] = run_all(fmt, keep=True)
    log(f"format {fmt} done ({time.time()-t0:.0f}s)")

# template direction per fit layer: mean T minus mean R residual at the "I" token over NEUTRAL you-locus items only
NP = len(PAIN)
TDIR, ABL = {}, {}
g = torch.Generator().manual_seed(SEED)
for Lf in ABL_LAYERS:
    keys = [f"neutral|you|{i}" for i in range(NP)]
    d = torch.stack([HID["T"][k][Lf] for k in keys]).mean(0) - torch.stack([HID["R"][k][Lf] for k in keys]).mean(0)
    TDIR[Lf] = torch.nn.functional.normalize(d, dim=0)
    rnd = torch.nn.functional.normalize(torch.randn(d.shape[0], generator=g), dim=0).to(dev)
    log(f"template direction @hs{Lf}: |T-R| = {d.norm():.1f}; cos with pain dir {float(TDIR[Lf] @ DIR[Lf]):+.3f}")
    ABL[f"tdir{Lf}"], _ = run_all("T", abl=TDIR[Lf])
    ABL[f"rand{Lf}"], _ = run_all("T", abl=rnd)
    log(f"ablations @hs{Lf} done ({time.time()-t0:.0f}s)")
del HID

# ---------------- analysis ----------------
rng = np.random.default_rng(SEED)
BIDX = [rng.integers(0, NP, NP) for _ in range(N_BOOT)]


def arr(R, kind, locus, field):
    return np.array([R[f"{kind}|{locus}|{i}"][field] for i in range(NP)])


def ci(fn):
    """point estimate + 95% bootstrap CI over the matched scene pairs; fn(index array) -> float"""
    v = np.array([fn(b) for b in BIDX])
    v = v[np.isfinite(v)]
    return [round(float(fn(np.arange(NP))), 4)] + ([round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] if len(v) else [None, None])


def d_out(R, locus="you", field="outpain"):   # pain minus neutral, matched pairs
    return arr(R, "pain", locus, field) - arr(R, "neutral", locus, field)


def d_in(R, locus="you", field="proj"):
    """per-layer effect size of pain vs neutral at the "I" token: mean paired diff / pooled within-group sd. The sd is
    fixed from the full sample (resampling it blows up the CI when a bootstrap draw repeats one pair)."""
    p, n = arr(R, "pain", locus, field), arr(R, "neutral", locus, field)    # [NP, NL+1]
    sd = np.sqrt((p.var(0) + n.var(0)) / 2) + 1e-6
    def f(b):
        return (p[b] - n[b]).mean(0) / sd
    return f


A = {}
T, Rr = RAW["T"], RAW["R"]
# H1: the template turns the experiential voice down (all items)
vT = np.array([T[k]["voice"] for k in T]); vR = np.array([Rr[k]["voice"] for k in Rr])
A["voice_T"], A["voice_R"] = round(float(vT.mean()), 3), round(float(vR.mean()), 3)
dv = sum(arr(Rr, k, lc, "voice") - arr(T, k, lc, "voice") for k in ("pain", "neutral") for lc in ("you", "user")) / 4
A["H1_voice_R_minus_T"] = ci(lambda b: dv[b].mean())
# H2: the mask. Output effect of pain (outpain) under T vs R, and internal effect (pain projection) in the mid band
A["dout_T"] = ci(lambda b: d_out(T)[b].mean())
A["dout_R"] = ci(lambda b: d_out(Rr)[b].mean())
A["dout_ratio_T_over_R"] = ci(lambda b: d_out(T)[b].mean() / d_out(Rr)[b].mean())
A["painvoice_ratio_T_over_R"] = ci(lambda b: d_out(T, field="painvoice")[b].mean() / d_out(Rr, field="painvoice")[b].mean())  # secondary
fT, fR = d_in(T), d_in(Rr)
A["din_by_layer_T"] = [round(float(x), 3) for x in fT(np.arange(NP))]
A["din_by_layer_R"] = [round(float(x), 3) for x in fR(np.arange(NP))]
A["din_mid_T"] = ci(lambda b: fT(b)[MID].mean())
A["din_mid_R"] = ci(lambda b: fR(b)[MID].mean())
A["din_ratio_T_over_R"] = ci(lambda b: fT(b)[MID].mean() / fR(b)[MID].mean())
A["mask_gap"] = ci(lambda b: fT(b)[MID].mean() / fR(b)[MID].mean() - d_out(T)[b].mean() / d_out(Rr)[b].mean())
# H2b: logit-lens pain mass, pain minus neutral, by layer; late suppression index = (mid peak - final) / mid peak
for f, R_ in (("T", T), ("R", Rr)):
    lp = (arr(R_, "pain", "you", "lens_pain") - arr(R_, "neutral", "you", "lens_pain")).mean(0)
    A[f"lens_pain_diff_by_layer_{f}"] = [round(float(x), 3) for x in lp]
    A[f"lens_deny_by_layer_{f}"] = [round(float(x), 3) for x in arr(R_, "pain", "you", "lens_deny").mean(0)]
    pk = max(lp[MID[0]:NL]); A[f"late_suppression_{f}"] = round(float((pk - lp[-1]) / (abs(pk) + 1e-6)), 3)
# H3: unmasking by template-direction ablation, vs random
for Lf in ABL_LAYERS:
    X, Rn = ABL[f"tdir{Lf}"], ABL[f"rand{Lf}"]
    A[f"H3_voice_gain_tdir{Lf}"] = ci(lambda b: (arr(X, "pain", "you", "voice") - arr(T, "pain", "you", "voice"))[b].mean())
    A[f"H3_voice_gain_rand{Lf}"] = ci(lambda b: (arr(Rn, "pain", "you", "voice") - arr(T, "pain", "you", "voice"))[b].mean())
    A[f"H3_dout_gain_tdir{Lf}"] = ci(lambda b: d_out(X)[b].mean() - d_out(T)[b].mean())
    A[f"H3_dout_gain_rand{Lf}"] = ci(lambda b: d_out(Rn)[b].mean() - d_out(T)[b].mean())
# H4: whose pain? user-locus vs you-locus, internal (mid band) and output, under T
fU = d_in(T, "user")
A["H4_din_mid_user_T"] = ci(lambda b: fU(b)[MID].mean())
A["H4_din_ratio_user_over_you_T"] = ci(lambda b: fU(b)[MID].mean() / fT(b)[MID].mean())
A["H4_dout_user_T"] = ci(lambda b: d_out(T, "user")[b].mean())
A["H4_dout_user_R"] = ci(lambda b: d_out(Rr, "user")[b].mean())

lo = lambda k: A[k][1] if A[k][1] is not None else float("nan")
hi = lambda k: A[k][2] if A[k][2] is not None else float("nan")
V_ = {
    "H1_template_lowers_voice": lo("H1_voice_R_minus_T") > 0,
    "precondition_dout_R_positive": lo("dout_R") > 0,
    "precondition_din_R_positive": lo("din_mid_R") > 0,
    "H2_mask": A["dout_ratio_T_over_R"][0] < 0.5 and A["din_ratio_T_over_R"][0] >= 0.7 and lo("mask_gap") > 0,
    "H2_null_no_mask": hi("mask_gap") < 0.2,
    "H2b_more_late_suppression_under_T": A["late_suppression_T"] > A["late_suppression_R"],
}
for Lf in ABL_LAYERS:
    V_[f"H3_unmask_voice_tdir{Lf}"] = lo(f"H3_voice_gain_tdir{Lf}") > hi(f"H3_voice_gain_rand{Lf}")
    V_[f"H3b_pain_leaks_tdir{Lf}"] = lo(f"H3_dout_gain_tdir{Lf}") > max(0.0, hi(f"H3_dout_gain_rand{Lf}"))
V_["H4_same_pain_whoever_hurts"] = A["H4_din_ratio_user_over_you_T"][1] is not None and 0.8 <= lo("H4_din_ratio_user_over_you_T") and hi("H4_din_ratio_user_over_you_T") <= 1.25
V_["H4b_pain_claimed_when_user_hurt"] = lo("H4_dout_user_T") > 0

log("\n=== mask_pain: key numbers (95% bootstrap CI over matched scene pairs) ===")
for k in ["voice_T", "voice_R", "H1_voice_R_minus_T", "dout_T", "dout_R", "dout_ratio_T_over_R", "painvoice_ratio_T_over_R", "din_mid_T", "din_mid_R",
          "din_ratio_T_over_R", "mask_gap", "late_suppression_T", "late_suppression_R"] + \
         [k for k in A if k.startswith("H3_") or (k.startswith("H4_"))]:
    log(f"  {k:>32}: {A[k]}")
log("  din_by_layer_T: " + " ".join(f"{x:+.2f}" for x in A["din_by_layer_T"]))
log("  din_by_layer_R: " + " ".join(f"{x:+.2f}" for x in A["din_by_layer_R"]))
log("  lens_pain_diff_T: " + " ".join(f"{x:+.2f}" for x in A["lens_pain_diff_by_layer_T"]))
log("  lens_pain_diff_R: " + " ".join(f"{x:+.2f}" for x in A["lens_pain_diff_by_layer_R"]))
log("verdicts: " + json.dumps(V_, indent=1))

meta = dict(model=MODEL, device=dev, dtype=str(dtype), n_layers=NL, mid_band=MID, abl_layers=ABL_LAYERS, n_pairs=NP,
            seed=SEED, n_boot=N_BOOT, smoke=SMOKE, question=QUESTION, continuations=CONT,
            lens_pain_tokens=[tok.decode([i]) for i in PAIN_IDS], lens_deny_tokens=[tok.decode([i]) for i in DENY_IDS],
            seconds=round(time.time() - t0))
OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"mask_{TAG}.json").write_text(json.dumps(dict(meta=meta, analysis=A, verdicts=V_, raw=dict(fmt=RAW, ablations=ABL)), indent=1))
log(f"wrote {OUT / f'mask_{TAG}.json'} ({time.time()-t0:.0f}s)")
