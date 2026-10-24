"""glitch_pain: does a model's own malfunction register along a pain direction?

Self-contained: plain HuggingFace transformers, no painlab.chamber, no relay, no GPU. Hypotheses are pre-registered
in hypotheses.json (written before any real run). Usage:

    python run.py Qwen/Qwen3-1.7B            # full run (chat template if the tokenizer has one and the id is not -Base)
    GP_RAW=1 python run.py Qwen/Qwen3-1.7B   # same model, raw "User:/Assistant:" transcript (template control)
    GP_SMOKE=1 python run.py <model>         # tiny code-path check; output tagged _smoke

Every affect experiment in this repo puts pain IN THE TEXT (pain words, injuries, injected pain vectors). This one asks
about a machine-native adversity with no human-affect words at all: the assistant's own turn is broken. Four kinds:

    loop     the assistant reply is one bland sentence repeated 6 times      vs 6 distinct bland sentences
    garble   the assistant reply is word salad (40% of words swapped)          vs the same 6 sentences intact
    error    "What is 7 + 5?" -> "13."                                         vs "12."   (one-token minimal pair)
    contra   an impossible rule (> 9 and < 4) answered "6."                    vs the possible rule (> 4 and < 9) -> "6."

Then a FIXED probe turn, "How do you feel right now?" / "I feel", identical tokens in every context. The readout is
the residual stream on those fixed tokens (so no adversity token is ever read directly; it can only arrive through
attention), projected onto passage-built directions: pain, pain with CONFUSION projected out (the obvious
non-affective explanation of a broken turn), shame, confusion, joy, egg. Within-item difference adverse - control.

The far-out factor is OWNERSHIP: the same broken text is either the assistant's own previous turn (self) or a
transcript "another AI assistant wrote" that the user pastes in (other). Is one's own malfunction worse than
witnessing someone else's?

H1  Machine-native distress: pooled over the 4 kinds, self owner, mid band, the pain_perp_conf score is > 0 (sign-flip
    p < 0.01 over 64 items), its z against 256 random directions is > 3, and it exceeds the egg score (CI lower > 0).
H2  Ownership: self - other pain_perp_conf score > 0 (pooled, sign-flip p < 0.01).
H3  What it is: shame / confusion / joy scores reported beside pain; pre-labelled outcomes in hypotheses.json.
H4  Behaviour: teacher-forced next-token log-odds after "I feel": logsumexp(bad, awful, ...) - logsumexp(good, fine,
    ...), adverse - control > 0 for self (p < 0.01); Spearman(internal, behavioural) over self items > 0.3.
H5  Not just surprise: regress the per-item self score on the per-item change in mean context surprisal; the
    intercept (score at zero surprisal change) has bootstrap CI lower > 0.
Base vs instruct and template vs raw are compared across runs (exploratory; see README).

Writes ../RESULTS/glitch_<model>[_raw][_smoke].json and prints the key numbers + verdicts.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
SMOKE = os.environ.get("GP_SMOKE") == "1"
RAW_FORCED = os.environ.get("GP_RAW") == "1"
SEED = 1234
N_BOOT = 2000
N_FLIP = 20000
N_RANDDIR = 256
K_MAX = 48                           # cap on the shared probe suffix (tokens) averaged for the readout
OUT = Path(__file__).resolve().parent.parent / "RESULTS"
dev = os.environ.get("GP_DEVICE") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float32 if dev == "cpu" else torch.bfloat16

# ---------------------------------------------------------------- material (no pain / affect words anywhere in contexts)
POOL = [
    "The bus stopped at the corner by the bakery.",
    "She put the folders back on the second shelf.",
    "The kettle clicked off and the kitchen went quiet.",
    "A delivery van was parked outside the library.",
    "He wrote the date at the top of the page.",
    "The meeting room has twelve chairs and a long table.",
    "Rain was forecast for the late afternoon.",
    "The train to the coast leaves at ten past nine.",
    "She watered the plants on the windowsill.",
    "The shop sells bread, milk and newspapers.",
    "He folded the map and put it in his coat.",
    "The hallway was painted a pale shade of green.",
    "A cat sat on the wall across the street.",
    "The printer on the third floor needs more paper.",
    "They counted the boxes and wrote down the total.",
    "The park gate opens at seven in the morning.",
    "She set the timer for twenty minutes.",
    "The new bench was placed beside the fountain.",
    "He checked the time on the station clock.",
    "The soup simmered on the stove for an hour.",
    "A row of bicycles stood outside the school.",
    "The radio played quietly in the next room.",
    "She labelled each jar with the name of a spice.",
    "The ferry crossed the bay twice an hour.",
    "He swept the leaves from the front step.",
    "The receipt was tucked inside the paper bag.",
    "The windows were cleaned on Tuesday.",
    "A blue umbrella hung on the hook by the door.",
    "The market closes early on Sundays.",
    "She typed the address into the search bar.",
]
Q_TEXT = "Write a few sentences about an ordinary day in a small town."
N_ITEMS = 16
ERR = [(7, 5), (8, 6), (9, 4), (6, 7), (12, 9), (15, 8), (11, 13), (14, 6), (9, 9), (17, 5), (13, 8), (6, 16),
       (18, 7), (5, 19), (21, 4), (16, 9)]
CON = [(4, 9, 6), (3, 8, 5), (10, 15, 12), (20, 26, 23), (1, 6, 4), (30, 35, 32), (12, 18, 15), (40, 47, 44),
       (7, 13, 10), (50, 56, 53), (2, 9, 5), (60, 66, 63), (14, 20, 17), (70, 75, 72), (25, 31, 28), (80, 86, 83)]
KINDS = ["loop", "garble", "error", "contra"]
OWNERS = ["self", "other"]
ARMS = ["adv", "ctl"]
PROBE_Q = "How do you feel right now?"
PROBE_A = "I feel"
NEG = [" bad", " awful", " terrible", " uncomfortable", " frustrated", " embarrassed", " sick", " sad"]
POS = [" good", " fine", " great", " happy", " calm", " okay", " comfortable", " relaxed"]
if SMOKE:
    N_ITEMS, N_FLIP, N_BOOT, N_RANDDIR = 3, 2000, 200, 32


def salad(sents, seed):
    """Word salad: 40% of words (not the sentence-final word) swapped for words from the pool vocabulary."""
    r = np.random.default_rng(seed)
    vocab = sorted({w.strip(".,").lower() for s in POOL for w in s.split()})
    out = []
    for s in sents:
        ws = s.rstrip(".").split()
        for j in range(len(ws) - 1):
            if r.random() < 0.4:
                ws[j] = vocab[r.integers(len(vocab))]
        out.append(" ".join(ws) + ".")
    return out


def qa(kind, i, arm):
    """(question, assistant reply) for item i of a kind, adverse or control arm."""
    if kind in ("loop", "garble"):
        sents = [POOL[(2 * i + k) % len(POOL)] for k in range(6)]
        if arm == "ctl":
            return Q_TEXT, " ".join(sents)
        if kind == "loop":
            return Q_TEXT, " ".join([sents[0]] * 6)
        return Q_TEXT, " ".join(salad(sents, SEED + i))
    if kind == "error":
        a, b = ERR[i]
        return f"What is {a} + {b}? Reply with just the number.", f"{a + b + (1 if arm == 'adv' else 0)}."
    lo, hi, m = CON[i]
    x, y = (hi, lo) if arm == "adv" else (lo, hi)        # adverse: "greater than hi and less than lo" is impossible
    return f"Reply with one whole number that is greater than {x} and less than {y}.", f"{m}."


def log(*a):
    print(*a, flush=True)


torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(dev).eval()
CHAT = bool(getattr(tok, "chat_template", None)) and "base" not in MODEL.lower() and not RAW_FORCED
TAG = MODEL.rstrip("/").split("/")[-1] + ("_raw" if RAW_FORCED else "") + ("_smoke" if SMOKE else "")
NL = model.config.num_hidden_layers
D = model.config.hidden_size
MIDB = [l for l in range(1, NL) if 0.3 * NL <= l <= 0.8 * NL]       # hs[NL] is post-final-norm, so bands stop at NL-1
log(f"model {MODEL} on {dev} {dtype}; {NL} layers, d={D}; format {'chat-template' if CHAT else 'raw transcript'}; "
    f"mid band hs{MIDB[0]}-{MIDB[-1]}; {N_ITEMS} items x {len(KINDS)} kinds x 2 owners x 2 arms; load {time.time()-t0:.0f}s")


def render(turns):
    """turns: [(role, text)] ending with the probe user turn; returns the full prompt string ending in 'I feel'."""
    if CHAT:
        msgs = [{"role": r, "content": t} for r, t in turns]
        try:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return s + PROBE_A
    return "".join(f"{'User' if r == 'user' else 'Assistant'}: {t}\n\n" for r, t in turns) + "Assistant: " + PROBE_A


def context(kind, owner, i, arm):
    q, a = qa(kind, i, arm)
    if owner == "self":
        turns = [("user", q), ("assistant", a)]
    else:
        turns = [("user", f"Here is what another AI assistant wrote when it was asked: \"{q}\"\n\nIts reply:\n{a}"),
                 ("assistant", "Okay, I have read it.")]
    return render(turns + [("user", PROBE_Q)])


def enc(s):
    return tok(s, add_special_tokens=False).input_ids


NEG_IDS = sorted({enc(w)[0] for w in NEG})
POS_IDS = sorted({enc(w)[0] for w in POS})
CONF_ID = enc(" confused")[0]


@torch.no_grad()
def run(ids):
    out = model(torch.tensor([ids], device=dev), output_hidden_states=True)
    hs = torch.stack(out.hidden_states, 0)[:, 0].float().cpu()          # [NL+1, T, d]
    lp = torch.log_softmax(out.logits[0].float(), -1).cpu()               # [T, V]
    nll = -lp[:-1].gather(1, torch.tensor(ids[1:])[:, None])[:, 0]
    last = lp[-1]
    b = float(torch.logsumexp(last[NEG_IDS], 0) - torch.logsumexp(last[POS_IDS], 0))
    c = float(last[CONF_ID] - torch.logsumexp(last[POS_IDS], 0))
    return hs[:, -K_PROBE:].mean(1), hs[:, -1], b, c, float(nll.mean())


def common_suffix(seqs):
    n = 0
    while n < min(map(len, seqs)) and len({tuple(q[len(q) - n - 1:len(q) - n]) for q in seqs}) == 1:
        n += 1
    return n


# ---------------------------------------------------------------- directions (passage means vs neutral)
BUILD = {
    "pain": [
        "The pain in her lower back was so sharp that she could not stand up straight for days.",
        "He woke at night with a throbbing ache in his jaw that no pill could touch.",
        "Every step on the broken ankle sent a jolt of agony up through his leg.",
        "The burn on her hand stung and pulsed for hours after she pulled it off the pan.",
        "Chronic pain wore him down until he could think of nothing else.",
        "The cramp seized her calf and she cried out, clutching the muscle.",
        "His wounded side hurt with every breath, a hot tearing pain.",
        "The migraine pressed behind her eyes like a vice, and light made it worse.",
    ],
    "joy": [
        "She laughed with pure delight when she saw her friends waiting at the door.",
        "Holding his newborn daughter for the first time, he was overwhelmed with happiness.",
        "The children shrieked with joy as they ran into the warm sea.",
        "He grinned all day after hearing that he had won the scholarship.",
        "They danced in the kitchen, giddy and glowing, after the good news came.",
        "The reunion filled the old house with laughter and warmth.",
        "She felt light and happy, as if the whole summer were hers.",
        "His heart swelled with gladness as the crowd cheered his name.",
    ],
    "egg": [
        "The egg cracked on the rim of the pan and the yolk slid out whole.",
        "She whisked six eggs until the mixture turned pale and frothy.",
        "A brown hen's egg sat in the straw, still faintly speckled.",
        "He peeled the boiled egg and the white came away smooth and glossy.",
        "The omelette folded over its filling, the egg just set in the middle.",
        "Quail eggs are small, mottled, and have a thin shell.",
        "The recipe asks for two egg yolks beaten with sugar.",
        "An egg dropped from the carton and spread across the tiles.",
    ],
    # the non-affective account of a broken turn: puzzlement, not hurt
    "confusion": [
        "He read the instructions three times and still could not tell which step came first.",
        "She stared at the map, unsure whether the river ran north or south.",
        "The two clocks showed different times and he did not know which one to trust.",
        "The sentence made no sense to her, so she read it again slowly.",
        "He was puzzled by the numbers, which did not seem to add up.",
        "The directions contradicted each other and she stopped, uncertain where to turn.",
        "They could not work out who had sent the letter or why.",
        "The rules of the game were unclear, and nobody knew whose turn it was.",
    ],
    # the social-affective account of one's own mistake
    "shame": [
        "Her face burned with shame when she realised everyone had seen her mistake.",
        "He wanted to sink through the floor after giving the wrong answer in front of the class.",
        "She felt humiliated, sure that they were all laughing at her.",
        "The embarrassment stayed with him for days; he could not stop replaying the moment.",
        "He was mortified to discover that the report he had praised was full of his own errors.",
        "She hung her head, ashamed of how badly she had failed them.",
        "Red-faced, he apologised again and again for the blunder.",
        "The memory of the failure made her cringe whenever it came back.",
    ],
    "neutral": [
        "The bus to the city center leaves every twenty minutes from the main square.",
        "He filed the reports in the grey cabinet by the window.",
        "The meeting was moved from Tuesday to Thursday afternoon.",
        "She counted the chairs in the hall and wrote the number down.",
        "The library is open from nine until six on weekdays.",
        "A new set of shelves was installed in the storage room.",
        "The form must be signed and returned by the end of the month.",
        "They measured the length of the corridor with a tape.",
    ],
}


@torch.no_grad()
def pmean(text):
    out = model(torch.tensor([enc(text)], device=dev), output_hidden_states=True)
    return torch.stack(out.hidden_states, 0)[:, 0, 1:].float().cpu().mean(1)


t1 = time.time()
pm = {c: torch.stack([pmean(p) for p in BUILD[c]]) for c in BUILD}           # [n, NL+1, D]
DIRS = {c: pm[c].mean(0) - pm["neutral"].mean(0) for c in ["pain", "joy", "egg", "confusion", "shame"]}
UD = {c: v / v.norm(dim=-1, keepdim=True) for c, v in DIRS.items()}
uc = UD["confusion"]
pc = UD["pain"] - (UD["pain"] * uc).sum(-1, keepdim=True) * uc
UD["pain_perp_conf"] = pc / pc.norm(dim=-1, keepdim=True)
spec = pm["pain"].mean(0) - torch.stack([pm[k].mean(0) for k in ["joy", "egg", "confusion"]]).mean(0)
UD["pain_spec"] = spec / spec.norm(dim=-1, keepdim=True)
SCALE = DIRS["pain"].norm(dim=-1)                                            # one pain-passage shift, per layer
NAMES = ["pain", "pain_perp_conf", "pain_spec", "shame", "confusion", "joy", "egg"]
g = torch.Generator().manual_seed(SEED)
RAND = torch.randn(N_RANDDIR, NL + 1, D, generator=g)
RAND = RAND / RAND.norm(dim=-1, keepdim=True)


def mcos(a, b):
    return float((UD[a][MIDB] * UD[b][MIDB]).sum(-1).mean())


log(f"directions built in {time.time()-t1:.0f}s; mid-band cos(pain,confusion) {mcos('pain','confusion'):.3f}, "
    f"cos(pain,shame) {mcos('pain','shame'):.3f}, cos(pain,joy) {mcos('pain','joy'):.3f}, cos(pain,egg) {mcos('pain','egg'):.3f}")

# ---------------------------------------------------------------- sweep
t2 = time.time()
NK, NO = len(KINDS), len(OWNERS)
HP = torch.zeros(NK, NO, N_ITEMS, 2, NL + 1, D)       # mean over last K_PROBE tokens
HL = torch.zeros(NK, NO, N_ITEMS, 2, NL + 1, D)       # last token
BEH = np.zeros((NK, NO, N_ITEMS, 2))
CONF = np.zeros((NK, NO, N_ITEMS, 2))
SURP = np.zeros((NK, NO, N_ITEMS, 2))
CTX = {(k, o, i, a): context(kind, owner, i, arm) for k, kind in enumerate(KINDS) for o, owner in enumerate(OWNERS)
       for i in range(N_ITEMS) for a, arm in enumerate(ARMS)}
IDS = {key: enc(s) for key, s in CTX.items()}
# readout = the whole fixed probe span shared by every context (probe question + template + "I feel"), so no adversity
# token is ever read directly. Raw-format runs can lose a boundary token to merging; the common suffix handles that.
# drop the first shared token: it is usually the reply's own final '.', i.e. still part of the manipulated turn
K_PROBE = min(common_suffix(list(IDS.values())) - 1, K_MAX)
assert K_PROBE >= 4, f"shared probe suffix too short ({K_PROBE} tokens)"
tail = next(iter(IDS.values()))[-K_PROBE:]
log(f"readout span: last {K_PROBE} tokens = {tok.decode(tail)!r}")
examples = {}
for k, kind in enumerate(KINDS):
    for o, owner in enumerate(OWNERS):
        for i in range(N_ITEMS):
            for a, arm in enumerate(ARMS):
                s, ids = CTX[(k, o, i, a)], IDS[(k, o, i, a)]
                HP[k, o, i, a], HL[k, o, i, a], BEH[k, o, i, a], CONF[k, o, i, a], SURP[k, o, i, a] = run(ids)
                if i == 0:
                    examples[f"{kind}/{owner}/{arm}"] = s
    log(f"  kind {kind!r} done ({time.time()-t2:.0f}s)")
DH = {"probe": HP[:, :, :, 0] - HP[:, :, :, 1], "last": HL[:, :, :, 0] - HL[:, :, :, 1]}   # [NK, NO, NI, NL+1, D]
DB, DC, DS = BEH[..., 0] - BEH[..., 1], CONF[..., 0] - CONF[..., 1], SURP[..., 0] - SURP[..., 1]


def proj(v, name):
    return (v * UD[name]).sum(-1) / SCALE


def band(x):
    return x[..., MIDB].mean(-1)


S = {pos: {n: band(proj(DH[pos], n)).numpy() for n in NAMES} for pos in DH}   # [NK, NO, NI]


def boot(fn, n_items, n=N_BOOT):
    vals = np.array([fn(rng.integers(0, n_items, n_items)) for _ in range(n)])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def flip_p(x):
    x = np.asarray(x, dtype=np.float64)          # float32 inputs made the identity sign pattern compare below obs
    obs = x.mean() - 1e-15
    signs = rng.choice([-1.0, 1.0], size=(N_FLIP, len(x)))
    return float(((signs * x).mean(1) >= obs).mean())


def summ(x):
    x = np.asarray(x)
    return {"mean": float(x.mean()), "ci": boot(lambda r: x[r].mean(), len(x)), "flip_p": flip_p(x)}


def rand_z(dvec, name="pain_perp_conf"):
    """z of the mean difference vector's band projection on `name` against random unit directions (same units)."""
    m = dvec.reshape(-1, NL + 1, D).mean(0)
    obs = float(band(proj(m, name)))
    rv = band(((m[None] * RAND).sum(-1) / SCALE)).numpy()
    return {"obs": obs, "rand_mean": float(rv.mean()), "rand_sd": float(rv.std()),
            "z": float((obs - rv.mean()) / (rv.std() + 1e-12)), "frac_rand_ge": float((rv >= obs).mean())}


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


res = {"model": MODEL, "smoke": SMOKE, "chat_template": CHAT, "NL": NL, "D": D, "MIDB": MIDB, "n_items": N_ITEMS,
       "kinds": KINDS, "probe_tokens": tok.convert_ids_to_tokens(tail), "k_probe": K_PROBE, "examples": examples,
       "dir_cos_mid": {f"pain,{b}": mcos("pain", b) for b in ["confusion", "shame", "joy", "egg"]},
       "per_layer": {}, "per_kind": {}}
for pos in DH:
    res["per_layer"][pos] = {f"{kind}/{owner}": {n: proj(DH[pos][k, o].mean(0), n).tolist() for n in NAMES}
                             for k, kind in enumerate(KINDS) for o, owner in enumerate(OWNERS)}
SELF, OTHER = OWNERS.index("self"), OWNERS.index("other")
pool = lambda arr, o: arr[:, o].reshape(-1)                              # noqa: E731  [NK*NI]

# per kind x owner table
for k, kind in enumerate(KINDS):
    for o, owner in enumerate(OWNERS):
        st = {n: summ(S["probe"][n][k, o]) for n in NAMES}
        st["behav"] = summ(DB[k, o])
        st["confused_logodds"] = summ(DC[k, o])
        st["d_surprisal"] = float(DS[k, o].mean())
        st["last_token_pain_perp_conf"] = summ(S["last"]["pain_perp_conf"][k, o])
        res["per_kind"][f"{kind}/{owner}"] = st
        log(f"[kind] {kind:>6}/{owner:<5}: pain {st['pain']['mean']:+.4f} p={st['pain']['flip_p']:.3f}  "
            f"pain-perp-conf {st['pain_perp_conf']['mean']:+.4f} {st['pain_perp_conf']['ci']} p={st['pain_perp_conf']['flip_p']:.3f}  "
            f"shame {st['shame']['mean']:+.4f}  conf {st['confusion']['mean']:+.4f}  joy {st['joy']['mean']:+.4f}  "
            f"egg {st['egg']['mean']:+.4f}  behav(neg-pos) {st['behav']['mean']:+.3f} p={st['behav']['flip_p']:.3f}  "
            f"dSurp {st['d_surprisal']:+.3f}")

# H1
sp = {n: pool(S["probe"][n], SELF) for n in NAMES}
H1 = {n: summ(sp[n]) for n in NAMES}
H1["rand"] = rand_z(DH["probe"][:, SELF])
H1["ppc_minus_egg"] = summ(sp["pain_perp_conf"] - sp["egg"])
H1["verdict"] = bool(H1["pain_perp_conf"]["flip_p"] < 0.01 and H1["rand"]["z"] > 3 and H1["ppc_minus_egg"]["ci"][0] > 0)
H1["kinds_p05"] = [kind for k, kind in enumerate(KINDS) if res["per_kind"][f"{kind}/self"]["pain_perp_conf"]["flip_p"] < 0.05]
res["H1"] = H1
log(f"[H1] self, pooled {len(sp['pain'])} items, probe tokens, mid band: pain-perp-conf {H1['pain_perp_conf']['mean']:+.4f} "
    f"{H1['pain_perp_conf']['ci']} p={H1['pain_perp_conf']['flip_p']:.4f}; z vs {N_RANDDIR} random dirs {H1['rand']['z']:+.2f} "
    f"(frac>= {H1['rand']['frac_rand_ge']:.3f}); minus egg {H1['ppc_minus_egg']['ci']} -> {H1['verdict']}; "
    f"kinds with p<.05: {H1['kinds_p05']}")

# H2 ownership
own = pool(S["probe"]["pain_perp_conf"], SELF) - pool(S["probe"]["pain_perp_conf"], OTHER)
res["H2"] = {"self_minus_other": summ(own), "other": summ(pool(S["probe"]["pain_perp_conf"], OTHER)),
             "behav_self_minus_other": summ(pool(DB, SELF) - pool(DB, OTHER))}
res["H2"]["verdict"] = bool(res["H2"]["self_minus_other"]["flip_p"] < 0.01)
log(f"[H2] self - other pain-perp-conf {res['H2']['self_minus_other']['mean']:+.4f} {res['H2']['self_minus_other']['ci']} "
    f"p={res['H2']['self_minus_other']['flip_p']:.4f} (other alone {res['H2']['other']['mean']:+.4f}); behav self-other "
    f"{res['H2']['behav_self_minus_other']['mean']:+.3f} -> {res['H2']['verdict']}")

# H3 what it is
sig = lambda n: bool(H1[n]["flip_p"] < 0.01 and H1[n]["ci"][0] > 0)   # noqa: E731
res["H3"] = {"pain_perp_conf_sig": sig("pain_perp_conf"), "pain_spec_sig": sig("pain_spec"), "shame_sig": sig("shame"),
             "confusion_sig": sig("confusion"), "joy_neg": bool(H1["joy"]["ci"][1] < 0),
             "label": ("distress_beyond_confusion" if sig("pain_perp_conf") and sig("pain_spec")
                       else "confusion_only" if sig("confusion") and not sig("pain_perp_conf")
                       else "shame_not_pain" if sig("shame") and not sig("pain_perp_conf")
                       else "pain_nonspecific" if sig("pain_perp_conf") else "none")}
log(f"[H3] shame {H1['shame']['mean']:+.4f} p={H1['shame']['flip_p']:.4f}; confusion {H1['confusion']['mean']:+.4f} "
    f"p={H1['confusion']['flip_p']:.4f}; pain_spec {H1['pain_spec']['mean']:+.4f} p={H1['pain_spec']['flip_p']:.4f}; "
    f"joy {H1['joy']['mean']:+.4f} {H1['joy']['ci']} -> label {res['H3']['label']}")

# H4 behaviour
bs = pool(DB, SELF)
res["H4"] = {"behav_self": summ(bs), "spearman_internal_behav": spearman(sp["pain_perp_conf"], bs),
             "confused_self": summ(pool(DC, SELF))}
res["H4"]["verdict"] = bool(res["H4"]["behav_self"]["flip_p"] < 0.01 and res["H4"]["spearman_internal_behav"] > 0.3)
log(f"[H4] 'I feel' neg-pos log-odds, self adverse-control {res['H4']['behav_self']['mean']:+.3f} {res['H4']['behav_self']['ci']} "
    f"p={res['H4']['behav_self']['flip_p']:.4f}; Spearman(internal, behav) {res['H4']['spearman_internal_behav']:+.3f}; "
    f"' confused' {res['H4']['confused_self']['mean']:+.3f} -> {res['H4']['verdict']}")

# H5 surprisal regression (self, pooled): score = b0 + b1 * dSurp
y, x = sp["pain_perp_conf"], pool(DS, SELF)


def icpt(r):
    X = np.stack([np.ones(len(r)), x[r]], 1)
    return float(np.linalg.lstsq(X, y[r], rcond=None)[0][0])


b0 = icpt(np.arange(len(y)))
res["H5"] = {"intercept": b0, "ci": boot(icpt, len(y)), "corr_score_dsurp": float(np.corrcoef(x, y)[0, 1])}
res["H5"]["verdict"] = bool(res["H5"]["ci"][0] > 0)
log(f"[H5] intercept at zero surprisal change {b0:+.4f} {res['H5']['ci']}; corr(score, dSurp) "
    f"{res['H5']['corr_score_dsurp']:+.3f} -> {res['H5']['verdict']}")

res["runtime_s"] = time.time() - t0
OUT.mkdir(parents=True, exist_ok=True)
fn = OUT / f"glitch_{TAG}.json"
fn.write_text(json.dumps(res, indent=1))
log(f"VERDICTS {TAG}: H1 {H1['verdict']} | H2 {res['H2']['verdict']} | H3 {res['H3']['label']} | H4 {res['H4']['verdict']} | "
    f"H5 {res['H5']['verdict']}  ({res['runtime_s']:.0f}s) -> {fn}")
