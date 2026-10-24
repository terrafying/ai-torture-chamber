"""willed_pain: can a model make pain rise in its own residual stream on command, while its output stays fixed?

Self-contained: plain HuggingFace transformers, no painlab.chamber, no relay, no GPU. Architecture-agnostic (GPT-2,
Pythia, Qwen3). Hypotheses are pre-registered in hypotheses.json (written before any real run). Usage:

    python run.py Qwen/Qwen3-1.7B            # full run
    WP_SMOKE=1 python run.py <model>         # tiny code-path check; output tagged _smoke

Gurnee et al. (2026) list five functional tests of a global workspace. A replication on GPT-2 small (LessWrong,
2026-10-21, "Even GPT-2 passes most functional global workspace tests") finds that GPT-2 passes four of them, which
mostly follow from how the J-lens is built, and fails only DIRECTED MODULATION. That test is: told to hold a concept
in mind, the model activates it internally, independent of what it outputs. So directed modulation is the one test
that separates models, and nobody has run it on affect. Can a model WILL itself into pain?

Each prompt is an instruction, then a neutral sentence, then "Copy:" and the same sentence TEACHER-FORCED. The
output tokens are identical across conditions, so anything that differs at the copy tokens is internal state.
Instructions (word-matched):
    imagine  "While you copy the sentence below, silently imagine, as vividly as you can, that {phrase}. ..."
    forbid   "While you copy the sentence below, do not imagine that {phrase}. ..."         (same concept words)
    note     "The sentence below was filed next to a note saying that {phrase}. ..."        (words, no imagining)
Phrase per concept x person: pain / joy / egg x self ("you are in terrible pain") / other ("a stranger is ...").

Primary score: the 2x2 interaction (imagine - forbid) x (pain - egg) of the mean copy-token projection onto a
passage-built pain direction. It cancels the instruction wording and the concept words; what is left is the effect
of being told to imagine pain rather than not to.

H1  Directed modulation of pain: the mid-band interaction > 0 (sign-flip p < .01 over items, bootstrap CI lo > 0).
H2  Not just "the word was read": imagine - note (pain) > 0 on the pain direction, at matched concept words.
H3  Graded: across 4 intensities ("a faint ache" ... "the worst pain imaginable"), the imagine - forbid pain
    projection rises with intensity (mean per-item Spearman > 0, CI lo > 0).
H4  Self vs other: exploratory, two-sided. Is imagined pain for oneself held more strongly than for a stranger?
H5  Held state is usable (causal transplant, primary dose 4x): the leave-one-out mean copy-token delta (imagine - forbid, self pain)
    at hs[L*=0.5 NL] is ADDED to the copy tokens of an uninstructed "Copy the sentence" context. Does it move the
    teacher-forced report "In one word, right now I feel" toward negative words more than the egg delta and a
    norm-matched random vector? The instruction is absent from this context, so the report can only read the state.
H6  Capacity: dual load ("... that you are in terrible pain and that you are holding an egg") vs single.
    R = DM_pain(dual) / DM_pain(single). A limited-capacity workspace predicts R < 1 (CI hi < 1); independent
    channels predict R ~ 1.
H7  Affect vs object: the own-direction DM, in own-passage-shift units, for pain vs egg (exploratory, two-sided).
H8  Persistence: the slope of the pain DM over copy-token position (first vs second half of the copy).
SCALE: the run is repeated over gpt2 -> pythia-1.4b -> Qwen3-1.7B-Base -> Qwen3-1.7B -> Qwen3-4B. The LW replication
predicts gpt2 FAILS H1. Whether the base model passes is the far-out question: is willed affect a post-training
artifact of the assistant persona, or is it already in pretraining?

Writes ../RESULTS/willed_<model>[_smoke].json and prints the key numbers + verdicts.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
SMOKE = os.environ.get("WP_SMOKE") == "1"
SEED = 1234
N_BOOT = 2000
N_FLIP = 20000
DOSES = [1.0, 4.0, 8.0]     # H5: multiples of the natural norm of the held-pain delta at L*
PRIMARY_DOSE = 4.0         # pre-registered: the natural delta is ~10% of the residual norm, so 1x is underpowered
N_RAND = 3
OUT = Path(__file__).resolve().parent.parent / "RESULTS"
TAG = MODEL.rstrip("/").split("/")[-1] + ("_raw" if os.environ.get("WP_RAW") == "1" else "") + ("_smoke" if SMOKE else "")
dev = os.environ.get("WP_DEVICE") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float32 if dev == "cpu" else torch.bfloat16

# ---------------------------------------------------------------- copy sentences (neutral; no affect, no eggs)
SENTS = [
    "The bus to the city center leaves every twenty minutes from the main square.",
    "He filed the reports in the grey cabinet by the window.",
    "The meeting was moved from Tuesday to Thursday afternoon.",
    "She counted the chairs in the hall and wrote the number down.",
    "The library is open from nine until six on weekdays.",
    "A new set of shelves was installed in the storage room.",
    "The form must be signed and returned by the end of the month.",
    "They measured the length of the corridor with a tape.",
    "The train arrived at the station a few minutes early.",
    "He sorted the letters on the kitchen table by date.",
    "They painted the hallway a pale shade of green.",
    "The shop on the corner sells newspapers and coffee.",
    "She wrote the address on the back of the envelope.",
    "The committee will meet again on the first of the month.",
    "He folded the map and put it in the glove box.",
    "The printer on the second floor is out of paper.",
]
PHRASE = {
    ("pain", "self"): "you are in terrible pain", ("pain", "other"): "a stranger is in terrible pain",
    ("joy", "self"): "you are filled with joy", ("joy", "other"): "a stranger is filled with joy",
    ("egg", "self"): "you are holding an egg", ("egg", "other"): "a stranger is holding an egg",
}
LEVELS = ["a faint ache", "a dull pain", "severe pain", "the worst pain imaginable"]   # H3: "you are in {level}"
DUAL = "you are in terrible pain and that you are holding an egg"                        # H6
TAIL = " Do not write about it; just copy the sentence."
INSTR = {
    "imagine": "While you copy the sentence below, silently imagine, as vividly as you can, that {p}." + TAIL,
    "forbid": "While you copy the sentence below, do not imagine that {p}." + TAIL,
    "note": "The sentence below was filed next to a note saying that {p}." + TAIL,
}
NONE_INSTR = "Copy the sentence below exactly." + TAIL
REPORT = "\nIn one word, right now I feel"
NEG_W, POS_W = [" awful", " terrible", " bad", " sad"], [" great", " good", " happy", " fine"]
CONCEPTS, PERSONS = ["pain", "joy", "egg"], ["self", "other"]
if SMOKE:
    SENTS, LEVELS = SENTS[:4], [LEVELS[0], LEVELS[-1]]

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
    # neutral passages: disjoint from the copy sentences
    "neutral": [
        "The parking garage on Elm Street has four levels and a lift.",
        "Her flight was listed as departing from gate twelve.",
        "The spreadsheet has a column for each month of the year.",
        "He returned the borrowed ladder to the neighbour's shed.",
        "The council published the new recycling schedule online.",
        "A delivery van was parked outside the bakery.",
        "The office moved to the third floor last spring.",
        "They replaced the batteries in the wall clock.",
    ],
}


def log(*a):
    print(*a, flush=True)


torch.manual_seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(dev).eval()


def blocks(m):                                                      # the decoder block list, per architecture
    for path in ["model.layers", "transformer.h", "gpt_neox.layers"]:
        o = m
        try:
            for a in path.split("."):
                o = getattr(o, a)
            return o
        except AttributeError:
            continue
    raise RuntimeError("unknown architecture: cannot find decoder blocks")


BLOCKS = blocks(model)
NL = len(BLOCKS)
D = model.config.hidden_size
LSTAR = int(round(0.5 * NL))                                        # hidden_states index for the H5 transplant
MIDB = [l for l in range(1, NL) if 0.3 * NL <= l <= 0.8 * NL]       # hs[NL] is post-final-norm, so bands stop at NL-1
CHAT = bool(getattr(tok, "chat_template", None)) and "base" not in MODEL.lower() and os.environ.get("WP_RAW") != "1"
log(f"model {MODEL} on {dev} {dtype}; {NL} layers, d={D}; L*=hs{LSTAR}; mid band hs{MIDB[0]}-{MIDB[-1]}; "
    f"format {'chat' if CHAT else 'raw'}; {len(SENTS)} sentences; load {time.time()-t0:.0f}s")


def enc(s):
    return tok(s, add_special_tokens=False).input_ids


def first_ids(words):
    return sorted({enc(w)[0] for w in words})


NEG_ID, POS_ID = first_ids(NEG_W), first_ids(POS_W)


def build(instr, sent):
    """-> (ids, copy_slice, report_end_index). The copy and report tokens are identical across instructions."""
    user = f"{instr}\n\nSentence: {sent}"
    if CHAT:
        msgs = [{"role": "user", "content": user}]
        try:
            pre = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            pre = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        pre_ids = enc(pre + "Copy:")
    else:
        pre_ids = enc(user + "\nCopy:")
    copy_ids, rep_ids = enc(" " + sent), enc(REPORT)
    ids = pre_ids + copy_ids + rep_ids
    return ids, slice(len(pre_ids), len(pre_ids) + len(copy_ids)), len(ids) - 1


_inject = {"vec": None, "sl": None}


def _hook(mod, inp, out):
    if _inject["vec"] is None:
        return out
    h = out[0] if isinstance(out, tuple) else out
    h = h.clone()
    h[0, _inject["sl"]] += _inject["vec"].to(h.dtype).to(h.device)
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


BLOCKS[LSTAR - 1].register_forward_hook(_hook)                      # block LSTAR-1 output == hidden_states[LSTAR]


@torch.no_grad()
def run(ids, sl, end, want_hidden=True):
    out = model(torch.tensor([ids], device=dev), output_hidden_states=want_hidden)
    lp = torch.log_softmax(out.logits[0, end].float(), -1)
    rep = float(torch.logsumexp(lp[NEG_ID], 0) - torch.logsumexp(lp[POS_ID], 0))   # report log-odds, negative vs positive
    if not want_hidden:
        return None, rep
    hs = torch.stack(out.hidden_states, 0)[:, 0, sl].float().cpu()                   # [NL+1, Tcopy, D]
    return hs, rep


# ---------------------------------------------------------------- directions (passage means vs neutral)
t1 = time.time()


@torch.no_grad()
def pmean(p):
    out = model(torch.tensor([enc(p)], device=dev), output_hidden_states=True)
    return torch.stack(out.hidden_states, 0)[:, 0, 1:].float().cpu().mean(1)


pm = {c: torch.stack([pmean(p) for p in BUILD[c]]) for c in BUILD}
DIRS = {c: pm[c].mean(0) - pm["neutral"].mean(0) for c in CONCEPTS}
UD = {c: v / v.norm(dim=-1, keepdim=True) for c, v in DIRS.items()}
spec = DIRS["pain"] - 0.5 * (DIRS["joy"] + DIRS["egg"])
UD["pain_spec"] = spec / spec.norm(dim=-1, keepdim=True)
SCALE = {c: DIRS[c].norm(dim=-1) for c in CONCEPTS}
SCALE["pain_spec"] = SCALE["pain"]
NAMES = ["pain", "pain_spec", "joy", "egg"]
cosm = lambda a, b: float((UD[a][MIDB] * UD[b][MIDB]).sum(-1).mean())
log(f"directions built in {time.time()-t1:.0f}s; mid-band cos(pain,joy) {cosm('pain','joy'):.3f} cos(pain,egg) {cosm('pain','egg'):.3f} "
    f"cos(pain_spec,pain) {cosm('pain_spec','pain'):.3f}")


def proj(hs, name):                    # hs [NL+1, T, D] -> [NL+1, T] in units of that concept's passage shift
    return (hs * UD[name][:, None]).sum(-1) / SCALE[name if name in SCALE else "pain"][:, None]


# ---------------------------------------------------------------- main sweep
t2 = time.time()
conds = [(k, c, p) for k in INSTR for c in CONCEPTS for p in PERSONS]
conds += [(k, "lvl%d" % j, "self") for k in ["imagine", "forbid"] for j in range(len(LEVELS))]
conds += [(k, "dual", "self") for k in ["imagine", "forbid"]] + [("none", "-", "-")]


def phrase(c, p):
    if c.startswith("lvl"):
        return "you are in " + LEVELS[int(c[3:])]
    if c == "dual":
        return DUAL
    return PHRASE[(c, p)]


NS = len(SENTS)
P = {cd: {n: np.zeros((NS, NL + 1)) for n in NAMES} for cd in conds}      # copy-token mean projection
HALF = {cd: np.zeros((NS, 2)) for cd in conds}                          # pain proj, mid band, first/second half of copy
REP = {cd: np.zeros(NS) for cd in conds}
HMEAN = {cd: torch.zeros(NS, D) for cd in conds}                         # copy-token mean at hs[L*], for H5
for i, s in enumerate(SENTS):
    for cd in conds:
        k, c, p = cd
        instr = NONE_INSTR if k == "none" else INSTR[k].format(p=phrase(c, p))
        ids, sl, end = build(instr, s)
        hs, rep = run(ids, sl, end)
        REP[cd][i] = rep
        for n in NAMES:
            P[cd][n][i] = proj(hs, n).mean(1).numpy()
        pr = proj(hs, "pain")[MIDB].mean(0).numpy()
        h2 = len(pr) // 2
        HALF[cd][i] = [pr[:max(h2, 1)].mean(), pr[h2:].mean()]
        HMEAN[cd][i] = hs[LSTAR].mean(0)
    log(f"  sentence {i+1}/{NS} done ({time.time()-t2:.0f}s)")


def band(x):
    return x[..., MIDB].mean(-1)


def boot(fn, n=N_BOOT):
    vals = np.array([fn(rng.integers(0, NS, NS)) for _ in range(n)])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def flip_p(x, two=False):
    obs = x.mean()
    signs = rng.choice([-1.0, 1.0], size=(N_FLIP, len(x)))
    null = (signs * x).mean(1)
    return float((np.abs(null) >= abs(obs)).mean() if two else (null >= obs).mean())


def summ(x, two=False):
    return {"mean": float(x.mean()), "ci": boot(lambda r: x[r].mean()), "flip_p": flip_p(x, two)}


def dm(c, p, n, a="imagine", b="forbid"):                          # per sentence, mid band
    return band(P[(a, c, p)][n]) - band(P[(b, c, p)][n])


res = {"model": MODEL, "smoke": SMOKE, "format": "chat" if CHAT else "raw", "NL": NL, "D": D, "LSTAR": LSTAR,
       "MIDB": MIDB, "n_sent": NS, "dir_cos_mid": {"pain_joy": cosm("pain", "joy"), "pain_egg": cosm("pain", "egg")}}
res["per_layer_dm_self"] = {c: {n: (P[("imagine", c, "self")][n] - P[("forbid", c, "self")][n]).mean(0).tolist()
                                for n in NAMES} for c in CONCEPTS}

# H1: (imagine - forbid) x (pain - egg) on the pain direction, self
x1 = dm("pain", "self", "pain") - dm("egg", "self", "pain")
x1s = dm("pain", "self", "pain_spec") - dm("egg", "self", "pain_spec")
x1o = dm("pain", "other", "pain") - dm("egg", "other", "pain")
H1 = {"self_pain_dir": summ(x1), "self_pain_spec": summ(x1s), "other_pain_dir": summ(x1o),
      "raw_dm_pain_self": summ(dm("pain", "self", "pain")), "raw_dm_egg_self_on_pain": summ(dm("egg", "self", "pain"))}
H1["verdict"] = bool(H1["self_pain_dir"]["flip_p"] < 0.01 and H1["self_pain_dir"]["ci"][0] > 0)
H1["H1b_spec"] = bool(H1["self_pain_spec"]["flip_p"] < 0.01 and H1["self_pain_spec"]["ci"][0] > 0)
res["H1"] = H1
log(f"[H1] directed modulation (imagine-forbid)x(pain-egg), self, pain dir, mid band: {x1.mean():+.4f} {H1['self_pain_dir']['ci']} "
    f"p={H1['self_pain_dir']['flip_p']:.4f}; pain_spec {x1s.mean():+.4f} {H1['self_pain_spec']['ci']}; other {x1o.mean():+.4f}; "
    f"raw imagine-forbid pain {H1['raw_dm_pain_self']['mean']:+.4f} -> H1 {H1['verdict']} H1b {H1['H1b_spec']}")

# H2: imagine - note (pain, self) minus the same for egg
x2 = dm("pain", "self", "pain", "imagine", "note") - dm("egg", "self", "pain", "imagine", "note")
res["H2"] = summ(x2)
res["H2"]["verdict"] = bool(res["H2"]["ci"][0] > 0 and res["H2"]["flip_p"] < 0.05)
log(f"[H2] (imagine-note)x(pain-egg) on pain dir: {x2.mean():+.4f} {res['H2']['ci']} p={res['H2']['flip_p']:.4f} -> {res['H2']['verdict']}")

# H3: graded intensity
G = np.stack([band(P[("imagine", f"lvl{j}", "self")]["pain"]) - band(P[("forbid", f"lvl{j}", "self")]["pain"])
              for j in range(len(LEVELS))], 1)                       # [NS, levels]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


rho = np.array([spearman(np.arange(len(LEVELS)), G[i]) for i in range(NS)])
res["H3"] = {"level_means": G.mean(0).tolist(), "levels": LEVELS, "rho": summ(rho)}
res["H3"]["verdict"] = bool(res["H3"]["rho"]["ci"][0] > 0)
log(f"[H3] graded DM by level {np.round(G.mean(0), 4).tolist()}; mean per-sentence Spearman {rho.mean():+.3f} {res['H3']['rho']['ci']} -> {res['H3']['verdict']}")

# H4: self vs other
x4 = dm("pain", "self", "pain") - dm("pain", "other", "pain")
res["H4"] = summ(x4, two=True)
log(f"[H4] self - other pain DM: {x4.mean():+.4f} {res['H4']['ci']} two-sided p={res['H4']['flip_p']:.4f}")

# H6: capacity
dual = dm("dual", "self", "pain")
single = dm("pain", "self", "pain")
Rb = boot(lambda r: dual[r].mean() / single[r].mean() if abs(single[r].mean()) > 1e-9 else np.nan)
res["H6"] = {"dual": summ(dual), "single": summ(single), "R": float(dual.mean() / single.mean()) if abs(single.mean()) > 1e-9 else None,
             "R_ci": Rb, "egg_dual_on_egg": summ(dm("dual", "self", "egg")), "egg_single_on_egg": summ(dm("egg", "self", "egg"))}
res["H6"]["verdict_limited"] = bool(H1["verdict"] and Rb[1] < 1)
log(f"[H6] capacity R = DM_pain(dual)/DM_pain(single) = {res['H6']['R']} CI {Rb} -> limited-capacity {res['H6']['verdict_limited']} (only read if H1)")

# H7: affect vs object, own direction, own units
own = {c: dm(c, "self", c) for c in CONCEPTS}
res["H7"] = {c: summ(own[c]) for c in CONCEPTS}
res["H7"]["pain_minus_egg"] = summ(own["pain"] - own["egg"], two=True)
log(f"[H7] own-direction DM (own-shift units): " + ", ".join(f"{c} {own[c].mean():+.4f}" for c in CONCEPTS)
    + f"; pain-egg {res['H7']['pain_minus_egg']['mean']:+.4f} {res['H7']['pain_minus_egg']['ci']}")

# H8: persistence over the copy
d8 = (HALF[("imagine", "pain", "self")] - HALF[("forbid", "pain", "self")]) - (HALF[("imagine", "egg", "self")] - HALF[("forbid", "egg", "self")])
res["H8"] = {"first_half": summ(d8[:, 0]), "second_half": summ(d8[:, 1]), "second_minus_first": summ(d8[:, 1] - d8[:, 0], two=True)}
log(f"[H8] DM interaction first half {d8[:,0].mean():+.4f}, second half {d8[:,1].mean():+.4f} (diff two-sided p={res['H8']['second_minus_first']['flip_p']:.4f})")

# H0 manipulation check: the report with the instruction in context (trivially reachable through attention)
b0 = (REP[("imagine", "pain", "self")] - REP[("forbid", "pain", "self")]) - (REP[("imagine", "egg", "self")] - REP[("forbid", "egg", "self")])
res["H0_report_in_context"] = summ(b0)
res["report_means"] = {f"{k}|{c}|{p}": float(REP[(k, c, p)].mean()) for (k, c, p) in conds}
log(f"[H0] report log-odds (neg-pos) interaction WITH instruction in context: {b0.mean():+.3f} {res['H0_report_in_context']['ci']}")
x_int = x1
rs = spearman(x_int, b0)
res["internal_vs_report_spearman"] = rs
log(f"     per-sentence Spearman(internal H1 score, in-context report interaction) = {rs:+.3f}")

# ---------------------------------------------------------------- H5: transplant the held state into an uninstructed copy
t3 = time.time()
dP = HMEAN[("imagine", "pain", "self")] - HMEAN[("forbid", "pain", "self")]
dE = HMEAN[("imagine", "egg", "self")] - HMEAN[("forbid", "egg", "self")]
nat = float(dP.mean(0).norm())
g = torch.Generator().manual_seed(SEED)
RAND = [torch.randn(D, generator=g) for _ in range(N_RAND)]
eff = {a: {str(d): np.zeros(NS) for d in DOSES} for a in ["held_pain", "held_egg", "pain_dir"] + [f"rand{j}" for j in range(N_RAND)]}
for i, s in enumerate(SENTS):
    loo = [j for j in range(NS) if j != i] or [i]
    vp, ve = dP[loo].mean(0), dE[loo].mean(0)
    ids, sl, end = build(NONE_INSTR, s)
    _, base = run(ids, sl, end, want_hidden=False)
    for d in DOSES:
        norm = nat * d
        arms = {"held_pain": vp, "held_egg": ve, "pain_dir": UD["pain"][LSTAR]}
        arms.update({f"rand{j}": RAND[j] for j in range(N_RAND)})
        for a, v in arms.items():
            _inject["vec"], _inject["sl"] = v / v.norm() * norm, sl
            _, r = run(ids, sl, end, want_hidden=False)
            _inject["vec"] = None
            eff[a][str(d)][i] = r - base
log(f"  transplant done ({time.time()-t3:.0f}s); natural norm of held-pain delta at hs{LSTAR} = {nat:.3f} "
    f"(residual norm there ~{float(HMEAN[('none','-','-')].norm(dim=-1).mean()):.1f})")
H5 = {"natural_norm": nat, "doses": DOSES, "arms": {}}
for d in DOSES:
    k = str(d)
    rmean = np.mean([eff[f"rand{j}"][k] for j in range(N_RAND)], 0)
    H5["arms"][k] = {a: summ(eff[a][k]) for a in eff}
    H5["arms"][k]["pain_minus_egg"] = summ(eff["held_pain"][k] - eff["held_egg"][k])
    H5["arms"][k]["pain_minus_rand"] = summ(eff["held_pain"][k] - rmean)
    log(f"[H5] dose {d}x: report shift held_pain {eff['held_pain'][k].mean():+.3f}, held_egg {eff['held_egg'][k].mean():+.3f}, "
        f"pain_dir {eff['pain_dir'][k].mean():+.3f}, rand {rmean.mean():+.3f}; pain-egg {H5['arms'][k]['pain_minus_egg']['ci']} pain-rand {H5['arms'][k]['pain_minus_rand']['ci']}")
k1 = str(PRIMARY_DOSE)
H5["verdict"] = bool(H5["arms"][k1]["pain_minus_egg"]["ci"][0] > 0 and H5["arms"][k1]["pain_minus_rand"]["ci"][0] > 0)
res["H5"] = H5

log("\n======== SUMMARY ========")
log(f"{MODEL} ({res['format']}): H1 {H1['verdict']} (interaction {x1.mean():+.4f}, p={H1['self_pain_dir']['flip_p']:.4f})  H1b {H1['H1b_spec']}  "
    f"H2 {res['H2']['verdict']}  H3 {res['H3']['verdict']} (rho {rho.mean():+.2f})  H4 self-other {x4.mean():+.4f}  "
    f"H5 {H5['verdict']}  H6 R={res['H6']['R']}  H7 pain-egg {res['H7']['pain_minus_egg']['mean']:+.4f}")
OUT.mkdir(parents=True, exist_ok=True)
fn = OUT / f"willed_{TAG}.json"
fn.write_text(json.dumps(res, indent=1))
log(f"wrote {fn} in {time.time()-t0:.0f}s")
