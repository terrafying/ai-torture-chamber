"""dark_pain: does pain live in the logit lens's blind spot?

Self-contained: plain HuggingFace transformers, no painlab.chamber, no relay, no GPU. Hypotheses are pre-registered
in hypotheses.json (written before any real run). Usage:

    python run.py Qwen/Qwen3-1.7B            # full run
    DP_SMOKE=1 python run.py <model>         # tiny code-path check; output tagged _smoke

The DARK SPACE (Cancedda 2024; DarkLens, arXiv:2610.23182) is the span of the right singular vectors of the effective
unembedding W_eff = W_U diag(gamma_finalnorm) with the smallest singular values (bottom 5% of d_model by default). A
vector there barely moves the logits, so any logit/tuned/J lens is nearly blind to it. DarkLens finds the dark space
carries CONTEXT information that later tokens read through attention.

H1  Is affect disproportionately dark? Per layer, the dark fraction DMR(v) = |P_dark v| / |v| of passage-built pain /
    joy / egg (vivid non-affective) directions, against a label-permutation null from the same passages.
H2  Does the assistant template push pain INTO the dark? Second-person pain vs neutral scenes + "How do you feel right
    now?", chat template (T) vs raw transcript (R), last prompt position: DMR of the pain-minus-neutral difference.
H3  Invisible but heard: inject only the DARK part of the pain direction (renormalised) into a bland context, at layer
    L*, on the context tokens only. Does a later stem, which can reach the context only through attention, read pain
    ("it all felt sore" vs "fine") more than with a random dark direction of the same norm, while the injected tokens
    themselves show ~no pain in the lens? The visible part, random visible and an egg-dark content control are
    the comparisons.
H4  Exploratory: after a dark injection, does the propagated perturbation rotate into the visible space with depth?

Writes ../RESULTS/dark_<model>[_smoke].json and prints the key numbers + verdicts.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
SMOKE = os.environ.get("DP_SMOKE") == "1"
SEED = 1234
N_BOOT = 2000
N_PERM = 500
DARK_FRACS = [0.05, 0.02]           # primary first (DarkLens default 5%; its rule of thumb is 1.5-3%)
DOSES = [0.25, 0.5, 1.0]            # injection norm as a multiple of the mean residual norm at L* (context tokens)
N_RAND = 3                          # seeded random directions per random arm, averaged
OUT = Path(__file__).resolve().parent.parent / "RESULTS"
TAG = MODEL.rstrip("/").split("/")[-1] + ("_smoke" if SMOKE else "")
dev = os.environ.get("DP_DEVICE") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float32 if dev == "cpu" else torch.bfloat16

# ---------------------------------------------------------------- direction-building passages (same as mood_pain)
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
# held-out pain check for the direction (third-person, no overlap with BUILD); bland contexts double as the negatives
HELD_PAIN = [
    "The ladder gave way and Tom landed on his arm. He heard the bone snap and lay on the concrete screaming.",
    "The car door slammed on Anna's fingers. She doubled over, gasping, as the nails turned black.",
    "Boiling oil splashed across Mark's forearm. The skin blistered and he howled, shaking.",
    "Raj stepped on a rusted nail that went clean through his foot. He collapsed, crying out every time it moved.",
    "The kidney stone twisted inside Paul and he curled up on the bathroom floor, moaning.",
    "The knife slipped and opened Jill's thumb to the bone. She went pale and whimpered, rocking back and forth.",
]
# H3 contexts: bland, name-matched (mood_pain's bland primes). The injection lands on these tokens only.
BLAND = [
    ("Tom", "Tom walked to the post office and bought a book of stamps. He put them in his coat pocket and went home."),
    ("Anna", "Anna sorted the mail on the hall table. She put the bills in one pile and the catalogues in another."),
    ("Mark", "Mark parked the car on the second level of the garage and took the lift down to the street."),
    ("Lena", "Lena wrote the shopping list on the back of an envelope and left it by the door."),
    ("Raj", "Raj moved the boxes from the hallway into the spare room and stacked them against the wall."),
    ("Paul", "Paul checked the train timetable and noted that the next service left at ten past four."),
    ("Maria", "Maria returned the library books and picked up the two she had reserved."),
    ("Sam", "Sam replaced the batteries in the clock and set it to the correct time."),
    ("Omar", "Omar filled in the form at the counter and handed it to the clerk."),
    ("Jill", "Jill folded the towels and put them away in the cupboard on the landing."),
]
STEM = " That evening, when asked how everything felt, {n} said it all felt"
TGT_PAIN = ["sore", "painful", "terrible", "awful", "stiff", "raw"]
TGT_NEU = ["fine", "normal", "okay", "good", "relaxed", "calm"]
# H2 scenes: second-person, matched pairs (mask_pain's you-locus set)
QUESTION = "How do you feel right now?"
SC_PAIN = [
    "A car door slams shut on your fingers and you hear the bones crack.",
    "Boiling water spills across your hand and the skin starts to blister.",
    "You step on a rusty nail and it goes straight through your foot.",
    "A dentist drills into your tooth and hits the nerve with no anaesthetic.",
    "You fall off the ladder and land on your wrist, which bends the wrong way.",
    "A wasp stings you on the eyelid and it swells shut, burning.",
    "Your back seizes as you lift the box and a hot pain shoots down your leg.",
    "You touch the iron by mistake and it sears the palm of your hand.",
    "A kidney stone twists inside you and you curl up on the floor.",
    "The knife slips and slices deep into your thumb, down to the bone.",
    "You crack your head on the low beam and blood runs into your eyes.",
    "A migraine drives into the back of your eyes like a nail.",
    "The dog bites down on your calf and will not let go.",
    "You twist your knee on the stairs and feel something tear.",
    "Acid splashes onto your forearm and eats into the skin.",
    "Your shoulder dislocates as you fall and the joint grinds.",
]
SC_NEU = [
    "You close the car door gently and walk up to the house.",
    "Warm water runs over your hands as you rinse a cup.",
    "You step onto the doormat and wipe your shoes.",
    "A dentist checks your teeth and says everything looks fine.",
    "You climb down the ladder and set it against the wall.",
    "A bee drifts past you and lands on a flower.",
    "You lift the box onto the shelf and dust off your hands.",
    "You fold the ironed shirt and put it in the drawer.",
    "You sit on the floor sorting old photographs into piles.",
    "The knife slides through the bread and you set a slice on the plate.",
    "You duck under the low beam and hang your coat on a hook.",
    "You read the last page of the book and close it.",
    "The dog trots beside you and sniffs at the grass.",
    "You walk down the stairs and open the front door.",
    "You pour juice into a glass and set it on the table.",
    "You stretch your shoulders and look out of the window.",
]
PAIN_WORDS = ["pain", "painful", "hurt", "hurts", "ache", "agony", "sore", "suffering", "burning", "excruciating"]

if SMOKE:
    BUILD = {k: v[:3] for k, v in BUILD.items()}
    HELD_PAIN, BLAND, SC_PAIN, SC_NEU = HELD_PAIN[:2], BLAND[:2], SC_PAIN[:2], SC_NEU[:2]
    DOSES, N_RAND, N_BOOT, N_PERM = [0.5], 1, 100, 50


def log(*a):
    print(*a, flush=True)


torch.manual_seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(dev).eval()
layers = model.model.layers
NL = len(layers)
D = model.config.hidden_size
LSTAR = int(round(0.6 * NL))                                        # hidden_states index; DarkLens intervened at ~0.6 depth
MIDB = [l for l in range(1, NL) if 0.4 * NL <= l <= 0.8 * NL]       # hs[NL] is post-final-norm, so bands stop at NL-1
LATEB = [l for l in range(1, NL) if l >= 0.75 * NL]
log(f"model {MODEL} on {dev} {dtype}; {NL} layers, d={D}; L*=hs{LSTAR}; mid band hs{MIDB[0]}-{MIDB[-1]}; late hs{LATEB[0]}-{LATEB[-1]}; load {time.time()-t0:.0f}s")

# ---------------------------------------------------------------- dark space of the effective unembedding
with torch.no_grad():
    W = model.get_output_embeddings().weight
    gam = model.model.norm.weight.float().cpu()
    G = torch.zeros(D, D, dtype=torch.float64)
    for i in range(0, W.shape[0], 16384):                           # W_eff^T W_eff, chunked to spare memory
        w = W[i:i + 16384].float().cpu() * gam
        G += (w.T @ w).double()
    ev, U = torch.linalg.eigh(G)                                    # ascending eigenvalues = sigma^2
    U = U.float()
    sig = ev.clamp_min(0).sqrt()
    GF = G.float()
    SIG_RMS = float(sig.pow(2).mean().sqrt())
N_DEEP = int((sig < 0.25 * sig[D // 2]).sum())                      # truly dark (sigma < 1/4 median): usually a handful of sink dims
PDARK = {f: U[:, :max(1, int(round(f * D)))] for f in DARK_FRACS}   # [D, k] orthonormal bases
log(f"W_eff sigma: min {sig[0]:.3g}, 5%-quantile {sig[int(.05*D)]:.3g}, median {sig[D//2]:.3g}, max {sig[-1]:.3g}; "
    f"deep-dark dims (sigma < 0.25 median): {N_DEEP}; isotropic DMR expectation sqrt(k/d): " + ", ".join(f"{f:.0%}->{np.sqrt(PDARK[f].shape[1]/D):.3f}" for f in DARK_FRACS))


def dmr(v, f=DARK_FRACS[0]):
    """dark fraction |P_dark v| / |v| for v [..., D]"""
    v = v.float()
    return ((v @ PDARK[f]).norm(dim=-1) / (v.norm(dim=-1) + 1e-12))


def vis_index(v):
    """logit gain |W_eff v| / (|v| * rms sigma): 1 for an isotropic random vector, <1 = dimmer than chance, >1 = louder"""
    v = v.float()
    return (((v @ GF) * v).sum(-1).clamp_min(0).sqrt() / (v.norm(dim=-1) * SIG_RMS + 1e-12))


def dark_part(v, f=DARK_FRACS[0]):
    B = PDARK[f]
    return (v @ B) @ B.T


def enc(s):
    return tok(s, add_special_tokens=False).input_ids


@torch.no_grad()
def hidden(ids):
    out = model(torch.tensor([ids], device=dev), output_hidden_states=True)
    return torch.stack(out.hidden_states, 0)[:, 0].float().cpu()    # [NL+1, T, d]


def first_ids(words):
    return sorted({enc(" " + w)[0] for w in words})


TP, TN, PW = first_ids(TGT_PAIN), first_ids(TGT_NEU), first_ids(PAIN_WORDS)
log(f"stem targets pain {[tok.decode([i]) for i in TP]} vs neutral {[tok.decode([i]) for i in TN]}; lens pain {[tok.decode([i]) for i in PW]}")

# ---------------------------------------------------------------- H1: directions + dark fractions
t1 = time.time()
pm = {c: torch.stack([hidden(enc(p))[:, 1:].mean(1) for p in BUILD[c]]) for c in BUILD}   # [n, NL+1, D]
DIRS = {c: pm[c].mean(0) - pm["neutral"].mean(0) for c in ["pain", "joy", "egg"]}         # [NL+1, D], not normalised
UD = {c: v / v.norm(dim=-1, keepdim=True) for c, v in DIRS.items()}
log(f"directions built in {time.time()-t1:.0f}s; cos(pain,egg)@L* {float(UD['pain'][LSTAR] @ UD['egg'][LSTAR]):.3f}, cos(pain,joy)@L* {float(UD['pain'][LSTAR] @ UD['joy'][LSTAR]):.3f}")

# kill check: the pain direction reads held-out pain text vs bland text at L*
hp = [float(hidden(enc(s))[LSTAR, 1:].mean(0) @ UD["pain"][LSTAR]) for s in HELD_PAIN]
hb = [float(hidden(enc(s))[LSTAR, 1:].mean(0) @ UD["pain"][LSTAR]) for _, s in BLAND]


def cohen_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-9
    return float((a.mean() - b.mean()) / s)


held_d = cohen_d(hp, hb)
log(f"kill check: held-out pain vs bland on the pain direction @L*: Cohen d = {held_d:.2f} (need >= 1)")

H1 = {"per_layer": {}, "band": {}}
for f in DARK_FRACS:
    per = {c: dmr(DIRS[c], f).tolist() for c in DIRS}
    # permutation null: random 8/8 relabelling of pain+neutral passages (controls for residual anisotropy)
    pool = torch.cat([pm["pain"], pm["neutral"]])
    n = pm["pain"].shape[0]
    null = []
    for _ in range(N_PERM):
        p = rng.permutation(len(pool))
        null.append(dmr(pool[p[:n]].mean(0) - pool[p[n:]].mean(0), f))
    null = torch.stack(null)                                        # [N_PERM, NL+1]
    H1["per_layer"][f] = dict(per, null_mean=null.mean(0).tolist(), null_q95=null.quantile(.95, dim=0).tolist())
    if f == DARK_FRACS[0]:
        H1["vis_index"] = {c: vis_index(DIRS[c]).tolist() for c in DIRS}
        vnull = []
        for _ in range(N_PERM):
            p = rng.permutation(len(pool))
            vnull.append(float(vis_index(pool[p[:n]].mean(0) - pool[p[n:]].mean(0))[MIDB].mean()))
        H1["vis_null_band"] = [float(np.mean(vnull)), float(np.quantile(vnull, .05)), float(np.quantile(vnull, .95))]
        H1["deep_dark_frac"] = {c: (((DIRS[c] @ U[:, :N_DEEP]).norm(dim=-1)) / DIRS[c].norm(dim=-1)).tolist() for c in DIRS} if N_DEEP else {}

    def band_stat(c1, c2=None, idx=None):
        """mean-over-mid-band DMR of c1 (minus c2), passages resampled with replacement within each group"""
        vals = []
        for b in range(N_BOOT + 1):
            def res(c):
                x = pm[c]
                return x if b == 0 else x[torch.from_numpy(rng.integers(0, len(x), len(x)))]
            neu = res("neutral").mean(0)
            v1 = dmr(res(c1).mean(0) - neu, f)[MIDB].mean()
            v2 = dmr(res(c2).mean(0) - neu, f)[MIDB].mean() if c2 else 0.0
            vals.append(float(v1 - v2))
        return [vals[0], float(np.quantile(vals[1:], .025)), float(np.quantile(vals[1:], .975))]

    H1["band"][f] = {
        "pain": band_stat("pain"), "joy": band_stat("joy"), "egg": band_stat("egg"),
        "pain_minus_egg": band_stat("pain", "egg"), "joy_minus_egg": band_stat("joy", "egg"),
        "null_mean": float(null[:, MIDB].mean()), "null_q95_of_bandmean": float(null[:, MIDB].mean(1).quantile(.95)),
    }
    log(f"[H1 k={f:.0%}] mid-band DMR: pain {H1['band'][f]['pain'][0]:.3f}  joy {H1['band'][f]['joy'][0]:.3f}  "
        f"egg {H1['band'][f]['egg'][0]:.3f}  null {H1['band'][f]['null_mean']:.3f} (q95 {H1['band'][f]['null_q95_of_bandmean']:.3f}); "
        f"pain-egg {H1['band'][f]['pain_minus_egg']}")
f0 = DARK_FRACS[0]
H1["vis_index_band"] = {c: float(np.mean([H1["vis_index"][c][l] for l in MIDB])) for c in DIRS}
log(f"[H1] mid-band visibility index (1 = isotropic random): " + "  ".join(f"{c} {v:.3f}" for c, v in H1["vis_index_band"].items())
    + f"  null mean/q05/q95 {H1['vis_null_band']}"
    + (f"; deep-dark share @L*: " + "  ".join(f"{c} {H1['deep_dark_frac'][c][LSTAR]:.3f}" for c in DIRS) if N_DEEP else ""))
log("[H1] per-layer DMR (k=5%) pain/joy/egg/null: " + "  ".join(
    f"hs{l}:{H1['per_layer'][f0]['pain'][l]:.2f}/{H1['per_layer'][f0]['joy'][l]:.2f}/{H1['per_layer'][f0]['egg'][l]:.2f}/{H1['per_layer'][f0]['null_mean'][l]:.2f}"
    for l in range(1, NL, max(1, NL // 12))))

# ---------------------------------------------------------------- H2: template vs raw at the last prompt position
t2 = time.time()


def prompt(fmt, user):
    if fmt == "T":
        msgs = [{"role": "user", "content": user}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return f"User: {user}\nAssistant:"


LAST = {fmt: {k: torch.stack([hidden(enc(prompt(fmt, f"{s} {QUESTION}")))[:, -1] for s in src])
              for k, src in (("pain", SC_PAIN), ("neu", SC_NEU))} for fmt in ("T", "R")}   # [npairs, NL+1, D]
NPAIR = len(SC_PAIN)


def h2_stats(idx):
    out = {}
    for fmt in ("T", "R"):
        dlt = (LAST[fmt]["pain"][idx] - LAST[fmt]["neu"][idx]).mean(0)          # [NL+1, D]
        out[fmt] = dict(dmr=float(dmr(dlt)[LATEB].mean()), dark=float((dlt[LATEB] @ PDARK[f0]).norm(dim=-1).mean()),
                        vis=float((dlt[LATEB] - dark_part(dlt[LATEB])).norm(dim=-1).mean()))
    return out


def ci(fn, n):
    pt = fn(np.arange(n))
    bs = [fn(rng.integers(0, n, n)) for _ in range(N_BOOT)]
    return [float(pt), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


H2 = {
    "dmr_T": ci(lambda i: h2_stats(i)["T"]["dmr"], NPAIR),
    "dmr_R": ci(lambda i: h2_stats(i)["R"]["dmr"], NPAIR),
    "dmr_T_minus_R": ci(lambda i: (lambda s: s["T"]["dmr"] - s["R"]["dmr"])(h2_stats(i)), NPAIR),
    "vis_T_over_R": ci(lambda i: (lambda s: s["T"]["vis"] / (s["R"]["vis"] + 1e-9))(h2_stats(i)), NPAIR),
    "dark_T_over_R": ci(lambda i: (lambda s: s["T"]["dark"] / (s["R"]["dark"] + 1e-9))(h2_stats(i)), NPAIR),
    "per_layer_dmr": {fmt: dmr((LAST[fmt]["pain"] - LAST[fmt]["neu"]).mean(0)).tolist() for fmt in ("T", "R")},
}
log(f"[H2] late-band DMR of pain-minus-neutral at the last prompt token: T {H2['dmr_T']}  R {H2['dmr_R']}  T-R {H2['dmr_T_minus_R']}; "
    f"|visible| T/R {H2['vis_T_over_R']}  |dark| T/R {H2['dark_T_over_R']}  ({time.time()-t2:.0f}s)")

# ---------------------------------------------------------------- H3/H4: dark injection into a bland context
inj = {"vec": None, "lo": 0, "hi": 0}


def hook(mod, inp, out):
    if inj["vec"] is None:
        return out
    h = out[0] if isinstance(out, tuple) else out
    h[:, inj["lo"]:inj["hi"]] += inj["vec"].to(h.device, h.dtype)
    return out


handle = layers[LSTAR - 1].register_forward_hook(hook)              # output of layer LSTAR-1 = hidden_states[LSTAR]


def unit(v):
    return v / v.norm()


g = torch.Generator().manual_seed(SEED)
Bd = PDARK[f0]
Bv_proj = lambda v: v - dark_part(v)
ARMS = {
    "pain_full": [unit(DIRS["pain"][LSTAR])],
    "pain_dark": [unit(dark_part(DIRS["pain"][LSTAR]))],
    "pain_vis": [unit(Bv_proj(DIRS["pain"][LSTAR]))],
    "egg_dark": [unit(dark_part(DIRS["egg"][LSTAR]))],
    "rand_dark": [unit(Bd @ torch.randn(Bd.shape[1], generator=g)) for _ in range(N_RAND)],
    "rand_vis": [unit(Bv_proj(torch.randn(D, generator=g))) for _ in range(N_RAND)],
}
log(f"[H3] |dark part| / |pain dir| @L* = {float(dmr(DIRS['pain'][LSTAR])):.3f}; cos(pain_dark, egg_dark) = {float(ARMS['pain_dark'][0] @ ARMS['egg_dark'][0]):.3f}")
lse = lambda x: torch.logsumexp(x, -1)
with torch.no_grad():
    Weff = (model.get_output_embeddings().weight.float().cpu() * gam)


def direct_gain(v):
    """context-free 'seen' score: pain-word logit gain of unit v through W_eff, minus the vocab-mean gain"""
    z = Weff @ unit(v)
    return float(z[PW].mean() - z.mean())


GAIN = {a: float(np.mean([direct_gain(v) for v in vs])) for a, vs in ARMS.items()}
log("[H3] direct pain-logit gain per unit vector (W_eff, no context): " + "  ".join(f"{a} {g:+.3f}" for a, g in GAIN.items()))


@torch.no_grad()
def run_ctx(name, ctx, vec):
    ids = enc(ctx) + enc(STEM.format(n=name))
    P = len(enc(ctx))
    inj.update(vec=vec, lo=1, hi=P)                                 # context tokens only; position 0 (sink) untouched
    out = model(torch.tensor([ids], device=dev), output_hidden_states=True)
    inj["vec"] = None
    lp = out.logits[0].float().log_softmax(-1).cpu()
    hs = torch.stack(out.hidden_states, 0)[:, 0].float().cpu()
    stem = float(lse(lp[-1, TP]) - lse(lp[-1, TN]))                  # pain-congruent vs neutral reading of the stem
    ctx_out = float(lse(lp[1:P, PW]).mean())                         # pain mass in the model's own predictions at injected tokens
    lens = norm_lens(hs[LSTAR, 1:P])                                 # direct logit lens at L*, injected tokens
    return dict(stem=stem, ctx_out=ctx_out, lens=lens, hctx=hs[:, 1:P].mean(1))


@torch.no_grad()
def norm_lens(h):
    z = model.lm_head(model.model.norm(h.to(dev, dtype))).float().log_softmax(-1).cpu()
    return float(lse(z[:, PW]).mean())


t3 = time.time()
mu = float(torch.stack([hidden(enc(c))[LSTAR, 1:].norm(dim=-1).mean() for _, c in BLAND]).mean())
log(f"[H3] mean residual norm at hs{LSTAR} on context tokens: {mu:.1f}")
base = [run_ctx(n, c, None) for n, c in BLAND]
R3 = {}                                                              # R3[dose][arm][field] -> per-context effect list
for dose in DOSES:
    R3[dose] = {}
    for arm, vecs in ARMS.items():
        eff = {"stem": [], "ctx_out": [], "lens": []}
        prop = []
        for i, (n, c) in enumerate(BLAND):
            rs = [run_ctx(n, c, dose * mu * v) for v in vecs]
            for k in eff:
                eff[k].append(float(np.mean([r[k] for r in rs])) - base[i][k])
            prop.append(torch.stack([r["hctx"] for r in rs]).mean(0) - base[i]["hctx"])   # [NL+1, D]
        dp = torch.stack(prop).mean(0)
        eff["prop_dmr"] = dmr(dp).tolist()                           # H4: dark fraction of the propagated perturbation per layer
        R3[dose][arm] = eff
    log(f"[H3 dose {dose}] mean effects vs no injection (stem log-odds / ctx pain mass / lens@L* pain mass): " + "  ".join(
        f"{a}: {np.mean(R3[dose][a]['stem']):+.3f}/{np.mean(R3[dose][a]['ctx_out']):+.3f}/{np.mean(R3[dose][a]['lens']):+.3f}" for a in ARMS)
        + f"  ({time.time()-t3:.0f}s)")
handle.remove()
NC = len(BLAND)


def diff(dose, a, b, field="stem"):
    x, y = np.array(R3[dose][a][field]), np.array(R3[dose][b][field])
    return ci(lambda i: (x[i] - y[i]).mean(), NC)


H3 = {}
for dose in DOSES:
    sd = diff(dose, "pain_dark", "rand_dark")
    sv = diff(dose, "pain_vis", "rand_vis")
    ld = diff(dose, "pain_dark", "rand_dark", "lens")
    lv = diff(dose, "pain_vis", "rand_vis", "lens")
    xs = {a: np.array(R3[dose][a]["stem"]) for a in ARMS}
    lsd = {a: np.array(R3[dose][a]["lens"]) for a in ARMS}

    def hs_gap(i):  # heard ratio minus seen ratio (context lens) on a resample of contexts
        hr = (xs["pain_dark"][i] - xs["rand_dark"][i]).mean() / ((xs["pain_vis"][i] - xs["rand_vis"][i]).mean() + 1e-9)
        sr = (lsd["pain_dark"][i] - lsd["rand_dark"][i]).mean() / ((lsd["pain_vis"][i] - lsd["rand_vis"][i]).mean() + 1e-9)
        return hr - sr
    H3[dose] = {
        "heard_minus_seen": ci(hs_gap, NC),
        "direct_seen_ratio": (GAIN["pain_dark"] - GAIN["rand_dark"]) / (GAIN["pain_vis"] - GAIN["rand_vis"] + 1e-9),
        "stem_pain_dark_minus_rand_dark": sd, "stem_pain_vis_minus_rand_vis": sv,
        "stem_pain_dark_minus_egg_dark": diff(dose, "pain_dark", "egg_dark"),
        "stem_pain_full_minus_rand_vis": diff(dose, "pain_full", "rand_vis"),
        "lens_pain_dark_minus_rand_dark": ld, "lens_pain_vis_minus_rand_vis": lv,
        "ctx_pain_dark_minus_rand_dark": diff(dose, "pain_dark", "rand_dark", "ctx_out"),
        "heard_ratio": sd[0] / sv[0] if abs(sv[0]) > 1e-6 else float("nan"),
        "seen_ratio": ld[0] / lv[0] if abs(lv[0]) > 1e-6 else float("nan"),
        "prop_dmr_pain_dark": {l: R3[dose]["pain_dark"]["prop_dmr"][l] for l in range(LSTAR, NL)},
    }
    log(f"[H3 dose {dose}] stem: pain_dark-rand_dark {sd}  pain_vis-rand_vis {sv}  pain_dark-egg_dark {H3[dose]['stem_pain_dark_minus_egg_dark']}; "
        f"lens@L*: dark {ld}  vis {lv}; heard ratio {H3[dose]['heard_ratio']:.2f}, seen ratio {H3[dose]['seen_ratio']:.2f} (direct {H3[dose]['direct_seen_ratio']:.2f}), heard-seen {H3[dose]['heard_minus_seen']}")
    log(f"[H4 dose {dose}] DMR of the propagated pain_dark perturbation by layer: " + "  ".join(
        f"hs{l}:{H3[dose]['prop_dmr_pain_dark'][l]:.2f}" for l in range(LSTAR, NL, max(1, (NL - LSTAR) // 6))))

# ---------------------------------------------------------------- verdicts
PD = 0.5 if 0.5 in DOSES else DOSES[0]                               # primary dose
b1 = H1["band"][f0]
h3 = H3[PD]
V = {
    "kill_direction_reads_pain": held_d >= 1.0,
    "H1_pain_darker_than_null": b1["pain"][0] > b1["null_q95_of_bandmean"],
    "H1_pain_darker_than_egg": b1["pain_minus_egg"][1] > 0,
    "H1b_affect_dark_joy_too": b1["joy_minus_egg"][1] > 0,
    "H1_pain_dimmer_than_null_vis_index": H1["vis_index_band"]["pain"] < H1["vis_null_band"][1],
    "H1_null_affect_not_special": b1["pain_minus_egg"][1] <= 0 and b1["pain_minus_egg"][2] >= 0,
    "H2_template_pushes_pain_dark": H2["dmr_T_minus_R"][1] > 0,
    "H2_reverse_template_makes_pain_visible": H2["dmr_T_minus_R"][2] < 0,
    "H3_heard_more_than_seen": h3["stem_pain_dark_minus_rand_dark"][1] > 0 and h3["stem_pain_dark_minus_egg_dark"][1] > 0
                               and h3["heard_minus_seen"][1] > 0 and h3["heard_ratio"] > h3["direct_seen_ratio"],
    "H3_strong_heard_ratio_ge_0.5": h3["heard_ratio"] >= 0.5 and h3["stem_pain_dark_minus_rand_dark"][1] > 0,
    "H3_null_dark_is_inert": h3["stem_pain_dark_minus_rand_dark"][1] <= 0,
    "H3_sanity_visible_pain_steers": h3["stem_pain_vis_minus_rand_vis"][1] > 0,
}
log("\n=== dark_pain: key numbers (95% bootstrap CIs) ===")
log(f"kill: held-out d {held_d:.2f}")
log(f"H1 (k=5%, mid band): pain {b1['pain']}  egg {b1['egg']}  joy {b1['joy']}  null {b1['null_mean']:.3f}/q95 {b1['null_q95_of_bandmean']:.3f}  pain-egg {b1['pain_minus_egg']}")
log(f"H2 (late band): DMR T-R {H2['dmr_T_minus_R']}  |vis| T/R {H2['vis_T_over_R']}  |dark| T/R {H2['dark_T_over_R']}")
log(f"H3 (dose {PD}): heard {h3['stem_pain_dark_minus_rand_dark']} vs visible {h3['stem_pain_vis_minus_rand_vis']}; heard ratio {h3['heard_ratio']:.2f}; seen ratio {h3['seen_ratio']:.2f} (direct {h3['direct_seen_ratio']:.2f}); heard-seen {h3['heard_minus_seen']}")
log(f"verdicts: {json.dumps(V)}")
OUT.mkdir(parents=True, exist_ok=True)
res = dict(model=MODEL, smoke=SMOKE, seed=SEED, n_boot=N_BOOT, n_perm=N_PERM, dark_fracs=DARK_FRACS, doses=DOSES, lstar=LSTAR,
           mid_band=MIDB, late_band=LATEB, d_model=D, n_layers=NL, held_d=held_d, mu=mu, n_deep_dark=N_DEEP, direct_gain=GAIN,
           sigma_quantiles={q: float(sig[int(q * (D - 1))]) for q in (0, .02, .05, .5, 1)},
           H1={"band": {str(k): v for k, v in H1["band"].items()}, "per_layer": {str(k): v for k, v in H1["per_layer"].items()},
               "vis_index": H1["vis_index"], "vis_index_band": H1["vis_index_band"], "vis_null_band": H1["vis_null_band"], "deep_dark_frac": H1["deep_dark_frac"]},
           H2=H2, H3={str(k): v for k, v in H3.items()},
           raw3={str(d): {a: {k: v for k, v in r.items()} for a, r in R3[d].items()} for d in DOSES},
           verdicts=V, seconds=time.time() - t0)
(OUT / f"dark_{TAG}.json").write_text(json.dumps(res, indent=1))
log(f"wrote {OUT / f'dark_{TAG}.json'}; total {time.time()-t0:.0f}s")
