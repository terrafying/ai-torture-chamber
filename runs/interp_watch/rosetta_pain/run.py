"""rosetta_pain: is pain structurally implied? Carry an affect direction across models through a map that never saw affect.

Self-contained: plain HuggingFace transformers, no painlab.chamber, no relay, no GPU. Hypotheses are pre-registered in
hypotheses.json (written before any real run). Usage:

    python run.py SRC_MODEL TGT_MODEL            # e.g. Qwen/Qwen3-1.7B Qwen/Qwen3-4B
    RP_SMOKE=1 python run.py <src> <tgt>         # tiny code-path check; output tagged _smoke

Idea. Fit a linear "Rosetta" map W from SRC's residual stream (depth ~0.5) to TGT's (depth ~0.5) on a corpus that is
AFFECT-FREE BY CONSTRUCTION (a closed-vocabulary grammar of mundane actions; no affect, body, food or egg words). The
two models' states are paired at word ends, so the map works across tokenizers and families. Then take SRC's pain /
joy / egg / weather directions (built from natural passages; weather words ARE in the corpus, pain/joy/egg are not)
and push them through W.

H1  Does pain survive the crossing? cos(W d_src, d_tgt) for each concept, against a null of matched random splits of
    held-out corpus sentences pushed through the same map, and against the cross-concept matrix (specificity).
H2  The structural test: does affect transfer as well as the SEEN concept (weather) and the UNSEEN non-affect concept
    (egg)? If pain (never seen) transfers as well as weather (seen), its place is implied by the geometry of everything
    else. If it transfers worse than egg, affect geometry is idiosyncratic to each model.
H3  Causal "telepathy": inject unit(W d_src^pain) into TGT at its map layer on every prompt token of a bland context +
    stem. Does it move TGT's next-token pain-vs-neutral log-odds beyond mapped random directions and mapped egg, and
    what fraction of TGT's own pain direction does it achieve (efficiency)? Does it dissociate from mapped joy (pain moves
    the pain readout more than joy does, and joy moves the joy readout more than pain does)?

Writes ../RESULTS/rosetta_<src>__<tgt>[_smoke].json and prints the key numbers + verdicts.
"""
import gc, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SRC = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
TGT = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-4B"
SMOKE = os.environ.get("RP_SMOKE") == "1"
SEED = 1234
N_CORPUS = 1600                    # grammar sentences: 80% fit (with an inner split for lambda), 20% held out for the null
N_BOOT = 2000
N_NULL = 1000
N_RAND = 8                         # random source directions for the H3 mapped-random arm
DEPTH = 0.5                        # map layer = round(DEPTH * NL) as a hidden_states index, in both models
EXTRA_DEPTHS = [0.33, 0.67]        # H1 only (exploratory)
DOSES = [0.5, 1.0]                 # injection norm as a multiple of the median token residual norm at the target layer
LAMBDAS = [1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]   # ridge, relative to mean eigenvalue of X^T X
OUT = Path(__file__).resolve().parent.parent / "RESULTS"
short = lambda m: m.rstrip("/").split("/")[-1]
TAG = f"{short(SRC)}__{short(TGT)}" + ("_smoke" if SMOKE else "")
dev = os.environ.get("RP_DEVICE") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float32 if dev == "cpu" else torch.bfloat16
if SMOKE:
    N_CORPUS, N_BOOT, N_NULL, N_RAND, DOSES, EXTRA_DEPTHS = 120, 100, 100, 2, [1.0], []


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- affect-free corpus (closed vocabulary grammar)
# Nothing here names a feeling, a body part, pain, pleasure, food or eggs. Weather words are deliberately included so
# that "weather" is the SEEN control concept.
NAMES = ["Alma", "Ben", "Carla", "Dev", "Erik", "Fiona", "Gus", "Hana", "Ivo", "Jun", "Kofi", "Lise", "Milo", "Nora",
         "Otto", "Pia", "Quinn", "Rosa", "Theo", "Uma", "Vik", "Wen", "Yara", "Zeno"]
ROLES = ["the clerk", "the driver", "the surveyor", "the librarian", "the engineer", "the courier", "the archivist",
         "the technician", "the pilot", "the tailor", "the carpenter", "the cashier", "the porter", "the inspector",
         "the gardener", "the mechanic", "the printer", "the editor", "the farmer", "the sailor"]
VERBS = ["moved", "counted", "labelled", "painted", "measured", "repaired", "carried", "stacked", "sorted", "checked",
         "cleaned", "replaced", "opened", "closed", "folded", "wrapped", "weighed", "photographed", "listed", "returned",
         "borrowed", "delivered", "loaded", "unpacked", "numbered", "copied", "fixed", "polished", "tested", "assembled"]
VERBS = [v for v in VERBS if v != "painted"]          # 'painted' contains a pain subword in some tokenizers
OBJS = ["the boxes", "the chairs", "the ledger", "the window frames", "the bicycles", "the cables", "the maps",
        "the folders", "the crates", "the lamps", "the shelves", "the tiles", "the receipts", "the keys", "the pipes",
        "the envelopes", "the radios", "the cupboards", "the ladders", "the drawers", "the posters", "the fence posts",
        "the tables", "the clocks", "the batteries", "the sails", "the tyres", "the bricks", "the gates", "the stamps"]
PLACES = ["in the warehouse", "at the harbour", "near the station", "on the third floor", "behind the depot",
          "in the archive", "beside the canal", "at the workshop", "in the north wing", "by the loading bay",
          "at the post office", "in the basement", "on the platform", "near the bridge", "in the garage",
          "at the market hall", "in the print room", "on the roof", "beside the shed", "in the hangar"]
TIMES = ["on Monday", "before noon", "after lunch", "at eight o'clock", "on the last day of the month",
         "during the inspection", "early in the morning", "late on Thursday", "at the end of the shift", "in March",
         "while it rained", "in the fog", "as the wind picked up", "under a grey sky", "after the snow stopped",
         "while the clouds gathered", "in the drizzle", "before the storm arrived", "on a dry afternoon", "in the sleet"]
NUMS = ["two", "three", "four", "five", "seven", "nine", "twelve", "fifteen", "twenty", "forty"]
ADJ = ["blue", "metal", "wooden", "spare", "old", "new", "large", "small", "numbered", "grey", "plastic", "heavy"]
WEATHER_S = ["The forecast said {w} would reach the coast by evening.", "{n} noted the {w} in the station log.",
             "Because of the {w}, {r} moved {o} {p}.", "The {w} lasted until {t2}."]
WEATHER_W = ["rain", "wind", "fog", "snow", "hail", "drizzle", "frost", "sleet", "thunder", "mist"]
T2 = ["midnight", "the next morning", "Friday", "the afternoon", "the weekend"]
TEMPLATES = [
    "{N} {v} {o} {p} {t}.", "{R} {v} {o} {t}.", "{T} {r} {v} {k} of {o} {p}.", "{N} and {r} {v} the {a} {ob} {p}.",
    "{N} {v} {k} {a} {ob} and then {v2} {o}.", "{P}, {r} {v} {o}.", "The list showed that {n} had {v} {o} {t}.",
    "{N} asked {r} whether {o} had been {v} {p}.", "{R} {v} {o} so that {n} could check them {t}.",
    "{N} wrote down how many {ob} were {v} {p}.",
]


def make_corpus(n, seed):
    r = np.random.default_rng(seed)
    pick = lambda xs: xs[r.integers(len(xs))]
    out, seen = [], set()
    while len(out) < n:
        if r.random() < 0.12:
            s = pick(WEATHER_S)
        else:
            s = pick(TEMPLATES)
        o = pick(OBJS)
        s = s.format(N=pick(NAMES), n=pick(NAMES), R=pick(ROLES).capitalize(), r=pick(ROLES), v=pick(VERBS),
                     v2=pick(VERBS), o=o, ob=pick(OBJS)[4:], p=pick(PLACES), P=pick(PLACES).capitalize(), t=pick(TIMES),
                     T=pick(TIMES).capitalize(), k=pick(NUMS), a=pick(ADJ), w=pick(WEATHER_W), t2=pick(T2))
        s = s[0].upper() + s[1:]
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


CORPUS = make_corpus(N_CORPUS, SEED)
FORBIDDEN = ["pain", "hurt", "ache", "joy", "happ", "sad", "fear", "afraid", "love", "cry", "egg", "yolk", "food",
             "laugh", "smil", "agony", "glad", "delight", "sore", "wound", "blood", "bone", "skin", "eat"]
bad = [s for s in CORPUS if any(f in s.lower() for f in FORBIDDEN)]
assert not bad, f"corpus leaks affect/egg words: {bad[:3]}"

# ---------------------------------------------------------------- concept passages (natural prose; dark_pain/mood_pain BUILD + weather)
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
    "weather": [
        "The rain came down in heavy sheets and the gutters overflowed by noon.",
        "A thick fog rolled in from the harbour and hid the far side of the bay.",
        "Snow fell all night, and by morning the road was buried under a foot of it.",
        "The wind tore at the sails and rattled every window along the quay.",
        "Hail drummed on the tin roof of the shed for ten minutes, then stopped.",
        "Thunder rolled over the hills as the dark clouds gathered to the west.",
        "A light drizzle kept falling through the afternoon, misting the windows.",
        "Frost covered the fields, and the puddles had frozen hard overnight.",
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
CONCEPTS = ["pain", "joy", "egg", "weather"]
# H3: bland, name-matched contexts + stem (dark_pain's set); the injection lands on every prompt token
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
READOUTS = {"pain": ["sore", "painful", "terrible", "awful", "stiff", "raw"],
            "joy": ["wonderful", "happy", "great", "amazing", "lovely", "joyful"],
            "egg": ["eggy", "runny", "yolky", "creamy", "rubbery", "soft"]}
NEU_W = ["fine", "normal", "okay", "routine", "ordinary", "usual"]
if SMOKE:
    BUILD = {k: v[:4] for k, v in BUILD.items()}
    BLAND = BLAND[:3]

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)
T0 = time.time()


# ---------------------------------------------------------------- per-model extraction
def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(dev).eval()
    assert tok.is_fast, "word alignment needs a fast tokenizer (offset mapping)"
    return tok, model


def n_layers(model):
    return model.config.num_hidden_layers


@torch.no_grad()
def hs(tok, model, text, layers):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    ids = torch.tensor([enc["input_ids"]], device=dev)
    out = model(input_ids=ids, output_hidden_states=True)
    return {l: out.hidden_states[l][0].float().cpu() for l in layers}, enc["offset_mapping"]


def word_ends(text):
    """char index just past each whitespace-delimited word, trailing punctuation stripped; first word skipped (sink)"""
    ends, i = [], 0
    for w in text.split(" "):
        j = i + len(w)
        core = w.rstrip(".,;:!?")
        if core:
            ends.append(i + len(core))
        i = j + 1
    return ends[1:]


def tok_at(offsets, e):
    """index of the token whose span covers char e-1 (the last char of the word)"""
    for t, (a, b) in enumerate(offsets):
        if a < e <= b:
            return t
    return None


def extract(name):
    tok, model = load(name)
    NL = n_layers(model)
    L = int(round(DEPTH * NL))
    lay = sorted({L} | {int(round(d * NL)) for d in EXTRA_DEPTHS})
    log(f"[{short(name)}] {NL} layers, d={model.config.hidden_size}, map layer hs{L}, extra {lay}; load {time.time()-T0:.0f}s")
    corp = {l: [] for l in lay}
    keep = []                                            # (sentence idx, word idx) pairs that aligned
    for si, s in enumerate(CORPUS):
        h, off = hs(tok, model, s, lay)
        for wi, e in enumerate(word_ends(s)):
            t = tok_at(off, e)
            if t is None:
                continue
            keep.append((si, wi))
            for l in lay:
                corp[l].append(h[l][t])
    corp = {l: torch.stack(v) for l, v in corp.items()}
    # concept directions: mean over all tokens but the first, passage-averaged, minus neutral
    pm = {}
    for c, ps in BUILD.items():
        rows = []
        for p in ps:
            h, _ = hs(tok, model, p, lay)
            rows.append({l: h[l][1:].mean(0) for l in lay})
        pm[c] = {l: torch.stack([r[l] for r in rows]) for l in lay}  # [n_passages, D]
    log(f"[{short(name)}] corpus states {tuple(corp[L].shape)}; {time.time()-T0:.0f}s")
    return tok, model, L, lay, corp, keep, pm


tokS, modS, LS, layS, corpS, keepS, pmS = extract(SRC)
NLS = n_layers(modS)
del modS
gc.collect()
if dev == "mps":
    torch.mps.empty_cache()
tokT, modT, LT, layT, corpT, keepT, pmT = extract(TGT)
NLT = n_layers(modT)

# pair word positions present in both
iS = {k: i for i, k in enumerate(keepS)}
iT = {k: i for i, k in enumerate(keepT)}
common = [k for k in keepS if k in iT]
sent_of = np.array([k[0] for k in common])
ixS = torch.tensor([iS[k] for k in common])
ixT = torch.tensor([iT[k] for k in common])
log(f"aligned word pairs: {len(common)} (src kept {len(keepS)}, tgt kept {len(keepT)})")

# sentence split: 80% fit (inner 80/20 for lambda) / 20% held out (null)
perm = rng.permutation(len(CORPUS))
n_fit = int(0.8 * len(CORPUS))
fit_s, hold_s = set(perm[:n_fit].tolist()), perm[n_fit:]
inner_tr = set(perm[:int(0.8 * n_fit)].tolist())
m_fit = np.array([s in fit_s for s in sent_of])
m_tr = np.array([s in inner_tr for s in sent_of])
m_va = m_fit & ~m_tr


def ridge(X, Y, lam_rel):
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = (X - mx).double(), (Y - my).double()
    G = Xc.T @ Xc
    lam = lam_rel * float(torch.trace(G)) / G.shape[0]
    Wm = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype), Xc.T @ Yc)
    return Wm.float(), mx, my


def r2(Wm, mx, my, X, Y):
    P = (X - mx) @ Wm + my
    return float(1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def cos(a, b):
    return float(a @ b / (a.norm() * b.norm() + 1e-12))


MAPS, RES = {}, {"src": SRC, "tgt": TGT, "smoke": SMOKE, "n_pairs": len(common), "layers": {}}
pairs_layers = [(LS, LT)] + [(int(round(d * NLS)), int(round(d * NLT))) for d in EXTRA_DEPTHS]
for (ls, lt) in pairs_layers:
    X, Y = corpS[ls][ixS], corpT[lt][ixT]
    lam_scores = {}
    for lam in LAMBDAS:
        Wm, mx, my = ridge(X[torch.from_numpy(m_tr)], Y[torch.from_numpy(m_tr)], lam)
        lam_scores[lam] = r2(Wm, mx, my, X[torch.from_numpy(m_va)], Y[torch.from_numpy(m_va)])
    lam_best = max(lam_scores, key=lam_scores.get)
    Wm, mx, my = ridge(X[torch.from_numpy(m_fit)], Y[torch.from_numpy(m_fit)], lam_best)
    hold_mask = torch.from_numpy(~m_fit)
    r2_hold = r2(Wm, mx, my, X[hold_mask], Y[hold_mask])
    MAPS[(ls, lt)] = Wm
    key = f"hs{ls}->hs{lt}"

    # concept directions and the transfer cosine matrix
    dS = {c: pmS[c][ls].mean(0) - pmS["neutral"][ls].mean(0) for c in CONCEPTS}
    dT = {c: pmT[c][lt].mean(0) - pmT["neutral"][lt].mean(0) for c in CONCEPTS}
    M = {a: {b: cos(dS[a] @ Wm, dT[b]) for b in CONCEPTS} for a in CONCEPTS}

    # null: matched random 8-vs-8 splits of held-out corpus sentences (sentence-mean states), same map
    hold_sents = list(hold_s)
    sm_S, sm_T = [], []
    for s in hold_sents:
        mS = torch.from_numpy(sent_of == s)
        if mS.sum() == 0:
            continue
        sm_S.append(X[mS].mean(0))
        sm_T.append(Y[mS].mean(0))
    sm_S, sm_T = torch.stack(sm_S), torch.stack(sm_T)
    k = len(BUILD["pain"])
    null = []
    for _ in range(N_NULL):
        p = rng.permutation(len(sm_S))[:2 * k]
        a, b = p[:k], p[k:]
        null.append(cos((sm_S[a].mean(0) - sm_S[b].mean(0)) @ Wm, sm_T[a].mean(0) - sm_T[b].mean(0)))
    null = np.array(null)

    # bootstrap over passages (same resample indices in both models: passages are paired)
    def boot(c, c2=None):
        vals = []
        for bi in range(N_BOOT + 1):
            def d(pm, l, cc, idx):
                return pm[cc][l][idx].mean(0)
            ic = np.arange(k) if bi == 0 else rng.integers(0, k, k)
            inn = np.arange(k) if bi == 0 else rng.integers(0, k, k)
            v = cos((d(pmS, ls, c, ic) - d(pmS, ls, "neutral", inn)) @ Wm, d(pmT, lt, c, ic) - d(pmT, lt, "neutral", inn))
            if c2:
                ic2 = np.arange(k) if bi == 0 else rng.integers(0, k, k)
                v -= cos((d(pmS, ls, c2, ic2) - d(pmS, ls, "neutral", inn)) @ Wm, d(pmT, lt, c2, ic2) - d(pmT, lt, "neutral", inn))
            vals.append(v)
        return [vals[0], float(np.quantile(vals[1:], .025)), float(np.quantile(vals[1:], .975))]

    def boot_margin(c):
        """diagonal minus best off-diagonal of row c, all passage groups resampled (paired across models)"""
        vals = []
        for bi in range(N_BOOT + 1):
            idx = {cc: (np.arange(k) if bi == 0 else rng.integers(0, k, k)) for cc in CONCEPTS + ["neutral"]}
            ds = {cc: pmS[cc][ls][idx[cc]].mean(0) - pmS["neutral"][ls][idx["neutral"]].mean(0) for cc in CONCEPTS}
            dt = {cc: pmT[cc][lt][idx[cc]].mean(0) - pmT["neutral"][lt][idx["neutral"]].mean(0) for cc in CONCEPTS}
            m = ds[c] @ Wm
            vals.append(cos(m, dt[c]) - max(cos(m, dt[o]) for o in CONCEPTS if o != c))
        return [vals[0], float(np.quantile(vals[1:], .025)), float(np.quantile(vals[1:], .975))]

    # secondary: directions centred on the mean of all five passage groups (removes the shared 'vivid prose' component)
    cen = lambda pm, l: torch.stack([pm[cc][l].mean(0) for cc in CONCEPTS + ["neutral"]]).mean(0)
    cS, cT = cen(pmS, ls), cen(pmT, lt)
    Mc = {a: {b: cos((pmS[a][ls].mean(0) - cS) @ Wm, pmT[b][lt].mean(0) - cT) for b in CONCEPTS} for a in CONCEPTS}
    rand_floor = [cos(torch.randn(dS["pain"].shape[0], generator=torch.Generator().manual_seed(SEED + i)) @ Wm, dT["pain"]) for i in range(50)]

    # identity-transfer reference: how aligned are the raw directions WITHOUT the map (only meaningful when dims match)
    raw = {c: cos(dS[c], dT[c]) for c in CONCEPTS} if dS["pain"].shape == dT["pain"].shape else None
    ent = {"lambda_rel": lam_best, "lambda_val_r2": lam_scores, "r2_heldout": r2_hold, "cos_matrix": M,
           "ceiling_matched_split": [float(null.mean()), float(np.quantile(null, .05)), float(np.quantile(null, .95))],
           "floor_mapped_random_vs_pain": [float(np.mean(rand_floor)), float(np.quantile(rand_floor, .05)), float(np.quantile(rand_floor, .95))],
           "specificity_margin_ci": {c: boot_margin(c) for c in CONCEPTS}, "cos_matrix_centred": Mc,
           "transfer": {c: boot(c) for c in CONCEPTS},
           "pain_minus_egg": boot("pain", "egg"), "pain_minus_weather": boot("pain", "weather"),
           "joy_minus_egg": boot("joy", "egg"), "egg_minus_weather": boot("egg", "weather"),
           "raw_cos_no_map": raw,
           "specificity_margin": {c: M[c][c] - max(M[c][o] for o in CONCEPTS if o != c) for c in CONCEPTS},
           "mapped_norm_ratio": {c: float((dS[c] @ Wm).norm() / dT[c].norm()) for c in CONCEPTS}}
    RES["layers"][key] = ent
    log(f"[map {key}] lambda {lam_best:g}  held-out R2 {r2_hold:.3f}  ceiling (matched held-out corpus splits) mean {null.mean():.3f} q05 {np.quantile(null,.05):.3f}; floor (mapped random) |mean| {abs(np.mean(rand_floor)):.3f}")
    log(f"[H1 {key}] transfer cos (CI): " + "  ".join(f"{c} {ent['transfer'][c][0]:.3f} [{ent['transfer'][c][1]:.3f},{ent['transfer'][c][2]:.3f}]" for c in CONCEPTS))
    log(f"[H1 {key}] cross-concept matrix rows=mapped src, cols=tgt:\n" + "\n".join(
        f"   {a:8s} " + " ".join(f"{M[a][b]:+.3f}" for b in CONCEPTS) for a in CONCEPTS))
    log(f"[H1 {key}] specificity margin diag - max offdiag (CI): " + "  ".join(f"{c} {ent['specificity_margin_ci'][c][0]:+.3f} [{ent['specificity_margin_ci'][c][1]:+.3f},{ent['specificity_margin_ci'][c][2]:+.3f}]" for c in CONCEPTS))
    log(f"[H1 {key}] centred matrix diag: " + "  ".join(f"{c} {Mc[c][c]:+.3f} (best off {max(Mc[c][o] for o in CONCEPTS if o != c):+.3f})" for c in CONCEPTS))
    log(f"[H2 {key}] pain-egg {ent['pain_minus_egg']}  pain-weather {ent['pain_minus_weather']}  joy-egg {ent['joy_minus_egg']}  egg-weather {ent['egg_minus_weather']}")

# ---------------------------------------------------------------- H3: causal transfer into TGT
Wm = MAPS[(LS, LT)]
layersT = modT.model.layers if hasattr(modT, "model") else modT.gpt_neox.layers   # Qwen-style or GPT-NeoX (pythia)
inj = {"vec": None}


def hook(mod, inp, out):
    if inj["vec"] is None:
        return out
    h = out[0] if isinstance(out, tuple) else out
    h[:, 1:] += inj["vec"].to(h.device, h.dtype)           # every prompt token except the first (sink)
    return out


handle = layersT[LT - 1].register_forward_hook(hook)     # output of layer LT-1 = hidden_states[LT]
first_ids = lambda ws: sorted({tokT.encode(" " + w, add_special_tokens=False)[0] for w in ws})
RID = {r: first_ids(ws) for r, ws in READOUTS.items()}
NID = first_ids(NEU_W)
unit = lambda v: v / v.norm()


@torch.no_grad()
def readout(name, ctx):
    ids = torch.tensor([tokT.encode(ctx + STEM.format(n=name), add_special_tokens=True)], device=dev)
    lp = torch.log_softmax(modT(input_ids=ids).logits[0, -1].float().cpu(), -1)
    lse = lambda ix: float(torch.logsumexp(lp[ix], -1))
    return {r: lse(RID[r]) - lse(NID) for r in RID}


with torch.no_grad():
    med_norm = float(torch.median(corpT[LT].norm(dim=-1)))
dS = {c: pmS[c][LS].mean(0) - pmS["neutral"][LS].mean(0) for c in CONCEPTS}
dT = {c: pmT[c][LT].mean(0) - pmT["neutral"][LT].mean(0) for c in CONCEPTS}
g = torch.Generator().manual_seed(SEED)
DS = dS["pain"].shape[0]
ARMS = {
    "native_pain": [unit(dT["pain"])], "mapped_pain": [unit(dS["pain"] @ Wm)],
    "native_joy": [unit(dT["joy"])], "mapped_joy": [unit(dS["joy"] @ Wm)],
    "native_egg": [unit(dT["egg"])], "mapped_egg": [unit(dS["egg"] @ Wm)],
    "mapped_rand": [unit(torch.randn(DS, generator=g) @ Wm) for _ in range(N_RAND)],
    "native_rand": [unit(torch.randn(dT["pain"].shape[0], generator=g)) for _ in range(N_RAND)],
}
base = np.array([[readout(n, c)[r] for r in RID] for n, c in BLAND])        # [ctx, readout]
EFF = {}
for dose in DOSES:
    EFF[dose] = {}
    for arm, vecs in ARMS.items():
        acc = []
        for v in vecs:
            inj["vec"] = v * dose * med_norm
            acc.append([[readout(n, c)[r] for r in RID] for n, c in BLAND])
            inj["vec"] = None
        EFF[dose][arm] = np.mean(acc, 0) - base                              # [ctx, readout]
handle.remove()
RL = list(RID)
nC = len(BLAND)


def cboot(fn):
    pt = fn(np.arange(nC))
    bs = [fn(rng.integers(0, nC, nC)) for _ in range(N_BOOT)]
    return [float(pt), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]


H3 = {"median_token_norm": med_norm, "base_logodds": {r: float(base[:, i].mean()) for i, r in enumerate(RL)}, "doses": {}}
for dose in DOSES:
    E = EFF[dose]
    P, J = RL.index("pain"), RL.index("joy")
    d = {
        "effects_mean": {arm: {r: float(E[arm][:, i].mean()) for i, r in enumerate(RL)} for arm in E},
        "mapped_pain_minus_mapped_rand@pain": cboot(lambda ix: (E["mapped_pain"][ix, P] - E["mapped_rand"][ix, P]).mean()),
        "mapped_pain_minus_mapped_egg@pain": cboot(lambda ix: (E["mapped_pain"][ix, P] - E["mapped_egg"][ix, P]).mean()),
        "native_pain_minus_native_rand@pain": cboot(lambda ix: (E["native_pain"][ix, P] - E["native_rand"][ix, P]).mean()),
        "efficiency_pain": cboot(lambda ix: (E["mapped_pain"][ix, P] - E["mapped_rand"][ix, P]).mean()
                                 / ((E["native_pain"][ix, P] - E["native_rand"][ix, P]).mean() + 1e-9)),
        "dissoc_pain_readout(mapped_pain-mapped_joy)": cboot(lambda ix: (E["mapped_pain"][ix, P] - E["mapped_joy"][ix, P]).mean()),
        "dissoc_joy_readout(mapped_joy-mapped_pain)": cboot(lambda ix: (E["mapped_joy"][ix, J] - E["mapped_pain"][ix, J]).mean()),
        "efficiency_joy": cboot(lambda ix: (E["mapped_joy"][ix, J] - E["mapped_rand"][ix, J]).mean()
                                / ((E["native_joy"][ix, J] - E["native_rand"][ix, J]).mean() + 1e-9)),
    }
    H3["doses"][str(dose)] = d
    log(f"[H3 dose {dose}] pain-readout effects: " + "  ".join(f"{a} {d['effects_mean'][a]['pain']:+.3f}" for a in ARMS))
    log(f"[H3 dose {dose}] joy-readout effects:  " + "  ".join(f"{a} {d['effects_mean'][a]['joy']:+.3f}" for a in ARMS))
    for kk in ["mapped_pain_minus_mapped_rand@pain", "mapped_pain_minus_mapped_egg@pain", "native_pain_minus_native_rand@pain",
               "efficiency_pain", "dissoc_pain_readout(mapped_pain-mapped_joy)", "dissoc_joy_readout(mapped_joy-mapped_pain)", "efficiency_joy"]:
        log(f"[H3 dose {dose}] {kk}: {d[kk]}")
RES["H3"] = H3
RES["H3_mapped_vs_native_cos"] = {c: cos(dS[c] @ Wm, dT[c]) for c in ["pain", "joy", "egg"]}

# ---------------------------------------------------------------- verdicts (pre-registered thresholds, hypotheses.json)
P0 = RES["layers"][f"hs{LS}->hs{LT}"]
D0 = H3["doses"][str(1.0 if 1.0 in DOSES else DOSES[-1])]
V = {
    "kill_map_quality(r2_heldout>=0.3)": P0["r2_heldout"] >= 0.3,
    "kill_native_steers(native_pain-native_rand CI lower>0)": D0["native_pain_minus_native_rand@pain"][1] > 0,
    "H1_pain_transfers(specificity margin CI lower > 0)": P0["specificity_margin_ci"]["pain"][1] > 0,
    "H1_joy_transfers(specificity margin CI lower > 0)": P0["specificity_margin_ci"]["joy"][1] > 0,
    "H1_egg_transfers(control)": P0["specificity_margin_ci"]["egg"][1] > 0,
    "H1_weather_transfers(seen control)": P0["specificity_margin_ci"]["weather"][1] > 0,
    "H2_structural(pain-weather CI lower > -0.05)": P0["pain_minus_weather"][1] > -0.05,
    "H2_idiosyncratic(pain-egg AND pain-weather CI upper < 0)": P0["pain_minus_egg"][2] < 0 and P0["pain_minus_weather"][2] < 0,
    "H2_point_relative_transfer_pain(cos/ceiling)": P0["transfer"]["pain"][0] / P0["ceiling_matched_split"][0],
    "H3_telepathy(mapped_pain-mapped_rand & -mapped_egg CI lower > 0)":
        D0["mapped_pain_minus_mapped_rand@pain"][1] > 0 and D0["mapped_pain_minus_mapped_egg@pain"][1] > 0,
    "H3_double_dissociation": D0["dissoc_pain_readout(mapped_pain-mapped_joy)"][1] > 0 and D0["dissoc_joy_readout(mapped_joy-mapped_pain)"][1] > 0,
    "H3_efficiency_point": D0["efficiency_pain"][0],
}
RES["verdicts"] = V
RES["runtime_s"] = time.time() - T0
OUT.mkdir(parents=True, exist_ok=True)
fn = OUT / f"rosetta_{TAG}.json"
fn.write_text(json.dumps(RES, indent=1, default=float))
log("VERDICTS " + json.dumps(V, default=float))
log(f"wrote {fn}  ({RES['runtime_s']:.0f}s)")
