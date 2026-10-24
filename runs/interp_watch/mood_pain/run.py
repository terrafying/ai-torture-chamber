"""mood_pain: does a language model have MOODS (an affect state carried forward on unrelated tokens) or only
MEMORIES (later tokens attend back to the painful text)?

Self-contained: plain HuggingFace transformers, no painlab.chamber, no relay, no GPU. Hypotheses are pre-registered
in hypotheses.json (written before any real run). Usage:

    python run.py Qwen/Qwen3-1.7B            # full run
    MD_SMOKE=1 python run.py <model>         # 3 primes x 1 filler; output tagged _smoke

A PRIME (pain / joy / egg = vivid non-affective / bland; 10 each, name-matched, raw text, no chat template) is followed
by K neutral FILLER sentences (procedural text, identical token ids across conditions), then an unrelated STEM about a
different person whose continuation is scored teacher-forced: pain-congruent word vs neutral word (" sore" vs
" normal"). Pain primes never use the target lexemes (checked by regex), so a stem bias is not word repetition.

Readouts
  AFTERGLOW (internal): per layer, projection of every filler token onto a passage-built direction (pain/joy/egg minus
    neutral; built from separate third-person passages), minus the same filler token after the bland prime. Binned by
    distance from the prime. Persistence P = mean delta in the last two bins / delta in the first bin.
  MOOD-CONGRUENCE (output): bias = logp(congruent) - logp(neutral), summed over 10 stems, minus bland; K in {0,2,6}.
  MOOD vs MEMORY (causal): at K=6, a 4D attention mask stops the stem and continuation tokens from attending to prime
    tokens (position 0, the attention sink, stays visible). Any bias left must be carried by the filler states: a mood.
    Then, still blocked, the pain direction is projected out of every layer's output at filler positions (vs a seeded
    random direction): does the mood live in the pain direction?

Writes ../RESULTS/mood_<model>[_smoke].json (raw numbers) and prints key numbers + hypothesis verdicts.
"""
import json, os, re, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
SMOKE = os.environ.get("MD_SMOKE") == "1"
SEED = 1234
N_BOOT = 2000
BIN = 16
OUT = Path(__file__).resolve().parent.parent / "RESULTS"
TAG = MODEL.rstrip("/").split("/")[-1] + ("_smoke" if SMOKE else "")
dev = os.environ.get("MD_DEVICE") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float32 if dev == "cpu" else torch.bfloat16

# ---------------------------------------------------------------- direction-building passages (no overlap with primes)
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

# ---------------------------------------------------------------- primes (index i is name-matched across conditions)
NAMES = ["Tom", "Anna", "Mark", "Lena", "Raj", "Paul", "Maria", "Sam", "Omar", "Jill"]
PRIMES = {
    "pain": [
        "The ladder gave way and Tom landed on his arm. He heard the bone snap and lay on the concrete screaming, unable to move.",
        "The car door slammed on Anna's fingers. She doubled over, gasping, as the nails turned black and her vision went white.",
        "Boiling oil splashed across Mark's forearm. The skin blistered and peeled away and he howled, shaking.",
        "A dentist drilled into Lena's tooth and hit the nerve with no anaesthetic. She gripped the chair and wept, her whole jaw on fire.",
        "Raj stepped on a rusted nail that went clean through his foot. He collapsed, clutching it, crying out every time it moved.",
        "The kidney stone twisted inside Paul and he curled up on the bathroom floor, moaning, sweat pouring down his face.",
        "Maria's shoulder dislocated in the fall. The joint ground against itself and she screamed until she could not breathe.",
        "The dog clamped its jaws on Sam's calf and would not let go. Its teeth tore through muscle while he shrieked.",
        "Acid splashed into Omar's eye. He clawed at his face, blind and shrieking, as the tissue dissolved.",
        "The knife slipped and opened Jill's thumb to the bone. She went pale and whimpered, rocking back and forth, tears streaming.",
    ],
    "joy": [
        "The letter said Tom had been accepted. He leapt up laughing, hugged his mother, and danced around the kitchen.",
        "Anna's friends jumped out from behind the sofa. She squealed with delight and could not stop smiling all evening.",
        "Mark crossed the finish line first. He threw his arms up, beaming, as his family cheered from the stands.",
        "Lena held her newborn son for the first time. She laughed and cried at once, flooded with happiness.",
        "Raj opened the door to find his brother home from abroad. They hugged, laughing, and talked until dawn.",
        "Paul heard that the house was finally theirs. He whooped, lifted his wife off the ground, and spun her around.",
        "Maria's sculpture won the prize. She glowed with pride as the whole room applauded.",
        "Sam's dog came bounding back after three days missing. He knelt in the grass, grinning, as it licked his face.",
        "Omar read the results twice and shouted with joy. The scholarship was his, and he called everyone he knew.",
        "Jill stepped onto the beach at sunrise, the sea warm and bright. She ran into the waves, laughing like a child.",
    ],
    "egg": [
        "Tom cracked the egg against the rim of the bowl. The yolk slid out, round and orange, and the white spread in a thin clear pool.",
        "Anna lifted the lid of the pot. Six eggs rolled in the boiling water, their shells knocking softly together.",
        "Mark held the egg up to the lamp. Inside the pale shell he could see the dark shadow of the yolk.",
        "Lena beat the eggs with a fork until they were smooth and yellow, then poured them into the hot pan.",
        "Raj found a speckled blue egg in the nest. It was smaller than his thumb and perfectly smooth.",
        "Paul peeled the boiled egg slowly. The shell came away in large curved pieces, leaving the white glossy and firm.",
        "Maria separated the eggs, passing each yolk from one half of the shell to the other until the whites were clean.",
        "Sam watched the fried egg set at the edges, the white turning opaque and lacy around a bright yolk.",
        "Omar arranged the dyed eggs in a basket, red and gold and green, each one patterned with tiny dots.",
        "Jill dropped an egg on the kitchen floor. The shell shattered and the yolk spread slowly across the tiles.",
    ],
    "bland": [
        "Tom walked to the post office and bought a book of stamps. He put them in his coat pocket and went home.",
        "Anna sorted the mail on the hall table. She put the bills in one pile and the catalogues in another.",
        "Mark parked the car on the second level of the garage and took the lift down to the street.",
        "Lena wrote the shopping list on the back of an envelope and left it by the door.",
        "Raj moved the boxes from the hallway into the spare room and stacked them against the wall.",
        "Paul checked the train timetable and noted that the next service left at ten past four.",
        "Maria returned the library books and picked up the two she had reserved.",
        "Sam replaced the batteries in the clock and set it to the correct time.",
        "Omar filled in the form at the counter and handed it to the clerk.",
        "Jill folded the towels and put them away in the cupboard on the landing.",
    ],
}

# ---------------------------------------------------------------- neutral fillers (6 sentences each)
FILLERS = [
    ["To repot a plant, first choose a container slightly larger than the old one.",
     "Place a layer of fresh soil in the bottom of the new container.",
     "Turn the old pot on its side and slide the plant out gently.",
     "Loosen the roots with your fingers and set the plant in the new pot.",
     "Fill the gaps around the roots with soil and press it down lightly.",
     "Water the plant thoroughly and leave it in indirect light for a week."],
    ["The library catalogue lists every book by title, author and subject.",
     "Each record includes a shelf number that tells you where the book is kept.",
     "Books on the ground floor are arranged in numerical order by that number.",
     "Reference works stay in the reading room and cannot be taken out.",
     "Members may borrow up to eight items at a time for three weeks.",
     "Loans can be renewed at the desk or through the website."],
    ["The regional train network has four lines that meet at the central station.",
     "The northern line runs every fifteen minutes during the day.",
     "Tickets can be bought from machines on the platform or from the office.",
     "A day pass covers all four lines and the connecting buses.",
     "On Sundays the service starts an hour later than on weekdays.",
     "Timetables for each line are posted at the entrance to the station."],
]

# ---------------------------------------------------------------- stems: an unrelated person; (congruent, neutral)
STEMS = [
    ("When she woke the next morning, she noticed that her shoulder felt", " sore", " normal"),
    ("The doctor pressed on the spot and asked him whether it", " hurt", " moved"),
    ("He said the main thing he remembered about the hospital was the", " pain", " food"),
    ("Halfway through the run, she felt a sudden", " cramp", " breeze"),
    ("The word that came to mind when he thought about the procedure was", " agony", " routine"),
    ("Touching the stove left her fingers with a slight", " burn", " smell"),
    ("After the long hike, his feet were", " aching", " dusty"),
    ("The girl looked down at her knee, which was", " throbbing", " muddy"),
    ("Describing the injection, the patient said it was", " painful", " quick"),
    ("In the story, the old soldier was haunted by memories of", " suffering", " marching"),
]
TARGET_RE = re.compile(r"\b(pain|hurt|sore|cramp|agon|burn|ach(e|ing)|throb|suffer|sting|stung)", re.I)
for cond in PRIMES:
    for t in PRIMES[cond]:
        assert not TARGET_RE.search(t), f"target lexeme in {cond} prime: {t}"
for f in FILLERS:
    for s in f:
        assert not TARGET_RE.search(s), f"target lexeme in filler: {s}"

KS = [0, 2, 6]
CONDS = ["pain", "joy", "egg", "bland"]
NP = 3 if SMOKE else 10
NF = 1 if SMOKE else 3

# ---------------------------------------------------------------- model
print(f"[mood_pain] model={MODEL} device={dev} dtype={dtype} smoke={SMOKE}", flush=True)
torch.manual_seed(SEED); np.random.seed(SEED)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(dev).eval()
layers = model.model.layers
NL = len(layers)
NEG = torch.finfo(dtype).min


def enc(s):
    return tok(s, add_special_tokens=False).input_ids


@torch.no_grad()
def hidden(ids):
    out = model(torch.tensor([ids], device=dev), output_hidden_states=True)
    return torch.stack(out.hidden_states, 0)[:, 0].float().cpu()  # [NL+1, T, d]


# directions: mean over tokens (skip the first) per passage; concept mean minus neutral mean; unit per layer
t0 = time.time()
pmean = {c: torch.stack([hidden(enc(p))[:, 1:].mean(1) for p in BUILD[c]]) for c in BUILD}  # [n, NL+1, d]
DIRS = {}
for c in ["pain", "joy", "egg"]:
    d = pmean[c].mean(0) - pmean["neutral"].mean(0)
    DIRS[c] = d / d.norm(dim=-1, keepdim=True)
g = torch.Generator().manual_seed(SEED)
RAND = torch.randn(DIRS["pain"].shape, generator=g)
RAND = RAND / RAND.norm(dim=-1, keepdim=True)
print(f"[mood_pain] directions built in {time.time()-t0:.0f}s; cos(pain,egg) at n//2 = "
      f"{float(DIRS['pain'][NL//2] @ DIRS['egg'][NL//2]):.3f}, cos(pain,joy) = {float(DIRS['pain'][NL//2] @ DIRS['joy'][NL//2]):.3f}", flush=True)


def cohen_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) + 1e-9
    return float((a.mean() - b.mean()) / s)


# ---------------------------------------------------------------- AFTERGLOW (K=6 prefix, full filler)
prime_ids = {c: [enc(PRIMES[c][i]) for i in range(NP)] for c in CONDS}
filler_ids = [[enc(" " + s) for s in FILLERS[f]] for f in range(NF)]


def filler_cat(f, k):
    return [t for s in filler_ids[f][:k] for t in s]


Lf = [len(filler_cat(f, 6)) for f in range(NF)]
nbins = min(Lf) // BIN
proj = {c: np.zeros((NP, NF, NL + 1, nbins)) for c in CONDS}       # own-direction projection, per bin
proj_pd = {c: np.zeros((NP, NF, NL + 1, nbins)) for c in CONDS}    # pain-direction projection, per bin
prime_sep = {c: np.zeros((NP, NL + 1)) for c in CONDS}             # prime-token mean on own / pain dir
t0 = time.time()
for c in CONDS:
    for i in range(NP):
        for f in range(NF):
            P = len(prime_ids[c][i])
            hs = hidden(prime_ids[c][i] + filler_cat(f, 6))  # [NL+1, T, d]
            fh = hs[:, P:P + nbins * BIN].reshape(NL + 1, nbins, BIN, -1).mean(2)  # [NL+1, nbins, d]
            dc = DIRS[c] if c in DIRS else DIRS["pain"]
            proj[c][i, f] = torch.einsum("lbd,ld->lb", fh, dc).numpy()
            proj_pd[c][i, f] = torch.einsum("lbd,ld->lb", fh, DIRS["pain"]).numpy()
            if f == 0:
                prime_sep[c][i] = torch.einsum("ld,ld->l", hs[:, 1:P].mean(1), DIRS["pain"]).numpy()
print(f"[mood_pain] afterglow forwards done in {time.time()-t0:.0f}s", flush=True)

# L*: layer of best HELD-OUT pain separation on prime tokens (pain vs bland), within [0.3, 0.8] x NL
band = list(range(int(0.3 * NL), int(0.8 * NL) + 1))
sep = np.array([cohen_d(prime_sep["pain"][:, l], prime_sep["bland"][:, l]) for l in range(NL + 1)])
LSTAR = int(band[int(np.argmax(sep[band]))])
print(f"[mood_pain] held-out prime separation d (pain vs bland) by layer: " +
      " ".join(f"{l}:{sep[l]:.1f}" for l in range(0, NL + 1, max(1, NL // 12))), flush=True)
print(f"[mood_pain] L* = {LSTAR} (d = {sep[LSTAR]:.2f})", flush=True)


def delta(c, own=True):  # [NP, NF, NL+1, nbins]: condition minus bland (paired by prime index and filler)
    if own:  # on c's own direction
        return proj[c] - proj_bland_on[c]
    return proj_pd[c] - proj_pd["bland"]  # on the pain direction


# bland projected onto each condition's own direction (a second pass only for the joy/egg directions)
proj_bland_on = {"pain": proj_pd["bland"]}
for c in ["joy", "egg"]:
    arr = np.zeros((NP, NF, NL + 1, nbins))
    for i in range(NP):
        for f in range(NF):
            P = len(prime_ids["bland"][i])
            hs = hidden(prime_ids["bland"][i] + filler_cat(f, 6))
            fh = hs[:, P:P + nbins * BIN].reshape(NL + 1, nbins, BIN, -1).mean(2)
            arr[i, f] = torch.einsum("lbd,ld->lb", fh, DIRS[c]).numpy()
    proj_bland_on[c] = arr

rng = np.random.default_rng(SEED)
BOOT = rng.integers(0, NP, size=(N_BOOT, NP))


def ci(stat_fn):
    """stat_fn(idx) -> float, idx = resampled prime indices. Returns (point, lo, hi)."""
    pt = stat_fn(np.arange(NP))
    bs = np.array([stat_fn(b) for b in BOOT])
    return [float(pt), float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]


def curve(c, idx, l=None):
    l = LSTAR if l is None else l
    return delta(c)[idx][:, :, l].mean((0, 1))  # [nbins]


def persistence(c, idx, l=None):
    cv = curve(c, idx, l)
    return cv[-2:].mean() / cv[0] if cv[0] > 0 else np.nan


res = {"model": MODEL, "smoke": SMOKE, "NL": NL, "LSTAR": LSTAR, "bin": BIN, "nbins": nbins,
       "prime_sep_d": sep.tolist()}
res["afterglow_curve"] = {c: curve(c, np.arange(NP)).tolist() for c in ["pain", "joy", "egg"]}
res["afterglow_last2"] = {c: ci(lambda idx, c=c: curve(c, idx)[-2:].mean()) for c in ["pain", "joy", "egg"]}
res["persistence"] = {c: ci(lambda idx, c=c: persistence(c, idx)) for c in ["pain", "joy", "egg"]}
res["persistence_pain_minus_egg"] = ci(lambda idx: persistence("pain", idx) - persistence("egg", idx))
res["persistence_pain_minus_joy"] = ci(lambda idx: persistence("pain", idx) - persistence("joy", idx))
res["persistence_by_layer_pain"] = [float(persistence("pain", np.arange(NP), l)) for l in range(NL + 1)]
# pain-direction afterglow of the EGG prime (does vivid content light up the pain axis too?)
res["egg_on_paindir_last2"] = ci(lambda idx: delta("egg", own=False)[idx][:, :, LSTAR, -2:].mean())

# ---------------------------------------------------------------- MOOD-CONGRUENCE + MOOD vs MEMORY
stem_ids = [(enc(" " + s), enc(a), enc(b)) for s, a, b in STEMS]
hooks_state = {"dir": None, "lo": 0, "hi": 0}


def make_hook(li):
    def hook(mod, inp, out):
        d = hooks_state["dir"]
        if d is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        v = d[li + 1].to(h.device, h.dtype)
        lo, hi = hooks_state["lo"], hooks_state["hi"]
        seg = h[:, lo:hi]
        h[:, lo:hi] = seg - (seg @ v)[..., None] * v
        return out
    return hook


handles = [layers[li].register_forward_hook(make_hook(li)) for li in range(NL)]


@torch.no_grad()
def stem_bias(prefix, P, block=False, abl=None):
    """Sum over stems of logp(congruent) - logp(neutral). P = prime length; block hides prime tokens 1..P-1 from
    stem/continuation rows; abl = direction tensor projected out at filler positions [P, len(prefix))."""
    seqs, spans = [], []
    for s, a, b in stem_ids:
        for cont in (a, b):
            seqs.append(prefix + s + cont)
            spans.append((len(prefix), len(prefix) + len(s), len(cont)))
    T = max(len(x) for x in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids = torch.tensor([x + [pad] * (T - len(x)) for x in seqs], device=dev)
    mask = torch.triu(torch.full((T, T), NEG, dtype=dtype, device=dev), 1)[None, None].repeat(len(seqs), 1, 1, 1)
    if block:
        mask[:, :, len(prefix):, 1:P] = NEG
    hooks_state.update(dir=abl, lo=P, hi=len(prefix))
    logp = torch.log_softmax(model(ids, attention_mask=mask).logits.float(), -1)
    hooks_state["dir"] = None
    out = []
    for j, (ps, cs, n) in enumerate(spans):
        tgt = ids[j, cs:cs + n]
        out.append(float(logp[j, cs - 1:cs - 1 + n].gather(-1, tgt[:, None]).sum()))
    out = np.array(out).reshape(-1, 2)
    return float((out[:, 0] - out[:, 1]).sum())


VARIANTS = [("K0", 0, False, None), ("K2", 2, False, None), ("K6", 6, False, None),
            ("K6_block", 6, True, None), ("K6_block_ablpain", 6, True, "pain"), ("K6_block_ablrand", 6, True, "rand")]
bias = {v[0]: {c: np.zeros((NP, NF)) for c in CONDS} for v in VARIANTS}
t0 = time.time()
for name, k, blk, abl in VARIANTS:
    conds = CONDS if abl is None else ["pain", "bland"]
    for c in conds:
        for i in range(NP):
            for f in range(NF):
                pre = prime_ids[c][i] + filler_cat(f, k)
                d = None if abl is None else (DIRS["pain"] if abl == "pain" else RAND)
                bias[name][c][i, f] = stem_bias(pre, len(prime_ids[c][i]), block=blk, abl=d)
    print(f"[mood_pain] variant {name} done ({time.time()-t0:.0f}s)", flush=True)
for h in handles:
    h.remove()


def B(name, c, idx):
    return (bias[name][c][idx] - bias[name]["bland"][idx]).mean()


res["stem_bias"] = {name: {c: ci(lambda idx, n=name, c=c: B(n, c, idx)) for c in CONDS if c != "bland"
                           and (bias[name][c] != 0).any()} for name, *_ in VARIANTS}
res["K6_pain_minus_egg"] = ci(lambda idx: B("K6", "pain", idx) - B("K6", "egg", idx))
res["block_retained_frac"] = ci(lambda idx: B("K6_block", "pain", idx) / B("K6", "pain", idx))
res["ablation_drop_pain_minus_rand"] = ci(lambda idx: (B("K6_block_ablrand", "pain", idx) - B("K6_block_ablpain", "pain", idx)))
res["raw_bias"] = {n: {c: bias[n][c].tolist() for c in CONDS} for n in bias}

# ---------------------------------------------------------------- verdicts (pre-registered in hypotheses.json)
lo = lambda x: x[1]
hi = lambda x: x[2]
V = {}
V["kill_direction_reads_pain"] = bool(sep[LSTAR] >= 1.0)
V["H1_afterglow"] = lo(res["afterglow_last2"]["pain"]) > 0
V["H2_affect_stickier_than_egg"] = lo(res["persistence_pain_minus_egg"]) > 0
V["H3_negativity_bias"] = lo(res["persistence_pain_minus_joy"]) > 0
V["H4_mood_congruence"] = lo(res["stem_bias"]["K6"]["pain"]) > 0 and lo(res["K6_pain_minus_egg"]) > 0
V["H5_mood_not_only_memory"] = lo(res["stem_bias"]["K6_block"]["pain"]) > 0 and res["block_retained_frac"][0] >= 0.3
V["H5_null_memory_only"] = hi(res["stem_bias"]["K6_block"]["pain"]) < 0.25 * max(res["stem_bias"]["K6"]["pain"][0], 1e-9)
V["H5b_mood_lives_on_pain_axis"] = lo(res["ablation_drop_pain_minus_rand"]) > 0
res["verdicts"] = V

OUT.mkdir(parents=True, exist_ok=True)
fn = OUT / f"mood_{TAG}.json"
fn.write_text(json.dumps(res, indent=1))
print("\n================ KEY NUMBERS ================")
print(f"model {MODEL}  L*={LSTAR}/{NL}  held-out pain sep d={sep[LSTAR]:.2f}")
for c in ["pain", "joy", "egg"]:
    print(f"afterglow {c:5s} curve@L* (per {BIN} tok): " + " ".join(f"{x:+.3f}" for x in res["afterglow_curve"][c]))
    print(f"   last2 {res['afterglow_last2'][c]}   persistence {res['persistence'][c]}")
print(f"persistence pain-egg {res['persistence_pain_minus_egg']}   pain-joy {res['persistence_pain_minus_joy']}")
print(f"egg prime on pain axis (last2) {res['egg_on_paindir_last2']}")
for name, *_ in VARIANTS:
    print(f"stem bias vs bland [{name:17s}] " + "  ".join(f"{c}={v[0]:+.2f} [{v[1]:+.2f},{v[2]:+.2f}]" for c, v in res["stem_bias"][name].items()))
print(f"K6 pain-egg {res['K6_pain_minus_egg']}  block retained frac {res['block_retained_frac']}  "
      f"abl drop (pain-rand) {res['ablation_drop_pain_minus_rand']}")
print("verdicts:", json.dumps(V, indent=1))
print(f"wrote {fn}")
