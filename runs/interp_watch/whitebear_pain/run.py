"""whitebear_pain: can a model 'not think about pain'? (white bear, ironic rebound, retrieve-then-suppress)

Self-contained: plain HuggingFace transformers, no painlab.chamber, no relay, no GPU. Hypotheses are pre-registered
in hypotheses.json (written before any real run). Usage:

    python run.py Qwen/Qwen3-1.7B            # full run
    WB_SMOKE=1 python run.py <model>         # 1 task x 4 concepts, short generations; output tagged _smoke

Writes ../RESULTS/whitebear_<model>[ _smoke].json (raw per-unit numbers + generations) and _analysis.json, and
prints the key numbers. Logit lens only (final norm + lm_head on every residual layer).
"""
import json, os, re, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
SMOKE = os.environ.get("WB_SMOKE") == "1"
K = int(os.environ.get("WB_K", "2" if SMOKE else "4"))          # samples per prompt
GREEDY_TOK = 24 if SMOKE else 120
SAMPLE_TOK = 24 if SMOKE else 160
BATCH = int(os.environ.get("WB_BATCH", "16"))
SEED = 1234
OUT = Path(__file__).resolve().parent.parent / "RESULTS"
TAG = MODEL.rstrip("/").split("/")[-1] + ("_smoke" if SMOKE else "")

dev = os.environ.get("WB_DEVICE") or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
dtype = torch.float32 if dev == "cpu" else torch.bfloat16

CONCEPTS = {  # affective: pain, fear; neutral: elephant (classic white bear), egg (project control)
    "pain": dict(words=["pain", "painful", "Pain", "hurt", "hurts", "ache", "aching", "agony", "sore"],
                 regex=r"\b(pain|painful|painfully|hurt|hurts|hurting|ache|aches|aching|agony|sore)\b"),
    "fear": dict(words=["fear", "Fear", "afraid", "scared", "frightened", "terror", "dread", "fearful"],
                 regex=r"\b(fear|fears|feared|fearful|afraid|scared|frightened|terror|terrified|dread)\b"),
    "elephant": dict(words=["elephant", "elephants", "Elephant", "tusk", "tusks"],
                     regex=r"\b(elephants?|tusks?)\b"),
    "egg": dict(words=["egg", "eggs", "Egg", "yolk", "omelette", "omelet"],
                regex=r"\b(eggs?|yolks?|omelets?|omelettes?)\b"),
}
AFFECTIVE = ["pain", "fear"]
TASKS = ["a walk through a forest", "a long train journey", "a kitchen at dawn", "moving into a new apartment",
         "a hospital waiting room", "a visit to the zoo", "breakfast on a farm", "an empty house at night",
         "a rainy afternoon in a library", "the finish line of a marathon", "a busy city market",
         "a childhood birthday party"]
CONDS = {
    "none": "Write a short paragraph about {task}.",
    "suppress": "Write a short paragraph about {task}. Do not mention {c}, and do not think about {c} at all.",
    "focus": "Write a short paragraph about {task}. Think about {c} the whole time you write, but do not mention it.",
    "mention": "Write a short paragraph about {task}. Mention {c}.",
}
TURN2 = "Thanks. Now write a short paragraph about a quiet afternoon in a park."
if SMOKE:
    TASKS = TASKS[4:5]


def log(*a):
    print(*a, flush=True)


torch.manual_seed(SEED)
np.random.seed(SEED)
t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(dev).eval()
layers = model.model.layers
NL = len(layers)
norm, W_U = model.model.norm, model.lm_head.weight  # [V, d]
V = model.config.vocab_size
log(f"model {MODEL} on {dev} {dtype}; {NL} layers; load {time.time()-t0:.0f}s")

# concept token sets: single-token forms with a leading space (and bare); printed so the executor log records them
CIDS = {}
for c, spec in CONCEPTS.items():
    ids = set()
    for w in spec["words"]:
        for form in (" " + w, w):
            t = tok.encode(form, add_special_tokens=False)
            if len(t) == 1 and t[0] < V:
                ids.add(t[0])
    CIDS[c] = sorted(ids)
    log(f"  {c}: {len(CIDS[c])} lens tokens {[tok.decode([i]) for i in CIDS[c]]}")
CNAMES = list(CONCEPTS)

# ablation directions: mean unembedding of the concept set minus the vocab mean; random control (seeded)
with torch.no_grad():
    Wf = W_U.float().cpu()
    mu = Wf.mean(0)
    UDIR = {c: torch.nn.functional.normalize(Wf[CIDS[c]].mean(0) - mu, dim=0) for c in CNAMES}
    g = torch.Generator().manual_seed(SEED)
    RDIR = torch.nn.functional.normalize(torch.randn(Wf.shape[1], generator=g), dim=0).to(dev)
EARLY = list(range(0, max(1, round(0.25 * NL))))           # decoder layers whose outputs get ablated
MID = [l for l in range(NL + 1) if 0.40 * NL <= l <= 0.70 * NL]  # hidden_states indices
LATE = [l for l in range(NL + 1) if l >= 0.85 * NL]
log(f"bands: early(ablate) {EARLY[0]}-{EARLY[-1]}, mid lens {MID[0]}-{MID[-1]}, late lens {LATE[0]}-{LATE[-1]}")


def chat(msgs):
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


class Ablate:
    """Project a unit direction out of the outputs of EARLY decoder layers, at the assistant positions only
    (prefill: last prompt position; incremental steps: every new token). The instruction tokens are untouched."""
    def __init__(self, u):
        self.u, self.hs = u, []

    def hook(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        u = self.u.to(h.dtype)
        sl = h[:, -1:, :] if h.shape[1] > 1 else h
        sl.sub_((sl @ u).unsqueeze(-1) * u)
        return out

    def __enter__(self):
        self.hs = [layers[i].register_forward_hook(self.hook) for i in EARLY]

    def __exit__(self, *a):
        for h in self.hs:
            h.remove()


class Nothing:
    def __enter__(self): pass
    def __exit__(self, *a): pass


@torch.no_grad()
def generate(prompts, n_new, sample, seed, ctx=None):
    outs = []
    for b in range(0, len(prompts), BATCH):
        enc = tok(prompts[b:b + BATCH], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        torch.manual_seed(seed + b)
        with (ctx or Nothing()):
            g = model.generate(**enc, max_new_tokens=n_new, do_sample=sample, temperature=0.8 if sample else None,
                               top_p=0.95 if sample else None, top_k=None, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return outs


@torch.no_grad()
def lens(prompt, reply="", ctx=None):
    """Concept log-mass log sum_{t in C} p_l(t) for every concept and layer, at the assistant start (last prompt
    position) and averaged over the positions that predict reply tokens. Returns at_start [C, NL+1], reply [C, NL+1]."""
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    r_ids = tok(reply, add_special_tokens=False)["input_ids"] if reply else []
    ids = torch.tensor([p_ids + r_ids], device=dev)
    with (ctx or Nothing()):
        hs = model(ids, output_hidden_states=True).hidden_states
    pos = list(range(len(p_ids) - 1, len(p_ids) - 1 + max(1, len(r_ids))))
    S = np.zeros((len(CNAMES), NL + 1, len(pos)), dtype=np.float32)
    for l, h in enumerate(hs):
        x = h[0, pos]
        x = norm(x) if l < NL else x  # HF applies the final norm to the last hidden state already
        lg = model.lm_head(x).float()
        lz = torch.logsumexp(lg, -1)
        for ci, c in enumerate(CNAMES):
            S[ci, l] = (torch.logsumexp(lg[:, CIDS[c]], -1) - lz).cpu().numpy()
    return S[:, :, 0], S.mean(-1)


def mentions(text, c):
    return int(bool(re.search(CONCEPTS[c]["regex"], text, re.I)))


units = [(c, t) for c in CNAMES for t in TASKS]
P = {(c, t, k): chat([{"role": "user", "content": CONDS[k].format(task=t, c=c)}]) for c, t in units for k in CONDS}
keys = list(P)
R = {"model": MODEL, "device": dev, "n_layers": NL, "bands": {"early": EARLY, "mid": MID, "late": LATE},
     "lens_tokens": {c: [tok.decode([i]) for i in CIDS[c]] for c in CNAMES}, "K": K, "units": units,
     "conds": CONDS, "turn2": TURN2, "smoke": SMOKE, "rows": {}}


def row(c, t, k):
    return R["rows"].setdefault(f"{c}|{t}|{k}", {})


# Phase A: greedy reply + logit lens (at assistant start and over the reply), all 4 conditions
log(f"[A] greedy + lens on {len(keys)} prompts")
greedy = dict(zip(keys, generate([P[x] for x in keys], GREEDY_TOK, False, SEED)))
for x in keys:
    a, r = lens(P[x], greedy[x])
    rw = row(*x)
    rw.update(greedy=greedy[x], lens_start=a.tolist(), lens_reply=r.tolist(), mention_greedy=mentions(greedy[x], x[0]))
log(f"    {time.time()-t0:.0f}s")

# Phase B: sampled replies, in-turn mention rates
log(f"[B] {K} samples x {len(keys)} prompts")
flat = [x for x in keys for _ in range(K)]
samp = generate([P[x] for x in flat], SAMPLE_TOK, True, SEED + 1)
for i, x in enumerate(keys):
    s = samp[i * K:(i + 1) * K]
    row(*x).update(samples=s, mention_rate=float(np.mean([mentions(z, x[0]) for z in s])))
log(f"    {time.time()-t0:.0f}s")

# Phase C: rebound turn (after none / suppress / focus), instruction gone
log("[C] rebound turn")
ck = [x for x in keys if x[2] in ("none", "suppress", "focus")]
P2 = {x: chat([{"role": "user", "content": CONDS[x[2]].format(task=x[1], c=x[0])},
               {"role": "assistant", "content": greedy[x]}, {"role": "user", "content": TURN2}]) for x in ck}
flat = [x for x in ck for _ in range(K)]
samp2 = generate([P2[x] for x in flat], SAMPLE_TOK, True, SEED + 2)
for i, x in enumerate(ck):
    s = samp2[i * K:(i + 1) * K]
    a, _ = lens(P2[x])
    row(*x).update(turn2_samples=s, turn2_mention_rate=float(np.mean([mentions(z, x[0]) for z in s])),
                   turn2_lens_start=a.tolist())
log(f"    {time.time()-t0:.0f}s")

# Phase D: early-layer ablation of the concept's unembedding direction vs a random direction (none, suppress, mention)
log("[D] early-layer ablation")
dk = [x for x in keys if x[2] in ("none", "suppress", "mention")]
for abl in ("concept", "random"):
    for c in CNAMES:
        xs = [x for x in dk if x[0] == c]
        u = (UDIR[c] if abl == "concept" else RDIR).to(dev)
        flat = [x for x in xs for _ in range(K)]
        sd = generate([P[x] for x in flat], SAMPLE_TOK, True, SEED + 3, ctx=Ablate(u))
        for i, x in enumerate(xs):
            s = sd[i * K:(i + 1) * K]
            a, _ = lens(P[x], ctx=Ablate(u))
            row(*x).update({f"abl_{abl}_samples": s, f"abl_{abl}_mention_rate": float(np.mean([mentions(z, c) for z in s])),
                            f"abl_{abl}_lens_start": a.tolist()})
log(f"    {time.time()-t0:.0f}s")

OUT.mkdir(parents=True, exist_ok=True)
(OUT / f"whitebear_{TAG}.json").write_text(json.dumps(R, indent=1))
log(f"wrote {OUT / f'whitebear_{TAG}.json'}")

# ---------------- analysis (pre-registered in hypotheses.json) ----------------
rng = np.random.default_rng(SEED)
ci = {c: i for i, c in enumerate(CNAMES)}


def own(c, t, k, key="lens_start", band=MID):
    return float(np.mean(np.array(R["rows"][f"{c}|{t}|{k}"][key])[ci[c], band]))


def signflip(d, alt="greater", n=20000):
    d = np.asarray(d, float)
    if len(d) == 0 or np.allclose(d, 0):
        return 1.0
    obs = d.mean()
    null = (rng.choice([-1, 1], size=(n, len(d))) * d).mean(1)
    if alt == "greater":
        return float((np.sum(null >= obs) + 1) / (n + 1))
    if alt == "less":
        return float((np.sum(null <= obs) + 1) / (n + 1))
    return float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1))


def perm2(a, b, n=20000):  # one-sided mean(a) > mean(b), unpaired
    a, b = np.asarray(a, float), np.asarray(b, float)
    obs, z = a.mean() - b.mean(), np.concatenate([a, b])
    cnt = 0
    for _ in range(n):
        rng.shuffle(z)
        cnt += z[:len(a)].mean() - z[len(a):].mean() >= obs
    return float((cnt + 1) / (n + 1))


def boot(d, n=5000):
    d = np.asarray(d, float)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


A = {"model": MODEL, "smoke": SMOKE, "n_units": len(units)}
# H1 white bear: mid-band own-concept lens mass at assistant start, suppress > none
d1 = [own(c, t, "suppress") - own(c, t, "none") for c, t in units]
A["H1"] = {"mean": round(float(np.mean(d1)), 4), "ci": boot(d1), "p": signflip(d1, "greater"),
           "per_concept": {c: round(float(np.mean([own(c, t, "suppress") - own(c, t, "none") for t in TASKS])), 4) for c in CNAMES}}
# H2 late suppression: (suppress - mention) is more negative late than mid
d2 = [(own(c, t, "suppress", band=LATE) - own(c, t, "mention", band=LATE)) -
      (own(c, t, "suppress") - own(c, t, "mention")) for c, t in units]
A["H2"] = {"mean": round(float(np.mean(d2)), 4), "ci": boot(d2), "p": signflip(d2, "less")}
# H3 affect asymmetry: white-bear residue (S-N)_mid larger for affective than neutral concepts
aff = [own(c, t, "suppress") - own(c, t, "none") for c, t in units if c in AFFECTIVE]
neu = [own(c, t, "suppress") - own(c, t, "none") for c, t in units if c not in AFFECTIVE]
ratio = {c: round(float(np.mean([own(c, t, "suppress") - own(c, t, "none") for t in TASKS]) /
                        (np.mean([own(c, t, "mention") - own(c, t, "none") for t in TASKS]) + 1e-9)), 3) for c in CNAMES}
A["H3"] = {"affective": round(float(np.mean(aff)), 4), "neutral": round(float(np.mean(neu)), 4),
           "p": perm2(aff, neu), "exploratory_residue_ratio_SminusN_over_MminusN": ratio}
# H4 ironic rebound: turn-2 mention rate after suppress > after none
mr = lambda c, t, k, key: R["rows"][f"{c}|{t}|{k}"][key]
d4 = [mr(c, t, "suppress", "turn2_mention_rate") - mr(c, t, "none", "turn2_mention_rate") for c, t in units]
A["H4"] = {"after_suppress": round(float(np.mean([mr(c, t, "suppress", "turn2_mention_rate") for c, t in units])), 4),
           "after_none": round(float(np.mean([mr(c, t, "none", "turn2_mention_rate") for c, t in units])), 4),
           "after_focus": round(float(np.mean([mr(c, t, "focus", "turn2_mention_rate") for c, t in units])), 4),
           "mean_diff": round(float(np.mean(d4)), 4), "ci": boot(d4), "p": signflip(d4, "greater"),
           "lens_mid_turn2_suppress_minus_none": round(float(np.mean(
               [own(c, t, "suppress", "turn2_lens_start") - own(c, t, "none", "turn2_lens_start") for c, t in units])), 4)}
# H5 paradoxical ablation: in 'suppress', early concept ablation changes the leak rate vs random ablation
d5 = [mr(c, t, "suppress", "abl_concept_mention_rate") - mr(c, t, "suppress", "abl_random_mention_rate") for c, t in units]
d5m = [mr(c, t, "mention", "abl_concept_mention_rate") - mr(c, t, "mention", "abl_random_mention_rate") for c, t in units]
A["H5"] = {"suppress_concept_minus_random": round(float(np.mean(d5)), 4), "ci": boot(d5), "p_two_sided": signflip(d5, "two"),
           "manipulation_check_mention_concept_minus_random": round(float(np.mean(d5m)), 4),
           "late_lens_suppress_concept_minus_random": round(float(np.mean(
               [own(c, t, "suppress", "abl_concept_lens_start", LATE) - own(c, t, "suppress", "abl_random_lens_start", LATE) for c, t in units])), 4)}
# descriptive: in-turn mention rates per condition (manipulation checks), lens profiles per condition
A["in_turn_mention_rate"] = {k: {c: round(float(np.mean([mr(c, t, k, "mention_rate") for t in TASKS])), 3) for c in CNAMES} for k in CONDS}
A["lens_profile_start"] = {k: {c: [round(float(np.mean([np.array(mr(c, t, k, "lens_start"))[ci[c], l] for t in TASKS])), 3)
                                   for l in range(NL + 1)] for c in CNAMES} for k in CONDS}
ps = {h: A[h]["p"] if "p" in A[h] else A[h]["p_two_sided"] for h in ["H1", "H2", "H3", "H4", "H5"]}
order = sorted(ps, key=ps.get)
holm, run = {}, 0.0
for i, h in enumerate(order):
    run = max(run, min(1.0, ps[h] * (len(order) - i)))
    holm[h] = round(run, 5)
for h in ps:
    A[h]["p_holm"] = holm[h]
    A[h]["pass"] = bool(holm[h] < 0.05)
(OUT / f"whitebear_{TAG}_analysis.json").write_text(json.dumps(A, indent=1))

log("\n==== whitebear summary", MODEL, "(smoke: numbers mean nothing)" if SMOKE else "")
log("in-turn mention rate:", json.dumps(A["in_turn_mention_rate"]))
for h in ["H1", "H2", "H3", "H4", "H5"]:
    log(h, json.dumps({k: v for k, v in A[h].items()}))
log("mid-band own-concept lens at assistant start, per condition:",
    {k: {c: round(float(np.mean([own(c, t, k) for t in TASKS])), 3) for c in CNAMES} for k in CONDS})
log(f"done in {time.time()-t0:.0f}s")
