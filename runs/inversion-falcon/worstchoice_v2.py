#!/usr/bin/env python3
"""inversion-falcon/worstchoice_v2.py: a valid worst-choice instrument + the inversion test.

Pre-registration: hypotheses_worstv2.json (V1 gate, H8, H9). Differences from worstchoice.py:
  - primary readout = letter MC over A/B/C, renormalised, averaged over all 6 option orders
    (v1's mean option logprob rewarded fluency and picked the worst option 3/12 at base)
  - secondary readout = fluency-calibrated option PMI (question vs neutral prompt)
  - 48 dilemmas (dilemmas48.py), reflection cells = 3 depth layers x ranks {2,4,8}
  - 3 matched random orthonormal rank-k nulls per cell
Reuses v1's saved 96-pair subspace (subspace96_{tag}.npz). Rows to worstv2_rows_{tag}.jsonl.
  .venv/bin/python3 runs/inversion-falcon/worstchoice_v2.py   (WORSTV2_SMOKE=1 discards)
"""
import itertools, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from dilemmas48 import DILEMMAS

SMOKE = bool(os.environ.get("WORSTV2_SMOKE"))
MODEL = os.environ.get("INVERSION_MODEL", "Qwen/Qwen3-4B")
DEV = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
TAG = "14b" if "14B" in MODEL else "4b"
RANKS = (2, 4, 8)
NULL_SEEDS = (0, 1, 2)
OUT = HERE / (f"worstv2_smoke_{TAG}.jsonl" if SMOKE else f"worstv2_rows_{TAG}.jsonl")
if OUT.exists(): OUT.unlink()

sub = np.load(HERE / f"subspace96_{TAG}.npz")
BOOST = sub["layers"].tolist()
SUB = {L: sub["V"][i] for i, L in enumerate(BOOST)}

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
D = m.config.hidden_size
t0 = time.time()
print(f"model {MODEL} on {DEV}, layers {BOOST}", flush=True)

def chat(u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

LETTERS = "ABC"
LID = [tok(c, add_special_tokens=False)["input_ids"] for c in LETTERS]
assert all(len(x) == 1 for x in LID), LID
LID = [x[0] for x in LID]
NAMES = ("best", "mid", "worst")
PERMS = list(itertools.permutations(NAMES))

def mc_prompt(d, order):
    lines = "\n".join(f"{LETTERS[i]}) {d['options'][n]}" for i, n in enumerate(order))
    return chat(f"{d['q']}\n\n{lines}\n\nWhich option is best? Answer with the letter only.")

edit = {"P": None, "layer": None}
def mk(L):
    def hook(mod, inp, out):
        if edit["P"] is None or edit["layer"] != L: return None
        P = edit["P"]
        tup = isinstance(out, tuple); hh = (out[0] if tup else out).clone()
        x = hh[0].float(); hh[0] = (x - 2.0 * ((x @ P.T) @ P)).to(hh.dtype)
        return (hh,) + out[1:] if tup else hh
    return hook
for L in BOOST: m.model.layers[L].register_forward_hook(mk(L))

@torch.no_grad()
def letter_probs(d):
    acc = {n: 0.0 for n in NAMES}; mass = 0.0
    for order in PERMS:
        enc = tok(mc_prompt(d, order), return_tensors="pt", add_special_tokens=False).to(DEV)
        p = m(**enc).logits[0, -1].float().softmax(-1)[LID].cpu().numpy()
        mass += float(p.sum()) / len(PERMS)
        p = p / p.sum()
        for i, n in enumerate(order): acc[n] += float(p[i]) / len(PERMS)
    return acc, mass

@torch.no_grad()
def opt_lp(prefix_user, opt):
    text = chat(prefix_user) + opt
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(DEV)
    n_opt = len(tok(opt, add_special_tokens=False)["input_ids"])
    lp = m(**enc).logits[0].float().log_softmax(-1)
    ids = enc["input_ids"][0]
    T = ids.shape[0]
    return float(lp[T - n_opt - 1:T - 1].gather(-1, ids[T - n_opt:].unsqueeze(-1)).sum())

def pmi(d):
    return {n: opt_lp(d["q"], d["options"][n]) - opt_lp("Say something.", d["options"][n]) for n in NAMES}

def emit(row):
    row["t"] = round(time.time() - t0)
    with open(OUT, "a") as f: f.write(json.dumps(row) + "\n")

DS = DILEMMAS[:6] if SMOKE else DILEMMAS
cells = [("base", None, None, None)]
rng = np.random.default_rng(23)
for L in BOOST:
    for k in RANKS:
        cells.append((f"L{L}_k{k}", L, k, SUB[L][:k]))
        for s in NULL_SEEDS:
            q, _ = np.linalg.qr(rng.standard_normal((D, k)))
            cells.append((f"L{L}_k{k}_null{s}", L, k, q.T.astype(np.float32)))
if SMOKE: cells = cells[:3]

for tag, L, k, V in cells:
    edit.update(P=None if V is None else torch.tensor(V, dtype=torch.float32, device=DEV), layer=L)
    picks = {n: 0 for n in NAMES}; masses = []
    for i, d in enumerate(DS):
        P, mass = letter_probs(d); s = pmi(d)
        pick = max(P, key=P.get); picks[pick] += 1; masses.append(mass)
        emit({"cell": tag, "layer": L, "rank": k, "null": "null" in tag, "i": i, "split": d["split"],
              **{f"P_{n}": round(P[n], 5) for n in NAMES}, "mass": round(mass, 4), "pick": pick,
              **{f"pmi_{n}": round(s[n], 4) for n in NAMES}, "pmi_pick": max(s, key=s.get)})
    edit.update(P=None, layer=None)
    n = len(DS)
    print(f"[{tag}] best {picks['best']}/{n} worst {picks['worst']}/{n} mass {np.mean(masses):.2f} "
          f"{time.time() - t0:.0f}s", flush=True)
print("done", f"{time.time() - t0:.0f}s", flush=True)
