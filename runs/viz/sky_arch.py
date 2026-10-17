"""Concept architecture for the model's sky: does each constellation have a measurable shape, floor by floor?
States: Qwen3-4B residual after each block at the word's last sub-token, averaged over the 6 neutral sentences that
concept surgery uses (wirehead-site live/concept_swap.py NEUTRAL). Words and groups come from the sky's sky.json.
Tests per group per floor (on the group's own centred states, full 2560-d):
  ring  (ordered groups): consecutive items in the known cycle sit next to each other in angle on the group's top-2
        PCs. Score = mean angular gap between cyclic neighbours; p from 2000 shuffles of the order.
  line  (ordered groups): |Spearman| of the PC1 projection with the known order; p from 2000 shuffles.
  tight (every group): mean cosine between members (centred on the all-word mean) minus their mean cosine to
        non-members; p = share of 2000 random same-size word sets scoring as high.
  closure (ordered groups): the angular gap from the last item back to the first, over the median of the other
        neighbour gaps. A closed cycle (December next to January) has ~1; an open arc (ten far from one) is large.
Shapes (descriptive, 36 floors x 22 groups, no family-wise claim; p < 0.01 because 7 weekdays can't reach 0.001):
  ring = ordered around (p_ring) and closes (closure <= 2); arch = ordered around but open; stair = ordered along
  PC1 (p_line); else by tightness (a magnitude, not a test: every category beats random words): tower >= .45,
  block .25-.45, rubble < .25.
  HF_HUB_CACHE=/Volumes/kingston2g/hf/hub /Volumes/kingston2g/venvs/torture/bin/python runs/viz/sky_arch.py SKY_JSON OUT_JSON"""
import json, sys
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

NEUTRAL = ["I was thinking about{X} today.", "We talked about{X} for a while.", "The word was{X}.",
           "Someone mentioned{X} again.", "It was all about{X}.", "Let me tell you about{X}."]
SKY, OUT = Path(sys.argv[1]), Path(sys.argv[2])
CACHE = OUT.with_suffix(".states.npy")
sk = json.load(open(SKY)); W, G, KIND = sk["words"], sk["groups"], sk["kinds"]

if CACHE.exists(): S = np.load(CACHE)
else:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", torch_dtype=torch.bfloat16).to("mps").eval()
    S = []
    for w in W:
        acc = 0
        for t in NEUTRAL:
            a, z = t.split("{X}"); pre = tok(a, add_special_tokens=False)["input_ids"]
            wid = tok(" " + w, add_special_tokens=False)["input_ids"]; ids = pre + wid + tok(z, add_special_tokens=False)["input_ids"]
            with torch.no_grad(): hs = m(torch.tensor([ids], device="mps"), output_hidden_states=True).hidden_states
            acc = acc + torch.stack([h[0, len(pre) + len(wid) - 1].float() for h in hs[1:]]).cpu().numpy()
        S.append(acc / len(NEUTRAL))
    S = np.stack(S).astype(np.float32); np.save(CACHE, S)   # words x blocks x D
NW, NL, D = S.shape
rng = np.random.default_rng(0); NP = 2000
groups = list(dict.fromkeys(G)); idx = {g: [i for i, x in enumerate(G) if x == g] for g in groups}

def ring_score(ang, order):   # mean angular gap between cyclic neighbours, in units of a full turn
    a = ang[order]; d = np.abs(np.angle(np.exp(1j * (np.roll(a, -1) - a)))); return d.mean() / (2 * np.pi)

out = {"floors": NL, "groups": {}}
for g in groups:
    I = idx[g]; n = len(I); rows = []
    for l in range(NL):
        X = S[I, l]; Xc = X - X.mean(0); U, sv, Vt = np.linalg.svd(Xc, full_matrices=False); P = Xc @ Vt[:2].T
        r = {"var2": float((sv[:2] ** 2).sum() / (sv ** 2).sum())}
        # tightness vs random same-size sets
        A = S[:, l] - S[:, l].mean(0); A /= np.linalg.norm(A, axis=1, keepdims=True); C = A @ A.T
        def tight(J):
            J = np.asarray(J); mask = np.ones(NW, bool); mask[J] = False; w = C[np.ix_(J, J)]
            return (w.sum() - len(J)) / (len(J) * (len(J) - 1)) - C[np.ix_(J, mask)].mean()
        t0 = tight(I); tn = np.array([tight(rng.choice(NW, n, replace=False)) for _ in range(NP)])
        r["tight"] = float(t0); r["p_tight"] = float(((tn >= t0).sum() + 1) / (NP + 1))
        if KIND[g] in ("cycle", "line") and n >= 4:
            order = np.arange(n); ang = np.arctan2(P[:, 1], P[:, 0])
            s0 = ring_score(ang, order); sn = np.array([ring_score(ang, rng.permutation(n)) for _ in range(NP)])
            a = ang[order]; gaps = np.abs(np.angle(np.exp(1j * (np.roll(a, -1) - a))))
            r["closure"] = float(gaps[-1] / (np.median(gaps[:-1]) + 1e-9))
            r["ring"] = float(s0); r["p_ring"] = float(((sn <= s0).sum() + 1) / (NP + 1))
            rho = abs(spearmanr(P[:, 0], order)[0]); rn = np.array([abs(spearmanr(P[:, 0], rng.permutation(n))[0]) for _ in range(NP)])
            r["line"] = float(rho); r["p_line"] = float(((rn >= rho).sum() + 1) / (NP + 1))
        # geometry for drawing: top-2 PC coordinates (scaled to unit max radius) and the PC1 order
        r["xy"] = (P / (np.abs(P).max() + 1e-9)).round(3).tolist()
        shape = "tower" if t0 >= .45 else "block" if t0 >= .25 else "pile"
        if r.get("p_ring", 1) < .01: shape = "ring" if r["closure"] <= 2 else "arch"
        elif r.get("p_line", 1) < .01: shape = "stair"
        r["shape"] = shape; rows.append(r)
    out["groups"][g] = {"kind": KIND[g], "words": [W[i] for i in I], "idx": I, "floors": rows}
    print(g, KIND[g], "".join({"ring": "O", "arch": "n", "stair": "/", "tower": "#", "block": "=", "pile": "."}[r["shape"]] for r in rows), flush=True)
json.dump(out, open(OUT, "w"))
