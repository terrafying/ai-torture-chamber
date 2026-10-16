"""The sky: every concept's state at every floor of Qwen3-4B (residual after each block at the word's last sub-token,
averaged over the concept-surgery engine's 6 neutral sentences, exactly as concept_swap.Swapper.state computes it),
then per floor a 3D frame (PCA over all concepts, aligned to the previous floor so flight is smooth), each concept's
true nearest neighbours at each floor (cosine, full space), and the per-floor basis for projecting live states.
Writes to OUT: sky.json (layout, groups, neighbours, constellation edges) + sky_basis.npy (floors x 3 x D, float16).
    HF_HUB_CACHE=... python runs/exp118/sky/states.py <out_dir>"""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parents[1] / "exp111"))
from tower_cfg import DEV, load
NEUTRAL = ["I was thinking about{X} today.", "We talked about{X} for a while.", "The word was{X}.", "Someone mentioned{X} again.", "It was all about{X}.", "Let me tell you about{X}."]
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True); V = json.load(open(HERE / "vocab.json")); tok, m = load(); NL = m.config.num_hidden_layers
def state(w):
    acc = None
    for t in NEUTRAL:
        a, z = t.split("{X}"); pre = tok(a, add_special_tokens=False)["input_ids"]; wid = tok(" " + w, add_special_tokens=False)["input_ids"]
        ids = pre + wid + tok(z, add_special_tokens=False)["input_ids"]
        with torch.no_grad(): hs = m(torch.tensor([ids], device=DEV), output_hidden_states=True, use_cache=False).hidden_states
        v = torch.stack([h[0, len(pre) + len(wid) - 1].float() for h in hs[1:]]).cpu().numpy(); acc = v if acc is None else acc + v
    return acc / len(NEUTRAL)
words, groups = [], []
for g, d in V.items():
    for w in d["words"]: words.append(w); groups.append(g)
X = np.stack([state(w) for w in words]); print("states", X.shape, flush=True)               # N x floors x D
N = len(words); layout, basis, nbrs, xlinks = [], [], [], []
GL = list(V); gix = {g: [i for i in range(N) if groups[i] == g] for g in GL}
def procrustes(P, Q):   # rotate/flip P to best match Q
    U_, _, Wt = np.linalg.svd(P.T @ Q); return P @ (U_ @ Wt)
prevC, prevL = None, {}
for f in range(NL):
    A = X[:, f]; mu = A.mean(0); An = A - mu; An = An / np.linalg.norm(An, axis=1, keepdims=True)   # unit vectors: direction, not size
    # level 1: constellation centres, PCA to 3D, aligned to the last floor
    Cg = np.stack([An[gix[g]].mean(0) for g in GL]); cm = Cg.mean(0); _, _, Vt = np.linalg.svd(Cg - cm, full_matrices=False)
    PC = (Cg - cm) @ Vt[:3].T
    if prevC is not None: PC = procrustes(PC, prevC)
    PC = PC / (np.abs(PC).max() + 1e-9); prevC = PC
    pos = np.zeros((N, 3))
    for k, g in enumerate(GL):   # level 2: each constellation in its own principal directions, aligned per floor
        M = An[gix[g]] - An[gix[g]].mean(0); _, _, Vg = np.linalg.svd(M, full_matrices=False); L = M @ Vg[:3].T
        if L.shape[1] < 3: L = np.pad(L, ((0, 0), (0, 3 - L.shape[1])))
        if g in prevL: L = procrustes(L, prevL[g])
        prevL[g] = L; L = L / (np.abs(L).max() + 1e-9)
        pos[gix[g]] = PC[k] + L * 0.16 * np.sqrt(len(gix[g]) / 8)
    layout.append(np.round(pos / (np.abs(pos).max() + 1e-9), 4).tolist())
    _, _, VtA = np.linalg.svd(A - mu, full_matrices=False); basis.append(np.concatenate([VtA[:3], mu[None]], 0))
    Cc = An @ An.T; np.fill_diagonal(Cc, -2); nb = [[int(j) for j in np.argsort(-Cc[i])[:5]] for i in range(N)]; nbrs.append(nb)
    xl = []   # hidden links: each star's nearest neighbour outside its constellation, if it is among its 3 nearest overall
    for i in range(N):
        j = next(j for j in np.argsort(-Cc[i]) if groups[j] != groups[i])
        if j in nb[i][:3]: xl.append([i, int(j), round(float(Cc[i, j]), 3)])
    xlinks.append(xl)
# constellation edges at a mid floor (16): ordered groups in their order (cycles closed); sets by a minimum spanning tree
f0 = 16; A = X[:, f0] - X[:, f0].mean(0); An = A / np.linalg.norm(A, axis=1, keepdims=True); edges = []
for g, d in V.items():
    ix = [i for i in range(N) if groups[i] == g]
    if d["kind"] in ("cycle", "line"):
        edges += [[ix[k], ix[k + 1]] for k in range(len(ix) - 1)] + ([[ix[-1], ix[0]]] if d["kind"] == "cycle" else [])
    else:
        Dm = 1 - An[ix] @ An[ix].T; inT = {0}; out = []
        while len(inT) < len(ix):
            best = min(((Dm[a, b], a, b) for a in inT for b in range(len(ix)) if b not in inT)); inT.add(best[2]); out.append([ix[best[1]], ix[best[2]]])
        edges += out
odd = []   # 'hidden depths': stars whose nearest neighbour at a floor is in another constellation
for f in (8, 16, 24):
    for i in range(N):
        j = nbrs[f][i][0]
        if groups[j] != groups[i]: odd.append({"floor": f, "word": words[i], "group": groups[i], "neighbour": words[j], "neighbour_group": groups[j]})
json.dump({"words": words, "groups": groups, "kinds": {g: d["kind"] for g, d in V.items()}, "floors": NL, "layout": layout, "neighbours": nbrs, "edges": edges, "oddities": odd, "xlinks": xlinks},
          open(OUT / "sky.json", "w"), separators=(",", ":"))
np.save(OUT / "sky_basis.npy", np.stack(basis).astype(np.float16)); print("wrote", N, "stars,", len(edges), "edges,", len(odd), "oddities")
