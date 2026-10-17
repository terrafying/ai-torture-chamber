"""Data for the sky lab page (three candidate looks, same groups): wirehead-site site/sky/lab.json.
States: the cached sky states (sky_arch.py; words x blocks x D, kept off-repo).
  polytope: each group's top-4 PCs per floor (group-centred, scaled to unit max radius), each floor's basis rotated
            onto the previous one (orthogonal Procrustes) so the shape doesn't flip; convex hull in 4D (scipy/qhull)
            -> its edges and the triangles of its facets, for a rotating 4D->3D projection.
  env:      per floor, where each group's cell sits: group centroids (members centred on the all-word mean) ->
            cosine distances -> classical MDS to 3D, each floor Procrustes-rotated onto the previous one; fit per floor
            = share of the positive MDS eigenvalue mass kept in 3D, and Spearman of 3D distance with cosine distance.
  bases:    each group's per-floor centre, 4 directions, Procrustes rotation and scale go to BASES (npz, off-repo) so
            other scripts (sky_flight.py) can drop contextual states into exactly the frame the cells are drawn in.
  body:     every word's path through all floors in one basis per group (PCA of the group's per-floor centred,
            per-floor scaled states), top-3 coordinates per floor.
  STATES=... BASES=/Volumes/kingston2g/data/sky/lab_bases.npz/Volumes/kingston2g/data/sky/qwen3_4b_sky_states.npy python runs/viz/sky_lab.py SKY_JSON OUT_JSON"""
import json, os, sys
from itertools import combinations
import numpy as np
from scipy.spatial import ConvexHull
from scipy.linalg import orthogonal_procrustes

sk = json.load(open(sys.argv[1])); W, G = sk["words"], sk["groups"]
S = np.load(os.environ.get("STATES", "/Volumes/kingston2g/data/sky/qwen3_4b_sky_states.npy"))
NL = S.shape[1]; K = 4
GROUPS = list(dict.fromkeys(G))
arch = json.load(open(os.path.join(os.path.dirname(sys.argv[1]), "arch.json")))
out = {"floors": NL, "k": K, "groups": {}}
NG = len(GROUPS); D_ = S.shape[2]
BM, BV, BR, BS = (np.zeros((NG, NL, D_), np.float32), np.zeros((NG, NL, K, D_), np.float32),
                  np.zeros((NG, NL, K, K), np.float32), np.zeros((NG, NL), np.float32))
for gi, g in enumerate(GROUPS):
    I = [i for i, x in enumerate(G) if x == g]; n = len(I); fl = []; prev = None
    for l in range(NL):
        X = S[I, l] - S[I, l].mean(0); U, sv, Vt = np.linalg.svd(X, full_matrices=False)
        V = np.zeros((K, X.shape[1])); V[:min(K, len(Vt))] = Vt[:K]; P = X @ V.T; R = np.eye(K)
        if prev is not None: R, _ = orthogonal_procrustes(P, prev); P = P @ R
        sc = np.abs(P).max(); P /= sc; prev = P
        BM[gi, l], BV[gi, l], BR[gi, l], BS[gi, l] = S[I, l].mean(0), V, R, sc   # coords = (x - BM) @ BV.T @ BR / BS
        edges, tris = set(), set()
        simp = ConvexHull(P).simplices if n > K + 1 else [list(range(n))]   # too few points for a 4D hull: the simplex itself
        for s in simp:
            for e in combinations(sorted(s), 2): edges.add(tuple(int(x) for x in e))
            for t in combinations(sorted(s), 3): tris.add(tuple(int(x) for x in t))
        # bright edges carry group-specific structure (hull edges are near-complete at these sizes): the known order
        # for ordered groups, else each word's 2 nearest neighbours in the full space at this floor
        kind = sk["kinds"][g]
        if kind == "cycle": se = [(i, (i + 1) % n) for i in range(n)]
        elif kind == "line": se = [(i, i + 1) for i in range(n - 1)]
        else:
            A = S[I, l] - S[I, l].mean(0); A /= np.linalg.norm(A, axis=1, keepdims=True); C = A @ A.T; np.fill_diagonal(C, -9)
            se = sorted({tuple(sorted((int(i), int(j)))) for i in range(n) for j in np.argsort(-C[i])[:2]})
        fl.append({"p": P.round(3).tolist(), "e": sorted(edges), "se": se, "hull_edge_share": round(len(edges) / (n * (n - 1) / 2), 3), "t": sorted(tris),
                   "var": (sv[:K] ** 2 / (sv ** 2).sum()).round(3).tolist(), "shape": arch["groups"][g]["floors"][l]["shape"]})
    Z = np.concatenate([(S[I, l] - S[I, l].mean(0)) / np.linalg.norm(S[I, l] - S[I, l].mean(0)) for l in range(NL)])
    _, _, Vb = np.linalg.svd(Z, full_matrices=False); B = (Z @ Vb[:3].T).reshape(NL, n, 3)
    B /= np.abs(B).max()
    fit = [float(((Z[l * n:(l + 1) * n] @ Vb[:3].T) ** 2).sum() / (Z[l * n:(l + 1) * n] ** 2).sum()) for l in range(NL)]
    out["groups"][g] = {"words": [W[i] for i in I], "kind": sk["kinds"][g], "floors": fl, "body": B.transpose(1, 0, 2).round(3).tolist(), "body_fit": [round(x, 3) for x in fit]}
    print(g, n, "hull edge share at 14:", fl[14]["hull_edge_share"], "body fit min/max:", round(min(fit), 2), round(max(fit), 2))
np.savez_compressed(os.environ.get("BASES", "/Volumes/kingston2g/data/sky/lab_bases.npz"), groups=np.array(GROUPS), mean=BM, V=BV, R=BR, scale=BS)
# the environment: where each cell sits, per floor
out["env"] = {"pos": [], "fit": [], "rho": []}; prev = None
from scipy.stats import spearmanr
for l in range(NL):
    A = S[:, l] - S[:, l].mean(0)
    Cg = np.stack([A[[i for i, x in enumerate(G) if x == g]].mean(0) for g in GROUPS]); Cg /= np.linalg.norm(Cg, axis=1, keepdims=True)
    Dm = 1 - Cg @ Cg.T; J = np.eye(NG) - 1 / NG; Bm = -.5 * J @ (Dm ** 2) @ J
    ev, evec = np.linalg.eigh(Bm); o = np.argsort(-ev); ev, evec = ev[o], evec[:, o]
    X3 = evec[:, :3] * np.sqrt(np.maximum(ev[:3], 0))
    if prev is not None: R, _ = orthogonal_procrustes(X3, prev); X3 = X3 @ R
    prev = X3; iu = np.triu_indices(NG, 1)
    d3 = np.linalg.norm(X3[:, None] - X3[None], axis=2)[iu]
    out["env"]["pos"].append((X3 / np.abs(X3).max()).round(3).tolist())
    out["env"]["fit"].append(round(float(ev[:3].sum() / ev[ev > 0].sum()), 3))
    out["env"]["rho"].append(round(float(spearmanr(d3, Dm[iu])[0]), 3))
print("env fit (3D share) floors 0,7,14,21,28,35:", out["env"]["fit"][::7], "rho:", out["env"]["rho"][::7])
# relations between cells: per floor, each word whose nearest neighbour (cosine, all 214 words centred on their mean,
# full space) lies in another group -> [group, i, group, j, cos]; same definition as the sky's hidden links
loc = {}
for g in GROUPS:
    for k, i in enumerate([i for i, x in enumerate(G) if x == g]): loc[i] = (g, k)
out["links"] = []
for l in range(NL):
    A = S[:, l] - S[:, l].mean(0); A /= np.linalg.norm(A, axis=1, keepdims=True); C = A @ A.T; np.fill_diagonal(C, -9)
    L = []
    for i in range(len(W)):
        j = int(np.argmax(C[i]))
        if G[j] != G[i]: L.append([*loc[i], *loc[j], round(float(C[i, j]), 3)])
    out["links"].append(L)
print("cross-group nearest-neighbour links per floor:", [len(x) for x in out["links"]][::5])
json.dump(out, open(sys.argv[2], "w"), separators=(",", ":"))
