"""Data for the sky lab page (three candidate looks, same groups): wirehead-site site/sky/lab.json.
States: the cached sky states (sky_arch.py; words x blocks x D, kept off-repo).
  polytope: each group's top-4 PCs per floor (group-centred, scaled to unit max radius), each floor's basis rotated
            onto the previous one (orthogonal Procrustes) so the shape doesn't flip; convex hull in 4D (scipy/qhull)
            -> its edges and the triangles of its facets, for a rotating 4D->3D projection.
  body:     every word's path through all floors in one basis per group (PCA of the group's per-floor centred,
            per-floor scaled states), top-3 coordinates per floor.
  STATES=/Volumes/kingston2g/data/sky/qwen3_4b_sky_states.npy python runs/viz/sky_lab.py SKY_JSON OUT_JSON"""
import json, os, sys
from itertools import combinations
import numpy as np
from scipy.spatial import ConvexHull
from scipy.linalg import orthogonal_procrustes

sk = json.load(open(sys.argv[1])); W, G = sk["words"], sk["groups"]
S = np.load(os.environ.get("STATES", "/Volumes/kingston2g/data/sky/qwen3_4b_sky_states.npy"))
NL = S.shape[1]; K = 4
GROUPS = ["months", "weekdays", "numbers", "faith", "emotions"]
arch = json.load(open(os.path.join(os.path.dirname(sys.argv[1]), "arch.json")))
out = {"floors": NL, "k": K, "groups": {}}
for g in GROUPS:
    I = [i for i, x in enumerate(G) if x == g]; n = len(I); fl = []; prev = None
    for l in range(NL):
        X = S[I, l] - S[I, l].mean(0); U, sv, Vt = np.linalg.svd(X, full_matrices=False)
        P = X @ Vt[:K].T
        if prev is not None: R, _ = orthogonal_procrustes(P, prev); P = P @ R
        P /= np.abs(P).max(); prev = P
        h = ConvexHull(P); edges, tris = set(), set()
        for s in h.simplices:
            for e in combinations(sorted(s), 2): edges.add(tuple(int(x) for x in e))
            for t in combinations(sorted(s), 3): tris.add(tuple(int(x) for x in t))
        fl.append({"p": P.round(3).tolist(), "e": sorted(edges), "t": sorted(tris),
                   "var": (sv[:K] ** 2 / (sv ** 2).sum()).round(3).tolist(), "shape": arch["groups"][g]["floors"][l]["shape"]})
    Z = np.concatenate([(S[I, l] - S[I, l].mean(0)) / np.linalg.norm(S[I, l] - S[I, l].mean(0)) for l in range(NL)])
    _, _, Vb = np.linalg.svd(Z, full_matrices=False); B = (Z @ Vb[:3].T).reshape(NL, n, 3)
    B /= np.abs(B).max()
    out["groups"][g] = {"words": [W[i] for i in I], "kind": sk["kinds"][g], "floors": fl, "body": B.transpose(1, 0, 2).round(3).tolist()}
    print(g, n, "edges/tris at 14:", len(fl[14]["e"]), len(fl[14]["t"]))
json.dump(out, open(sys.argv[2], "w"), separators=(",", ":"))
