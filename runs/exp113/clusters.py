"""exp113 step 2 (hypotheses.json): which neurons' strong firings fall into unrelated-looking clusters.
Strong = >= 0.30 x the neuron's corpus max; eligible = >= 12 strong among the top 32 (top32.npz). Each strong firing's
context (24 tokens before, the token, 6 after) is embedded once per corpus position with all-MiniLM-L6-v2, then each
eligible neuron's contexts are clustered (average linkage, cosine distance cut 0.70). Multi-cluster = >= 2 clusters of
>= 4 with centroid cosine < 0.30. Same-token = one firing token >= 50% of strong firings. Script-filtered = label
category language/script or >= 30% non-Latin firing tokens. Writes runs/exp113/base_rate.json and candidates.json
(top 40 multi-cluster, different-token, unfiltered, by (1 - centroid cos) x mean strong value / max).
    python runs/exp113/clusters.py"""
import json, sys, collections, unicodedata
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parent / "exp111"))
from tower_cfg import DATA as D, load
from scipy.cluster.hierarchy import linkage, fcluster
STRONG, MIN_N, CUT, MIN_CL, MAX_COS, SAME, NONLAT, TOPN = .30, 12, .70, 4, .30, .50, .30, 40
T = np.load(D / "top32.npz"); VAL, IDX, CTX = T["val"].astype(np.float32), T["idx"], int(T["ctx"])
C = np.load(D / "corpus_all.npz"); MAX = C["max"].reshape(-1); NL, I = C["max"].shape
LAB = json.load(open(D / "labels.json")); docs = json.load(open(D / "corpus_ids.json")); tok = load(tokenizer_only=True)
strong = VAL >= STRONG * MAX[:, None]; ns = strong.sum(1); elig = np.where(ns >= MIN_N)[0]
print("neurons", len(VAL), "eligible", len(elig), flush=True)

pos = np.unique(IDX[elig][strong[elig]]); print("unique firing positions", len(pos), flush=True)
def window(g):
    d, t = divmod(int(g), CTX); ids = docs[d]
    return tok.decode(ids[max(0, t - 24): t]), tok.decode(ids[t: t + 1]), tok.decode(ids[t + 1: t + 7])
W = {int(g): window(g) for g in pos}
from sentence_transformers import SentenceTransformer
emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
E = emb.encode([a + b + c for a, b, c in (W[int(g)] for g in pos)], batch_size=512, normalize_embeddings=True, show_progress_bar=False)
row = {int(g): k for k, g in enumerate(pos)}; print("embedded", flush=True)

def latin(s):
    s = s.strip()
    return all(ch.isascii() or "LATIN" in unicodedata.name(ch, "") or unicodedata.category(ch)[0] in "PSZN" for ch in s) if s else True
out = []; counts = collections.Counter()
for k in elig:
    l, n = divmod(int(k), I); gs = IDX[k][strong[k]]; vs = VAL[k][strong[k]]; X = E[[row[int(g)] for g in gs]]
    toks = [W[int(g)][1].strip().lower() for g in gs]; tc = collections.Counter(toks)
    same = tc.most_common(1)[0][1] / len(toks) >= SAME
    lab = LAB.get(f"{l}:{n}"); script = (lab or {}).get("cat") == "language/script" or sum(not latin(W[int(g)][1]) for g in gs) / len(gs) >= NONLAT
    cl = fcluster(linkage(X, "average", metric="cosine"), CUT, "distance")
    big = [c for c, v in collections.Counter(cl).most_common() if v >= MIN_CL]
    multi = False; ccos = None
    if len(big) >= 2:
        a, b = X[cl == big[0]].mean(0), X[cl == big[1]].mean(0); ccos = float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)); multi = ccos < MAX_COS
    counts["eligible"] += 1; counts["multi"] += multi; counts["multi_same_token"] += multi and same
    counts["multi_script"] += multi and script and not same; counts["multi_candidate"] += multi and not same and not script
    if multi and not same and not script:
        score = (1 - ccos) * float(vs.mean() / MAX[k])
        clusters = []
        for c in big:
            mem = np.where(cl == c)[0]
            clusters.append([{"pre": W[int(gs[j])][0][-160:], "tok": W[int(gs[j])][1], "post": W[int(gs[j])][2][:60], "v": round(float(vs[j] / MAX[k]), 3)} for j in mem])
        rest = [j for j in range(len(gs)) if cl[j] not in big]
        out.append({"k": int(k), "l": l, "n": n, "score": round(score, 4), "centroid_cos": round(ccos, 3), "n_strong": int(len(gs)),
                    "label": (lab or {}).get("label"), "cat": (lab or {}).get("cat"), "top_tokens": tc.most_common(5), "clusters": clusters,
                    "loose": [{"pre": W[int(gs[j])][0][-160:], "tok": W[int(gs[j])][1], "post": W[int(gs[j])][2][:60], "v": round(float(vs[j] / MAX[k]), 3)} for j in rest]})
out.sort(key=lambda x: -x["score"])
base = {k: int(v) for k, v in counts.items()}
base["multi_share"] = round(counts["multi"] / max(1, counts["eligible"]), 4)
base["H1_multi_share_ge_10pct"] = base["multi_share"] >= .10
by_floor = collections.Counter(x["l"] for x in out)
base["candidates_by_floor"] = dict(sorted(by_floor.items()))
json.dump(base, open(HERE / "base_rate.json", "w"), indent=1)
json.dump(out[:TOPN], open(HERE / "candidates.json", "w"), indent=1, ensure_ascii=False)
json.dump(out, open(D / "exp113_all_candidates.json", "w"), ensure_ascii=False)
print(json.dumps(base))
for x in out[:15]: print(x["l"], x["n"], x["score"], x["top_tokens"][:3], "|", x["label"])
