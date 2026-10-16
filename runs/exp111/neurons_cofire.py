"""exp111: the tower's functional map, from behaviour. Weights alone barely predict the circuits' causal edges
(neurons_connectome.py check, partner-shuffled AUC ~0.55), so here neurons are related by how they fire together.
Runs the model over cached FineWeb-Edu documents (corpus_ids.json), records every active neuron's activation / its
corpus range at every token, and keeps each neuron's top-K partners by Pearson correlation across tokens (and the
strongest anti-correlated ones). Active = in some stream frame's top 150, a circuit node, or labelled.
neurons_map.py turns the graph into modules and a layout. Also keeps each neuron's mean (for hold-at-typical edits).
Checks (written to cofire_check.json):
  circuits  |corr| of each cross-floor circuit edge vs the same sender with a shuffled circuit receiver on the same floor
  labels    share of a neuron's top-K partners with its blind-label category, vs that category's base rate
  floors    how far modules spread across floors (a functional map, not just the tower again)
    HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python neurons_cofire.py [n_docs] [K]"""
import json, sys, collections
import numpy as np, torch
from tower_cfg import DATA as D, DEV, load
NDOC = int(sys.argv[1]) if len(sys.argv) > 1 else 80; K = int(sys.argv[2]) if len(sys.argv) > 2 else 32
tok, m = load(); NL, I = m.config.num_hidden_layers, m.config.intermediate_size
C = dict(np.load(D / "corpus_all.npz")); SCALE = np.maximum(np.abs(C["max"]), np.abs(C["min"])).clip(1e-3)
LAB = json.load(open(D / "labels.json")); circ = json.load(open(D / "circuits.json")); st = json.load(open(D / "stream.json"))
act = set()
for p in st["prompts"].values():
    for c in p["conds"].values():
        for f in c["frames"]: act.update(f["i"][:150])
for g in circ: act.update(x["l"] * I + x["n"] for x in g["nodes"])
act.update(int(k.split(":")[0]) * I + int(k.split(":")[1]) for k in LAB)
idx = np.array(sorted(act)); lay = idx // I; neu = idx % I; N = len(idx); pos = {int(g): k for k, g in enumerate(idx)}
sel = {l: (np.where(lay == l)[0], torch.tensor(neu[lay == l], device=DEV), torch.tensor(SCALE[l, neu[lay == l]], device=DEV, dtype=torch.float32))
       for l in range(NL) if (lay == l).any()}
docs = json.load(open(D / "corpus_ids.json"))[-NDOC:]; T = sum(len(d) for d in docs)
X = torch.zeros(T, N, dtype=torch.float16); cur = {"t": 0, "n": 0}
def hook(l):
    def f(mod, inp):
        if l not in sel: return None
        cols, nn, sc = sel[l]; a = inp[0][0, :, nn].float() / sc
        X[cur["t"]:cur["t"] + a.shape[0], cols] = a.half().cpu()
    return f
for l in range(NL): m.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook(l))
with torch.no_grad():
    for di, ids in enumerate(docs):
        m(torch.tensor([ids], device=DEV)); cur["t"] += len(ids)
        if di % 20 == 0: print("doc", di, flush=True)
print("activations", tuple(X.shape), flush=True)
Z = X.float(); Z -= Z.mean(0); Z /= Z.norm(dim=0).clamp(min=1e-6)            # columns: unit-norm, centred -> dot = Pearson
Zd = Z.to(DEV)
nn_i = np.zeros((N, K), np.int32); nn_w = np.zeros((N, K), np.float32); an_i = np.zeros((N, 8), np.int32); an_w = np.zeros((N, 8), np.float32)
for s in range(0, N, 1024):
    R = Zd[:, s:s + 1024].T @ Zd                                               # b x N correlations
    R[torch.arange(R.shape[0]), torch.arange(s, s + R.shape[0])] = -2
    v, i = R.topk(K, dim=1); nn_i[s:s + 1024] = i.cpu().numpy(); nn_w[s:s + 1024] = v.cpu().numpy()
    R[torch.arange(R.shape[0]), torch.arange(s, s + R.shape[0])] = 2
    v, i = (-R).topk(8, dim=1); an_i[s:s + 1024] = i.cpu().numpy(); an_w[s:s + 1024] = -v.cpu().numpy()
print("knn done", flush=True)
np.savez_compressed(D / "cofire.npz", idx=idx, nn_i=nn_i, nn_w=nn_w, an_i=an_i, an_w=an_w,   # saved first: the checks below are optional
                    mean=X.float().mean(0).numpy(), std=X.float().std(0).numpy())
def corr(a, b): return (Zd[:, a] * Zd[:, b]).sum(0).cpu().numpy()

# ---- checks
from scipy.stats import mannwhitneyu
rng = np.random.default_rng(0); chk = {"tokens": int(T), "neurons": int(N), "K": K}
ea, eb, ew, nbl = [], [], [], collections.defaultdict(set)
for g in circ:
    for x in g["nodes"]: nbl[x["l"]].add(pos[x["l"] * I + x["n"]])
    for a, b, w in g["edges"]:
        A, B = g["nodes"][a], g["nodes"][b]
        if A["l"] == B["l"]: continue
        ea.append(pos[A["l"] * I + A["n"]]); eb.append(pos[B["l"] * I + B["n"]]); ew.append(abs(w))
ea, eb, ew = map(np.array, (ea, eb, ew)); nbl = {l: np.array(sorted(v)) for l, v in nbl.items()}
rb = np.array([rng.choice(nbl[lay[b]]) for b in eb])
e, r = np.abs(corr(ea, eb)), np.abs(corr(ea, rb)); top = ew >= np.quantile(ew, .9)
chk["circuits"] = {"edges": int(len(ea)), "auc_vs_shuffled_receiver": round(float(mannwhitneyu(e, r).statistic / len(e) / len(r)), 3),
                   "auc_top10pct_edges": round(float(mannwhitneyu(e[top], r).statistic / top.sum() / len(r)), 3),
                   "median_abs_corr_edge": round(float(np.median(e)), 3), "median_abs_corr_shuffled": round(float(np.median(r)), 3)}
cat = np.array([LAB.get(f"{l}:{n}", {}).get("cat") for l, n in zip(lay, neu)], dtype=object)
has = np.array([c is not None for c in cat]); base = collections.Counter(cat[has]); nlab = has.sum(); rows = {}
for c0, cnt in base.items():
    me = np.where(cat == c0)[0]
    same = [(cat[nn_i[k]] == c0).sum() / max(1, has[nn_i[k]].sum()) for k in me if has[nn_i[k]].any()]
    rows[c0] = {"n": int(cnt), "partner_share_same": round(float(np.mean(same)), 3) if same else None, "base_rate": round(cnt / nlab, 3)}
chk["labels"] = rows
print(json.dumps(chk, indent=1), flush=True)

json.dump(chk, open(D / "cofire_check.json", "w"), indent=1)
print("wrote cofire.npz + cofire_check.json")
