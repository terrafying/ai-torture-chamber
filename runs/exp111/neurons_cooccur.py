"""exp111: the functional map, third substrate. Dense co-firing (neurons_cofire.py) measures rhythm: its modules sort by
floor and activity level, not meaning. The blind labels describe each neuron's STRONG firings, so here two neurons are
related when they fire strongly on the same tokens: an event is activation >= THR x the neuron's positive corpus max,
counted over cached FineWeb-Edu documents (corpus_ids.json), scored by normalised PMI with a minimum co-count.
Writes cooccur.npz (idx, nn_i, nn_w = top-K partners by NPMI, events per neuron) and prints the same checks as
neurons_map.py: label enrichment of Leiden modules vs a shuffle, and how much floor / event rate explain them.
    HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python neurons_cooccur.py [n_docs] [thr] [K]"""
import json, sys, collections
import numpy as np, torch
from tower_cfg import DATA as D, DEV, load
NDOC = int(sys.argv[1]) if len(sys.argv) > 1 else 800; THR = float(sys.argv[2]) if len(sys.argv) > 2 else .3
K = int(sys.argv[3]) if len(sys.argv) > 3 else 32; MINCO = 4
tok, m = load(); NL, I = m.config.num_hidden_layers, m.config.intermediate_size
C = dict(np.load(D / "corpus_all.npz")); PMAX = np.clip(C["max"], 1e-3, None)
LAB = json.load(open(D / "labels.json")); circ = json.load(open(D / "circuits.json")); st = json.load(open(D / "stream.json"))
act = set()
for p in st["prompts"].values():
    for c in p["conds"].values():
        for f in c["frames"]: act.update(f["i"][:150])
for g in circ: act.update(x["l"] * I + x["n"] for x in g["nodes"])
act.update(int(k.split(":")[0]) * I + int(k.split(":")[1]) for k in LAB)
idx = np.array(sorted(act)); lay = idx // I; neu = idx % I; N = len(idx)
sel = {l: (torch.tensor(np.where(lay == l)[0], device=DEV), torch.tensor(neu[lay == l], device=DEV),
           torch.tensor(THR * PMAX[l, neu[lay == l]], device=DEV, dtype=torch.float32)) for l in range(NL) if (lay == l).any()}
docs = json.load(open(D / "corpus_ids.json"))[:NDOC]
CO = torch.zeros(N, N, device=DEV); EV = torch.zeros(N, device=DEV); cur = {}
def hook(l):
    def f(mod, inp):
        if l in sel:
            cols, nn, th = sel[l]; cur[l] = (cols, (inp[0][0, :, nn].float() >= th))
    return f
for l in range(NL): m.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook(l))
T = 0
with torch.no_grad():
    for di, ids in enumerate(docs):
        m(torch.tensor([ids], device=DEV)); B = torch.zeros(len(ids), N, device=DEV)
        for l, (cols, e) in cur.items(): B[:, cols] = e.float()
        CO += B.T @ B; EV += B.sum(0); T += len(ids)
        if di % 50 == 0: print("doc", di, "tokens", T, "events/token", round(float(B.sum() / len(ids)), 1), flush=True)
ev = EV.cpu().numpy(); print("neurons with >=10 events:", int((ev >= 10).sum()), "of", N, flush=True)
p_i = EV / T; nn_i = np.zeros((N, K), np.int32); nn_w = np.zeros((N, K), np.float32)
for s in range(0, N, 2048):
    c = CO[s:s + 2048]; pij = c / T
    npmi = torch.log(pij / (p_i[s:s + 2048, None] * p_i[None, :]).clamp(min=1e-12)) / (-torch.log(pij.clamp(min=1e-12)))
    npmi[c < MINCO] = -1; npmi[torch.arange(c.shape[0]), torch.arange(s, s + c.shape[0])] = -1
    v, i = npmi.topk(K, dim=1); nn_i[s:s + 2048] = i.cpu().numpy(); nn_w[s:s + 2048] = v.cpu().numpy()
np.savez_compressed(D / "cooccur.npz", idx=idx, nn_i=nn_i, nn_w=nn_w, events=ev, tokens=T, thr=THR)
print("wrote cooccur.npz", flush=True)

# ---- checks (same as neurons_map.py)
import igraph as ig, leidenalg
labd = np.array([LAB.get(f"{l}:{n}", {}).get("cat", "") for l, n in zip(lay, neu)], dtype=object); has = labd != ""
BASE = collections.Counter(labd[has]); NB = has.sum(); rng = np.random.default_rng(0)
ok = nn_w > 0                                                         # partners with a real positive association
nb = [((labd[nn_i[k][ok[k]]] == labd[k]) & has[nn_i[k][ok[k]]]).sum() / max(1, has[nn_i[k][ok[k]]].sum()) for k in np.where(has & (ok.sum(1) >= 5))[0]]
chk = {"tokens": int(T), "thr": THR, "neurons": int(N), "with_10_events": int((ev >= 10).sum()),
       "partner_same_category": round(float(np.mean(nb)), 3),
       "partner_same_category_expected": round(float(sum((v / NB) ** 2 for v in BASE.values())), 3)}
def enrich(mm, ks):
    out = []
    for k in ks:
        me = (mm == k) & has
        if me.sum() >= 5: cc = collections.Counter(labd[me]); out.append(max(v / me.sum() / (BASE[c] / NB) for c, v in cc.items()))
    return float(np.median(out)) if out else None
def eta2(x, mm):
    g = collections.defaultdict(list)
    for a, v in zip(mm, x): g[a].append(v)
    return float(sum(len(v) * (np.mean(v) - x.mean()) ** 2 for v in g.values()) / ((x - x.mean()) ** 2).sum())
src = np.repeat(np.arange(N), K); dst = nn_i.reshape(-1); w = nn_w.reshape(-1); keep = w > 0.05
G = ig.Graph(n=N, edges=np.stack([src[keep], dst[keep]], 1).tolist()); G.es["weight"] = w[keep].tolist(); G.simplify(combine_edges="max")
live = ev >= 10
for res in (1, 5, 20):
    mod = np.array(leidenalg.find_partition(G, leidenalg.RBConfigurationVertexPartition, weights="weight", resolution_parameter=res, seed=0).membership)
    mod[~live] = -1; sizes = collections.Counter(mod[live].tolist()); big = [k for k, v in sizes.items() if v >= 30]
    chk[f"res{res}"] = {"modules_30plus": len(big), "neurons_in": int(sum(sizes[k] for k in big)),
        "enrich": enrich(mod, big), "enrich_shuffled": enrich(np.where(live, rng.permutation(mod), -1), big),
        "floor_eta2": round(eta2(lay[live].astype(float), mod[live]), 3), "rate_eta2": round(eta2(np.log1p(ev[live]), mod[live]), 3),
        "median_floor_span": float(np.median([np.subtract(*np.percentile(lay[mod == k], [90, 10])) for k in big])) if big else None}
    print(res, chk[f"res{res}"], flush=True)
json.dump(chk, open(D / "cooccur_check.json", "w"), indent=1)
print(json.dumps(chk))
