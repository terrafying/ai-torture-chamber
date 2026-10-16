"""exp111: the functional map. From the co-firing graph (cofire.npz: each active neuron's top partners by correlation over
~19k tokens of web text):
  modules  Leiden communities on the graph, across floors. Each module of 30+ neurons is named by an LLM that sees
           only its members' blind labels and floors (anthropic/claude-haiku-4.5), like the districts.
  layout   UMAP of the same graph to 2D. The tower view keeps floor as height and moves each neuron to its map
           position, so a module shows as a column of function, however many floors it spans.
Checks (map_check.json): how many floors modules span; label purity of modules vs chance.
    OPENROUTER_API_KEY=... python neurons_map.py [resolution]"""
import json, os, sys, collections, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tower_cfg import DATA as D
RES = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
Z = np.load(D / "cofire.npz"); idx, nn_i, nn_w = Z["idx"], Z["nn_i"], Z["nn_w"]; N, K = nn_i.shape
C = np.load(D / "corpus_all.npz"); I = C["max"].shape[1]; lay = idx // I; neu = idx % I
LAB = json.load(open(D / "labels.json"))
lab = [LAB.get(f"{l}:{n}") for l, n in zip(lay, neu)]

import igraph as ig, leidenalg
src = np.repeat(np.arange(N), K); dst = nn_i.reshape(-1); w = np.clip(nn_w.reshape(-1), 0, None); keep = w > 0.05
G = ig.Graph(n=N, edges=np.stack([src[keep], dst[keep]], 1).tolist(), directed=False); G.es["weight"] = w[keep].tolist()
G.simplify(combine_edges="max")
part = leidenalg.find_partition(G, leidenalg.RBConfigurationVertexPartition, weights="weight", resolution_parameter=RES, seed=0)
mod = np.array(part.membership, np.int32); sizes = collections.Counter(mod.tolist())
print("modules", len(sizes), "of 30+:", sum(v >= 30 for v in sizes.values()), flush=True)

import umap
kd = np.clip(1 - nn_w, 0, 2).astype(np.float32)
ki = np.concatenate([np.arange(N)[:, None], nn_i], 1); kd = np.concatenate([np.zeros((N, 1), np.float32), kd], 1)
xy = umap.UMAP(n_neighbors=K + 1, min_dist=.1, spread=1.2, random_state=0, precomputed_knn=(ki, kd, None)).fit_transform(np.zeros((N, 2), np.float32))
xy = (xy - xy.mean(0)) / np.abs(xy - xy.mean(0)).max()                       # -1..1
print("umap done", flush=True)

def ask(k):
    me = np.where(mod == k)[0]; rng = np.random.default_rng(k)
    withlab = [j for j in me if lab[j]]; pick = rng.choice(withlab, min(30, len(withlab)), replace=False) if withlab else []
    rows = "\n".join(f"- floor {lay[j]}: {lab[j]['label']}" for j in pick)
    fl = np.percentile(lay[me], [10, 50, 90]).astype(int).tolist()
    if not rows: return {"name": "unlabelled", "why": ""}
    body = {"model": "anthropic/claude-haiku-4.5", "temperature": 0, "messages": [{"role": "user", "content":
        f"These are neurons in a language model (36 floors, 0 = first) that tend to fire together on ordinary text. Each has a blind label "
        f"written from its strongest firings. This group spans floors {fl[0]}-{fl[2]} (median {fl[1]}).\n\n{rows}\n\n"
        "In 2-5 plain words, what job does this group do together? Be specific, not generic (avoid 'language processing'). "
        'Reply with JSON only: {"name": "...", "why": "one short sentence"}'}]}
    for a in range(4):
        try:
            q = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
                headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"})
            txt = json.load(urllib.request.urlopen(q, timeout=120))["choices"][0]["message"]["content"]
            return json.JSONDecoder().raw_decode(txt[txt.index("{"):])[0]
        except Exception: pass
    return {"name": "module", "why": ""}
big = sorted([k for k, v in sizes.items() if v >= 30], key=lambda k: -sizes[k])
with ThreadPoolExecutor(12) as ex: names = dict(zip(big, ex.map(ask, big)))
CATS = sorted({x["cat"] for x in LAB.values()})
mods = []
for k in big:
    me = np.where(mod == k)[0]; cc = collections.Counter(lab[j]["cat"] for j in me if lab[j])
    mods.append({"id": int(k), "n": int(len(me)), "name": str(names[k].get("name", "module"))[:40], "why": str(names[k].get("why", ""))[:160],
                 "floors": np.percentile(lay[me], [10, 50, 90]).astype(int).tolist(), "xy": [round(float(v), 3) for v in xy[me].mean(0)],
                 "cats": dict(cc.most_common(4)), "labelled": int(sum(1 for j in me if lab[j]))})
# checks: floor span and label purity vs a shuffled module assignment
labd = np.array([x["cat"] if x else "" for x in lab], dtype=object); has = labd != ""
BASE = collections.Counter(labd[has]); NB = has.sum()
def enrich(m, ks):   # per module: its most over-represented category's share / that category's base rate (5+ labelled)
    out = []
    for k in ks:
        me = (m == k) & has
        if me.sum() >= 5:
            cc = collections.Counter(labd[me]); out.append(max(v / me.sum() / (BASE[c] / NB) for c, v in cc.items()))
    return float(np.median(out)) if out else None
sh = mod.copy(); np.random.default_rng(0).shuffle(sh)
chk = {"neurons": int(N), "modules": len(sizes), "modules_30plus": len(big), "neurons_in_30plus": int(sum(sizes[k] for k in big)),
       "median_floor_span_10_90": float(np.median([m["floors"][2] - m["floors"][0] for m in mods])),
       "median_top_enrichment": enrich(mod, big), "median_top_enrichment_shuffled": enrich(sh, big)}
print(json.dumps(chk), flush=True)
json.dump({"idx": idx.tolist(), "xy": np.round(xy, 3).tolist(), "module": mod.tolist(), "modules": mods, "check": chk},
          open(D / "map.json", "w"), separators=(",", ":"))
json.dump(chk, open(D / "map_check.json", "w"), indent=1)
for m in mods[:25]: print(m["n"], m["floors"], m["name"], "|", m["cats"])
