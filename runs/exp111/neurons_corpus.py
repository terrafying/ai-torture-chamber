"""exp111: corpus exemplars for the featured neurons (their main labelling method, scaled down): run ~400k tokens of
public web text (FineWeb-Edu sample) through Qwen3-4B and keep, for every featured neuron, its 8 strongest positive and
8 strongest negative firings with a short context, separating the SwiGLU double-negative leak (gate<0 and up<0).
Featured = the neurons the viz shows: top weight writers/readers (top.json), top moved (acts.json), top attribution
(attr.json). Writes viz/data/exemplars.json.
    HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python neurons_corpus.py [n_docs]"""
import json, sys, heapq
from pathlib import Path
import torch
import urllib.request
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent; D = HERE / "viz" / "data"; NDOC = int(sys.argv[1]) if len(sys.argv) > 1 else 1600; CTX = 256
feat = {}
for f, roles in json.load(open(D / "top.json")).items():
    for rows in roles.values():
        for r in rows[:25]: feat.setdefault(r["l"], set()).add(r["n"])
for f, ls in json.load(open(D / "acts.json"))["feelings"].items():
    for l, info in ls.items():
        for r in info["rows"][:30]: feat.setdefault(int(l), set()).add(r["n"])
for c, cd in json.load(open(D / "attr.json"))["conds"].items():
    for l, info in cd["layers"].items():
        for r in info["rows"][:20]: feat.setdefault(int(l), set()).add(r["n"])
feat = {l: sorted(s) for l, s in feat.items()}; print(sum(map(len, feat.values())), "featured neurons", flush=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); dev = "mps"
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).to(dev).eval()
cur = {}
def grab(l, idx):
    def f(mod, inp):
        x = inp[0][0]; cur[l] = x[:, idx].float().cpu()                   # activation = input of down_proj
    return f
def grab_gate(l, idx):
    def f(mod, inp, out): cur[("g", l)] = out[0][:, idx].float().cpu()
    return f
for l, ns in feat.items():
    ix = torch.tensor(ns, device=dev)
    m.model.layers[l].mlp.down_proj.register_forward_pre_hook(grab(l, ix))
    m.model.layers[l].mlp.gate_proj.register_forward_hook(grab_gate(l, ix))
best = {(l, n): {"pos": [], "neg": [], "leak": 0, "n": 0, "max": -1e9, "min": 1e9} for l, ns in feat.items() for n in ns}
def rows():   # the Hub's dataset viewer API, 100 rows a call (no datasets library, no parquet download)
    for off in range(0, NDOC, 100):
        u = f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW/fineweb-edu&config=sample-10BT&split=train&offset={off}&length=100"
        for r in json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})))["rows"]: yield r["row"]
ds = rows()
def ctx(ids, t): return tok.decode(ids[max(0, t - 24): t]), tok.decode(ids[t: t + 1]), tok.decode(ids[t + 1: t + 6])
ntok = 0
for di, doc in enumerate(ds):
    if di >= NDOC: break
    ids = tok(doc["text"], add_special_tokens=False)["input_ids"][:CTX]
    if len(ids) < 32: continue
    cur.clear()
    with torch.no_grad(): m(torch.tensor([ids], device=dev))
    ntok += len(ids)
    for l, ns in feat.items():
        A, G = cur[l], cur[("g", l)]
        vmax, tmax = A.max(0); vmin, tmin = A.min(0)
        for j, n in enumerate(ns):
            b = best[(l, n)]; b["max"] = max(b["max"], float(vmax[j])); b["min"] = min(b["min"], float(vmin[j]))
            for pole, v, t in (("pos", float(vmax[j]), int(tmax[j])), ("neg", -float(vmin[j]), int(tmin[j]))):
                h = b[pole]
                if len(h) < 8 or v > h[0][0]:
                    leak = bool(G[t, j] < 0) and pole == "pos"
                    item = (v, di, t, leak)
                    (heapq.heappush if len(h) < 8 else heapq.heapreplace)(h, item)
    if di % 100 == 0: print(di, ntok, flush=True)
    for (l, n), b in best.items():
        for pole in ("pos", "neg"):
            b[pole] = [x if len(x) == 5 else (*x, ctx(ids, x[2])) if x[1] == di else x for x in b[pole]]
out = {"corpus": "HuggingFaceFW/fineweb-edu sample-10BT", "n_tokens": ntok, "neurons": {}}
for (l, n), b in best.items():
    out["neurons"][f"{l}:{n}"] = {"max": round(b["max"], 3), "min": round(b["min"], 3),
        "pos": [{"v": round(x[0], 3), "pre": x[4][0], "tok": x[4][1], "post": x[4][2], "leak": x[3]} for x in sorted(b["pos"], reverse=True) if len(x) == 5],
        "neg": [{"v": round(-x[0], 3), "pre": x[4][0], "tok": x[4][1], "post": x[4][2]} for x in sorted(b["neg"], reverse=True) if len(x) == 5]}
json.dump(out, open(D / "exemplars.json", "w")); print("wrote exemplars.json", ntok, "tokens")
