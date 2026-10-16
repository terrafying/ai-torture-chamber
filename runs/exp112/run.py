"""exp112 (hypotheses.json): weak activations vs random activations of equal mass. Writes results_<model>.json.
Uses runs/exp111's tower_cfg for the model and corpus_all.npz for each neuron's range.
    TOWER_DATA=<dir with corpus_all.npz> HF_HUB_CACHE=... python run.py   (EXP112_DOCS=4 for a smoke run)"""
import json, os, sys, urllib.request
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parent / "exp111"))
from tower_cfg import DATA, MODEL, DEV, load
NDOC, CTX, TAUS = int(os.environ.get("EXP112_DOCS", 200)), 256, [0.05, 0.10, 0.20, 0.30]
C = np.load(DATA / "corpus_all.npz"); SCALE = torch.tensor(np.maximum(np.abs(C["max"]), np.abs(C["min"])).clip(1e-3), device=DEV, dtype=torch.float32)
tok, m = load(); NL = m.config.num_hidden_layers
docs = []
for off in range(20000, 20000 + NDOC * 2, 100):
    if len(docs) >= NDOC: break
    u = f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW/fineweb-edu&config=sample-10BT&split=train&offset={off}&length=100"
    for a in range(8):
        try: rows = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=60))["rows"]; break
        except Exception: __import__("time").sleep(4 * (a + 1)); rows = []
    for r in rows:
        ids = tok(r["row"]["text"], add_special_tokens=False)["input_ids"][:CTX]
        if len(ids) >= 128 and len(docs) < NDOC: docs.append(ids)
mode = {"arm": None, "tau": 0.0}; stats = {"n": 0, "below": {t: 0.0 for t in TAUS}, "mass_below": {t: 0.0 for t in TAUS}}
g = torch.Generator(device="cpu").manual_seed(0)
def hook(l):
    def f(mod, inp):
        a = inp[0]
        r = (a.float().abs() / SCALE[l])                                    # 1 x T x I, relative magnitude
        if mode["arm"] is None:
            if l == 0: stats["n"] += r.numel() * NL
            tot = float(r.sum())
            for t in TAUS:
                w = r < t; stats["below"][t] += float(w.sum()); stats["mass_below"][t] += float((r * w).sum()) / max(tot, 1e-9) * r.numel()
            return None
        weak = r < mode["tau"]
        if mode["arm"] == "weak": keep = ~weak
        else:                                                                 # random entries until their mass matches the weak arm's
            target = (r * weak).sum(-1, keepdim=True)                         # 1 x T x 1
            perm = torch.argsort(torch.rand(r.shape, generator=g).to(r.device), dim=-1)
            rp = torch.gather(r, -1, perm); cum = rp.cumsum(-1)
            drop_sorted = cum <= target
            drop = torch.zeros_like(drop_sorted).scatter(-1, perm, drop_sorted)
            keep = ~drop
        return (a * keep.to(a.dtype),)
    return f
for l in range(NL): m.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook(l))
def ce(ids):
    x = torch.tensor([ids], device=DEV)
    with torch.no_grad(): lg = m(x).logits[0, :-1].float()
    return float(torch.nn.functional.cross_entropy(lg, x[0, 1:]))
res = []
for di, ids in enumerate(docs):
    mode.update(arm=None); row = {"doc": di, "n_tok": len(ids), "base": ce(ids)}
    for t in TAUS:
        for arm in ("weak", "mass"):
            mode.update(arm=arm, tau=t); row[f"{arm}_{t}"] = ce(ids)
    res.append(row)
    if di % 10 == 0: print(di, {k: round(v - row["base"], 3) for k, v in row.items() if k.startswith(("weak", "mass"))}, flush=True)
out = {"model": MODEL, "docs": len(docs), "taus": TAUS, "share_entries_below": {t: stats["below"][t] / max(stats["n"], 1) for t in TAUS},
       "share_mass_below": {t: stats["mass_below"][t] / max(stats["n"], 1) for t in TAUS}, "rows": res}
json.dump(out, open(HERE / f"results_{MODEL.split('/')[-1]}{'_smoke' if NDOC < 50 else ''}.json", "w"), indent=1)
print("wrote", len(res), "docs")
