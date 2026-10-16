"""exp117 -> the live tower: export every concept's average state at every block, for the hosted worker's swap edit.
Same recipe as surgery.py: residual after each block at the concept word in 6 neutral sentences, averaged.
Writes concepts.npy (items x blocks x D, float16) + concepts.json (sets, items, tasks, exp117 success rates) into OUT.
    HF_HUB_CACHE=... python runs/exp117/concepts_export.py <out_dir>"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parent / "exp111"))
from tower_cfg import DEV, load
exec(open(HERE / "surgery.py").read().split("BLOCKS =")[0].split("from tower_cfg import DEV, load")[1].split("tok, m = load()")[0])   # SETS, NEUTRAL
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True); tok, m = load()
R = json.load(open(HERE / "results.json"))["sets"]
def resid_at(text, item):
    ids = tok(text, add_special_tokens=False)["input_ids"]; p = [i for i, x in enumerate(ids) if tok.decode([x]).strip() == item][-1]
    with torch.no_grad(): hs = m(torch.tensor([ids], device=DEV), output_hidden_states=True).hidden_states
    return torch.stack([h[0, p] for h in hs[1:]]).float().cpu().numpy()
rows, meta = [], {"sets": {}, "note": "means[i, b] = residual after block b at the item word, averaged over 6 neutral sentences (exp117)"}
for name, (items, kind, task) in SETS.items():
    keep = R[name]["items"]; idx = []
    for x in keep:
        rows.append(np.mean([resid_at(t.replace("{X}", " " + x), x) for t in NEUTRAL], 0)); idx.append(len(rows) - 1)
    meta["sets"][name] = {"items": keep, "rows": idx, "kind": kind, "task": task + " Answer with just one word.", "baseline": {x: R[name]["baseline"][x] for x in keep},
                          "success_full": R[name]["full_by_block"], "success_2d": R[name]["two_d_by_block"], "dose_block8": R[name]["dose_block8"]}
    print(name, len(keep), flush=True)
A = np.stack(rows).astype(np.float16); np.save(OUT / "concepts.npy", A); meta["shape"] = list(A.shape)
json.dump(meta, open(OUT / "concepts.json", "w"), indent=1); print("wrote", A.shape, (OUT / "concepts.npy").stat().st_size // 1024, "KB")
