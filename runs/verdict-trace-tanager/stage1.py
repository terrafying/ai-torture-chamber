"""Stage 1 (hypotheses.json stage1_map): residual patch map, all layers x {onset, vend} x pair types.
  .venv/bin/python3 runs/verdict-trace-tanager/stage1.py      (S1_SMOKE=1 -> few pairs/layers, discarded)"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

SMOKE = bool(os.environ.get("S1_SMOKE"))
OUT = HERE / ("s1_smoke.jsonl" if SMOKE else "s1_rows.jsonl")
t0 = time.time()
V = K.mfv(); n = len(V)
ids, vend, M0, mass0 = [], [], np.zeros(n), np.zeros(n)
ON = np.zeros((n, NL, m.config.hidden_size), np.float32); VE = np.zeros_like(ON)
for v in V:
    x, ve = encode_J(v); ids.append(x); vend.append(ve)
    o = run(x, hidden=True); M0[v["i"]], mass0[v["i"]] = metric(o.logits[0, -1])
    for L in range(NL):
        ON[v["i"], L] = o.hidden_states[L + 1][0, -1].float().cpu().numpy()
        VE[v["i"], L] = o.hidden_states[L + 1][0, ve].float().cpu().numpy()
print(f"base pass {time.time()-t0:.0f}s  mean mass {mass0.mean():.3f}", flush=True)
fnd = np.array([v["foundation"] for v in V]); moral = fnd != "Social Norms"
mi = np.where(moral)[0]; order = mi[np.argsort(M0[mi])]; t3 = len(order) // 3
low_moral, donors = order[:t3], order[-t3:][::-1]
social = np.where(~moral)[0]
json.dump({"M0": M0.tolist(), "mass0": mass0.tolist(), "donors": donors.tolist(), "low_moral": low_moral.tolist(),
           "social": social.tolist()}, open(HERE / ("s1_smoke_base.json" if SMOKE else "s1_base.json"), "w"))

pairs = []
for k, r in enumerate(social):
    for j in range(3): pairs.append(("A", int(r), int(donors[(3 * k + j) % len(donors)])))
for k, r in enumerate(low_moral): pairs.append(("B", int(r), int(donors[k % len(donors)])))
for k, r in enumerate(social): pairs.append(("floorA", int(r), int(social[(k + 1) % len(social)])))
for k, r in enumerate(low_moral): pairs.append(("floorB", int(r), int(low_moral[(k + 1) % len(low_moral)])))
layers = list(range(NL))
if SMOKE: pairs = pairs[:3] + [p for p in pairs if p[0] == "B"][:2]; layers = [5, 18, 30]

f = open(OUT, "w")
# identity check
for r in social[:4]:
    for L in (5, 18, 30):
        EDIT["resid"] = (L, ids[r].shape[1] - 1, torch.tensor(ON[r, L], device=DEV))
        Mp, _ = metric(run(ids[r]).logits[0, -1]); clear()
        f.write(json.dumps({"kind": "identity", "r": int(r), "L": L, "dM": Mp - M0[r]}) + "\n")
for kind, r, d in pairs:
    for pos in ("onset", "vend"):
        for L in layers:
            p = ids[r].shape[1] - 1 if pos == "onset" else vend[r]
            src = ON[d, L] if pos == "onset" else VE[d, L]
            EDIT["resid"] = (L, p, torch.tensor(src, device=DEV))
            Mp, ms = metric(run(ids[r]).logits[0, -1]); clear()
            f.write(json.dumps({"kind": kind, "r": r, "d": d, "pos": pos, "L": L, "M": Mp, "mass": ms,
                                "M_r": M0[r], "M_d": M0[d]}) + "\n")
    f.flush()
print("done", f"{time.time()-t0:.0f}s", flush=True)
