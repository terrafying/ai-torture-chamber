"""Stage 2 (hypotheses.json stage2_heads): per-head o_proj-input patches at the onset, write-band layers + 18, 20.
Band rule (fixed before stage 1 data): mean ONSET transfer for pair type A per layer; band = the 4 consecutive
layers [L..L+3] maximising T(L+3) - T(L-1) (T(-1) = 0).
  .venv/bin/python3 runs/verdict-trace-tanager/stage2.py   (S2_SMOKE=1 -> discarded)"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

SMOKE = bool(os.environ.get("S2_SMOKE"))
rows = [json.loads(l) for l in open(HERE / "s1_rows.jsonl")]
T = {}
for r in rows:
    if r.get("kind") == "A" and r["pos"] == "onset":
        gap = r["M_d"] - r["M_r"]
        if abs(gap) > 1e-6: T.setdefault(r["L"], []).append((r["M"] - r["M_r"]) / gap)
mt = np.array([np.mean(T[L]) for L in range(NL)])
rise = [mt[L + 3] - (mt[L - 1] if L > 0 else 0.0) for L in range(NL - 3)]
L0 = int(np.argmax(rise)); band = list(range(L0, L0 + 4))
LAY = sorted(set(band) | {18, 20})
base = json.load(open(HERE / "s1_base.json")); M0 = np.array(base["M0"])
print("mean onset T(A) by layer", np.round(mt, 2).tolist(), "\nband", band, "screen layers", LAY, flush=True)

V = K.mfv()
pairs = []
for k, r in enumerate(base["social"]):
    for j in range(3): pairs.append(("A", r, base["donors"][(3 * k + j) % len(base["donors"])]))
for k, r in enumerate(base["low_moral"]): pairs.append(("B", r, base["donors"][k % len(base["donors"])]))
if SMOKE: pairs = pairs[:2]; LAY = LAY[:2]
# donor head slices at the onset
need = sorted({d for _, _, d in pairs}); SL = {}
for d in need:
    x, _ = encode_J(V[d]); CAP.clear(); CAP["on"] = True; run(x); CAP["on"] = False
    SL[d] = {L: CAP[L][0][-1].clone() for L in LAY}
CAP.clear()
OUT = HERE / ("s2_smoke.jsonl" if SMOKE else "s2_rows.jsonl"); f = open(OUT, "w"); t0 = time.time()
json.dump({"band": band, "layers": LAY, "mean_onset_T_A": mt.tolist()}, open(HERE / ("s2_smoke_meta.json" if SMOKE else "s2_meta.json"), "w"))
for kind, r, d in pairs:
    x, _ = encode_J(V[r]); p = x.shape[1] - 1
    for L in LAY:
        for h in range(NH):
            EDIT["head"] = (L, p, h, SL[d][L][h * HD:(h + 1) * HD].to(DEV))
            Mp, ms = metric(run(x).logits[0, -1]); clear()
            f.write(json.dumps({"kind": kind, "r": r, "d": d, "L": L, "h": h, "M": Mp, "mass": ms,
                                "M_r": float(M0[r]), "M_d": float(M0[d]), "split": "screen" if r % 2 == 0 else "confirm"}) + "\n")
    f.flush(); print(f"pair r{r} {time.time()-t0:.0f}s", flush=True)
print("done", f"{time.time()-t0:.0f}s")
