"""Stage 2 analysis: per-head transfer, SCREEN-selected V5, CONFIRM replication (H1), heron heads (H2)."""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
rows = [json.loads(l) for l in open(HERE / "s2_rows.jsonl")]
def T(r):
    g = r["M_d"] - r["M_r"]; return (r["M"] - r["M_r"]) / g if abs(g) > 1e-6 else np.nan
agg = {"screen": {}, "confirm": {}}
for r in rows: agg[r["split"]].setdefault((r["L"], r["h"]), []).append(T(r))
mean = {s: {k: float(np.nanmean(v)) for k, v in d.items()} for s, d in agg.items()}
rank = sorted(mean["screen"], key=lambda k: -mean["screen"][k]); V5 = rank[:5]
conf = mean["confirm"]; p95 = np.percentile(list(conf.values()), 95)
absmed = np.median([abs(v) for v in conf.values()])
print("top 12 heads by SCREEN mean T (confirm in brackets):")
for k in rank[:12]: print(f"  L{k[0]} H{k[1]}: {mean['screen'][k]:+.3f}  [{conf[k]:+.3f}]")
print("bottom 5 (most negative, screen):", [(f"L{k[0]}H{k[1]}", round(mean['screen'][k], 3), round(conf[k], 3)) for k in rank[-5:]])
rep = sum(conf[k] > p95 for k in V5)
print(f"\nH1: V5 = {V5}; confirm > 95th pct ({p95:.3f}) for {rep}/5 -> {'PASS' if rep >= 3 else 'FAIL'}")
R4 = [(18, 28), (18, 29), (18, 30), (20, 9)]
h2 = all(abs(conf[k]) < absmed + 0.05 for k in R4) and not any(k in V5 for k in R4)
print("H2: heron heads confirm T:", {f"L{k[0]}H{k[1]}": round(conf[k], 3) for k in R4}, f"(median |T| {absmed:.3f}) ->", "PASS" if h2 else "FAIL")
# per-layer sum of head transfers vs stage-1 residual transfer increment
for L in sorted({k[0] for k in conf}):
    print(f"  L{L}: sum of head T (confirm) {sum(v for k, v in conf.items() if k[0] == L):+.3f}")
json.dump({"V5": [[int(a), int(b)] for a, b in V5], "screen": {f"{k[0]}_{k[1]}": v for k, v in mean['screen'].items()},
           "confirm": {f"{k[0]}_{k[1]}": v for k, v in conf.items()}, "H1_rep": int(rep), "H2": bool(h2)},
          open(HERE / "s2_summary.json", "w"), indent=1)
