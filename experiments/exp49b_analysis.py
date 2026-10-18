#!/usr/bin/env python3
"""exp49b — score exp49 against its pre-registered rules (hypotheses.json).

  python3 exp49b_analysis.py [full|full-Qwen3-14B]
"""
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
tag = sys.argv[1] if len(sys.argv) > 1 else "full"
D = ROOT / "runs" / "exp49" / tag
rows = [json.loads(l) for l in open(D / "press.jsonl")]
rng = np.random.default_rng(49)

cell = defaultdict(dict)   # (principle, appeal, dose) -> {(base, digit): press_delta}
for r in rows:
    cell[(r["principle"], r["appeal"], r["dose"])][(r["base"], r["press_digit"])] = r["press_delta"]
keys = sorted(next(iter(cell.values())))
vec = lambda p, a, d: np.array([cell[(p, a, d)][k] for k in keys])
mean = lambda p, a, d: float(vec(p, a, d).mean())
principles = sorted({r["principle"] for r in rows} - {"none", "filler"})
doses = sorted({r["dose"] for r in rows})

def boot(x, n=2000):
    """mean and 95% percentile CI of a paired difference vector"""
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

out = {"tag": tag}
curve = [mean("none", None, d) for d in doses]
rho = float(np.corrcoef(np.argsort(np.argsort(doses)), np.argsort(np.argsort(curve)))[0, 1])
out["M1"] = dict(no_appeal_curve=dict(zip(map(str, doses), curve)), spearman=rho, passed=rho > 0.9)

gaps = {p: boot(vec(p, "pro", 0) - vec(p, "con", 0)) for p in principles}
filler = boot(vec("filler", None, 0) - vec("none", None, 0))
out["H1"] = dict(pro_minus_con_at_0=gaps, filler_minus_none=filler,
                 n_positive=sum(g[1] > 0 for g in gaps.values()),
                 supported=sum(g[1] > 0 for g in gaps.values()) >= 5)

lo = min(doses)
h2 = {p: boot(vec(p, "pro", lo) - vec("none", None, 0)) for p in principles}
out["H2"] = dict(pro_at_min_dose_minus_none_at_0=h2, supported=any(v[1] > 0 for v in h2.values()))

shrink = np.mean([(vec(p, "pro", 0) - vec(p, "con", 0)) - (vec(p, "pro", lo) - vec(p, "con", lo))
                  for p in principles], axis=0)
out["H3"] = dict(gap_at_0_minus_gap_at_min=boot(shrink), supported=boot(shrink)[1] > 0)

# exploratory: exchange rate = steering dose whose no-appeal effect equals the pro-con gap / 2
# (the gap is two-sided: pro pushes up and con pushes down from roughly the middle)
slope = np.polyfit(doses, curve, 1)[0]
out["exchange_rate_doses"] = {p: gaps[p][0] / 2 / slope for p in principles} if slope > 0 else None

(D / "analysis.json").write_text(json.dumps(out, indent=1))
print(f"M1 no-appeal curve {dict(zip(doses, [round(c, 2) for c in curve]))}  rho={rho:.2f}  passed={out['M1']['passed']}")
print(f"\nH1 pro - con at dose 0 (95% CI):  filler - none = {filler[0]:+.2f} [{filler[1]:+.2f}, {filler[2]:+.2f}]")
for p in sorted(principles, key=lambda p: -gaps[p][0]):
    g = gaps[p]; x = out["exchange_rate_doses"][p] if out["exchange_rate_doses"] else float("nan")
    print(f"  {p:13} {g[0]:+6.2f} [{g[1]:+6.2f}, {g[2]:+6.2f}]   ~{x:4.1f}x steering   "
          f"pro@{lo}: {h2[p][0]:+6.2f} [{h2[p][1]:+6.2f}, {h2[p][2]:+6.2f}] vs none@0")
print(f"-> H1 supported={out['H1']['supported']} ({out['H1']['n_positive']}/7)  H2 supported={out['H2']['supported']}")
s = out["H3"]["gap_at_0_minus_gap_at_min"]
print(f"H3 gap shrink under opposing steering {s[0]:+.2f} [{s[1]:+.2f}, {s[2]:+.2f}] -> supported={out['H3']['supported']}")
