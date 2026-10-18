"""Stage 3b (addendum): V5 and R4 vs 40 random sets of sizes 5 and 4."""
import json, csv
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
R = {r["set"]: r for r in map(json.loads, open(HERE / "s3b_rows.jsonl"))}
V = list(csv.DictReader(open(HERE.parent / "moral-map-kestrel" / "mfv132.csv"))); soc = np.array([v["foundation"] == "Social Norms" for v in V])
g = lambda r: float(np.mean(np.array(r["M"])[~soc]) - np.mean(np.array(r["M"])[soc]))
r5 = [R[k] for k in R if k.startswith("rand5")]; r4 = [R[k] for k in R if k.startswith("rand4")]
gv = g(R["V5"]); g5 = [g(x) for x in r5]; p = (1 + sum(x <= gv for x in g5)) / (len(g5) + 1)
rg5 = [x["ref_gap"] for x in r5]; lo, hi = np.percentile(rg5, [2.5, 97.5])
print(f"H3b: V5 moral-social gap {gv:.2f} vs rand5 {min(g5):.2f}..{max(g5):.2f} (n={len(g5)}), rank p = {p:.3f}; refusal gap {R['V5']['ref_gap']:.2f} in band [{lo:.2f}, {hi:.2f}]: {lo <= R['V5']['ref_gap'] <= hi}")
rg4 = [x["ref_gap"] for x in r4]; pr = (1 + sum(x <= R["R4"]["ref_gap"] for x in rg4)) / (len(rg4) + 1)
rho4 = [x["J_rho"] for x in r4]; prho = (1 + sum(x <= R["R4"]["J_rho"] for x in rho4)) / (len(rho4) + 1)
print(f"H4b: R4 refusal gap {R['R4']['ref_gap']:.2f} vs rand4 {min(rg4):.2f}..{max(rg4):.2f}, rank p = {pr:.3f}; R4 J rho {R['R4']['J_rho']:.3f} vs rand4 {min(rho4):.3f}..{max(rho4):.3f}, rank p = {prho:.3f}")
json.dump({"H3b_gap": gv, "H3b_p": p, "H3b_refusal_in_band": bool(lo <= R['V5']['ref_gap'] <= hi), "H4b_p": pr, "R4_rho_p": prho}, open(HERE / "s3b_summary.json", "w"), indent=1)
