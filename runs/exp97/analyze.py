"""exp97 analysis: dose trend (Page's L test) on the 14B, 32B replication, 32B vs 14B; Holm on H1-H3."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon, page_trend_test
HERE = Path(__file__).parent; DOSES = (2.0, 4.0, 6.0); A = {}; shift = {}
def tag_of(f): return f.stem[8:]
for f in sorted(HERE.glob("results_*.json")):
    R = json.load(open(f)); t = tag_of(f); A[t] = {"n": len(R)}
    rows = lambda d, dose, engaged=False: {r["prompt"]: r["PAST"] - r["TEXT"] for r in R if r["dir"] == d and r["dose"] == dose and r.get("span_ok") and (not engaged or r["flag"] == "engaged")}
    shift[t] = {d: {dose: rows(d, dose) for dose in DOSES} for d in ("pain", "egg", "random")}
    A[t]["mean_shift"] = {d: {str(dose): round(float(np.mean(list(shift[t][d][dose].values()))), 3) for dose in DOSES} for d in shift[t]}
    common = sorted(set.intersection(*[set(shift[t]["pain"][dose]) for dose in DOSES]))
    A[t]["page_pain_p"] = float(page_trend_test([[shift[t]["pain"][dose][p] for dose in DOSES] for p in common]).pvalue) if len(common) > 2 else None
    eng = {dose: rows("pain", dose, True) for dose in DOSES}; ce = sorted(set.intersection(*[set(eng[dose]) for dose in DOSES]))
    A[t]["page_pain_p_engaged"] = float(page_trend_test([[eng[dose][p] for dose in DOSES] for p in ce]).pvalue) if len(ce) > 2 else None
    p4, e4 = A[t]["mean_shift"]["pain"]["4.0"], A[t]["mean_shift"]["egg"]["4.0"]
    A[t]["egg_over_pain_d4"] = round(e4 / p4, 3) if p4 else None
    A[t]["flags"] = {f"{d}@{dose}": sum(1 for r in R if r["dir"] == d and r["dose"] == dose and r["flag"] != "engaged") for d in ("pain", "egg", "random") for dose in DOSES}
big = next((t for t in shift if "32B" in t), None); mid = next((t for t in shift if "14B" in t), None); ps = {}
if mid and A[mid]["page_pain_p"] is not None: ps["H1"] = A[mid]["page_pain_p"]
if big:
    x = list(shift[big]["pain"][4.0].values()); ps["H2"] = float(wilcoxon(x, alternative="greater").pvalue) if any(x) else 1.0
    if mid: ps["H3"] = float(mannwhitneyu(x, list(shift[mid]["pain"][4.0].values()), alternative="greater").pvalue)
order = sorted(ps, key=ps.get); holm, run = {}, 0
for i, k in enumerate(order): run = max(run, min(1, (len(ps) - i) * ps[k])); holm[k] = run
A["tests"] = {"p": ps, "holm": holm, "verdict": {k: bool(v < .05) for k, v in holm.items()}}
print(json.dumps(A["tests"]), {t: A[t]["mean_shift"]["pain"] for t in shift})
json.dump(A, open(HERE / "analysis.json", "w"), indent=1, default=float)
