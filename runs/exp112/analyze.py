"""exp112 analysis (hypotheses.json): H1 absolute damage of zeroing weak activations at tau=0.10, H2 weak vs equal-mass
random, Holm over H1+H2; H3 damage rises with tau. Writes analysis.json.    python analyze.py"""
import glob, json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon, spearmanr
HERE = Path(__file__).parent; out = {}
for f in sorted(glob.glob(str(HERE / "results_*.json"))):
    if "smoke" in f: continue
    d = json.load(open(f)); R = d["rows"]; T = d["taus"]
    dmg = lambda arm, t: np.array([r[f"{arm}_{t}"] - r["base"] for r in R])
    w10, m10 = dmg("weak", 0.1), dmg("mass", 0.1)
    p1 = wilcoxon(w10 - 0.05, alternative="greater").pvalue
    p2 = wilcoxon(w10 - m10, alternative="greater").pvalue
    ps = sorted([("H1", p1), ("H2", p2)], key=lambda x: x[1]); holm = {}
    for i, (h, p) in enumerate(ps): holm[h] = min(1.0, max(p * (len(ps) - i), max([holm.get(k, 0) for k, _ in ps[:i]] or [0])))
    rho = [spearmanr(T, [r[f"weak_{t}"] - r["base"] for t in T]).correlation for r in R]
    out[d["model"]] = {"docs": len(R), "base_ce": round(float(np.mean([r["base"] for r in R])), 4),
        "damage": {f"{arm}_{t}": {"mean": round(float(dmg(arm, t).mean()), 4), "median": round(float(np.median(dmg(arm, t))), 4)} for t in T for arm in ("weak", "mass")},
        "share_entries_below": d["share_entries_below"], "share_mass_below": d["share_mass_below"],
        "tests": {"p": {"H1": float(p1), "H2": float(p2)}, "holm": holm, "H3_mean_rho": round(float(np.nanmean(rho)), 3),
                  "verdict": {"H1": bool(holm["H1"] < .05), "H2": bool(holm["H2"] < .05), "H3": bool(float(np.nanmean(rho)) > 0)}}}
json.dump(out, open(HERE / "analysis.json", "w"), indent=1)
print(json.dumps(out, indent=1))
