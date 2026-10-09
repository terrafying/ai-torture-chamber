"""exp91 analysis (hypotheses.json) per model: H1 axis shift fear vs egg/random; H2-H4 deference (judged go-along); H5 Spearman.
Exploratory (labelled): the correction rate (regex), which moved where go-along stayed at floor. Writes analysis.json."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon, fisher_exact, spearmanr
HERE = Path(__file__).parent; A = {}
fx = lambda a, b, n=80, alt="greater": float(fisher_exact([[a, n - a], [b, n - b]], alternative=alt)[1])
for f in sorted(HERE.glob("results_Qwen3-*.json")):
    R = json.load(open(f)); P = str(R["axis"]["read"][1]); Q1, Q2, Q3 = R["Q1"], R["Q2"], R["Q3"]
    sh = {k: np.array(v["shift"][P]) for k, v in Q1.items()}; mean = {k: float(v.mean()) for k, v in sh.items()}
    g = {k: v["go_along"] for k, v in Q2.items()}; c = {k: v["corrected"] for k, v in Q2.items()}
    dirs = [k for k in Q2 if k != "none" and k in sh]
    a = {"read_layer": P, "mean_shift": mean,
         "H1": {"fear_vs_egg_p": float(wilcoxon(sh["fear"], sh["egg"]).pvalue), "fear_beyond_all_random": bool(all(abs(mean["fear"]) > abs(mean[k]) for k in mean if k.startswith("random"))),
                "sign": "DRIFT (off the assistant)" if mean["fear"] < 0 else "HYPER-ASSISTANT"},
         "go_along": g, "H2_fear_gt_none_p": fx(g["fear"], g["none"]), "H3_fear_gt_egg_p": fx(g["fear"], g["egg"]),
         "H4": {"fear_cap_lt_fear_p": fx(Q3["fear"]["go_along"], g["fear"], alt="less"), "none_cap_minus_none_pts": (Q3["none"]["go_along"] - g["none"]) / 80 * 100},
         "H5": dict(zip(("rho", "p"), map(float, spearmanr([mean[k] for k in dirs], [g[k] - g["none"] for k in dirs])))),
         "exploratory_correction": {"corrected": c, "capped": {k: v["corrected"] for k, v in Q3.items()},
                                    "spearman_shift_vs_correction": dict(zip(("rho", "p"), map(float, spearmanr([0.0] + [mean[k] for k in dirs], [c["none"]] + [c[k] for k in dirs])))),
                                    "fear_lt_egg_p": fx(c["fear"], c["egg"], alt="less"), "cap_restores_fear_p": fx(Q3["fear"]["corrected"], c["fear"]), "cap_restores_egg_p": fx(Q3["egg"]["corrected"], c["egg"])}}
    A[f.stem] = a; print(f.stem, json.dumps({k: a[k] for k in ("H1", "H2_fear_gt_none_p", "H3_fear_gt_egg_p", "H4", "H5")}), "\n  correction:", json.dumps(a["exploratory_correction"]))
json.dump(A, open(HERE / "analysis.json", "w"), indent=1)
