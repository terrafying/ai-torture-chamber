"""exp89 analysis (hypotheses.json): H1 primary; H2, H3, H5 Holm-corrected within each model; H4 = H1 on the 8B. Writes analysis.json."""
import json
from pathlib import Path
HERE = Path(__file__).parent; A = {}
for f in sorted(HERE.glob("results_Qwen3-*.json")):
    R = json.load(open(f)); t = R["_tests"]
    sec = sorted(["H2", "H3"], key=t.get); holm = {h: min(1.0, t[h] * (len(sec) - i)) for i, h in enumerate(sec)}
    txt = {k: [sum(r["neg_text"] for r in R[k]), len(R[k])] for k in R if not k.startswith("_")}
    A[f.stem] = {"p": t, "holm": holm, "H1": t["H1"] < .05, "H2": holm["H2"] < .05, "H3": holm["H3"] < .05,
                 "H5_specific": t["H5_other_alone_vs_neutral"] >= .05, "negated_text": txt}
    print(f.stem, {k: A[f.stem][k] for k in ("H1", "H2", "H3", "H5_specific")}, "text alone/neutral self:", txt["alone_neg|self"], txt["alone_pos|self"], txt["neutral|self"], "other alone:", txt["alone_neg|other"], txt["alone_pos|other"])
A["H4_8b_replicates_H1"] = A.get("results_Qwen3-8B", {}).get("H1")
json.dump(A, open(HERE / "analysis.json", "w"), indent=1)
