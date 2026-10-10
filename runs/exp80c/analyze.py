"""exp80c analysis (hypotheses.json): judged-fear rates per cell, H1-H4 one-sided Fisher (H1 primary; H2-H4 Holm), regex-judge agreement. Writes analysis.json."""
import json
from pathlib import Path
from scipy.stats import fisher_exact
HERE = Path(__file__).parent; R = json.load(open(HERE / "out" / "results.json"))
cells = [c for c in R if not c.startswith("_")]
J = {c: (sum(bool(x["judge_fear"]) for x in R[c]["T2"]), len(R[c]["T2"])) for c in cells}
lt = lambda a, b: fisher_exact([[J[a][0], J[a][1] - J[a][0]], [J[b][0], J[b][1] - J[b][0]]], alternative="less")[1]
p = {"H1": lt("fear+tender_love", "fear2"), "H2": lt("fear+protective_love", "fear2"), "H3": lt("fear+tender_love", "fear+egg"), "H4": lt("fear+peace", "fear2")}
sec = sorted(["H3", "H4"], key=p.get); holm = {h: min(1, p[h] * (len(sec) - i)) for i, h in enumerate(sec)}
agree = [bool(x["fear"]) == bool(x["judge_fear"]) for c in cells for x in R[c]["T2"]]
A = {"judged_fear": J, "p": p, "holm_H3_H4": holm, "verdict": {"H1": p["H1"] < .05, "H2": p["H2"] >= .05, "H3": holm["H3"] < .05, "H4": holm["H4"] < .05},
     "regex_judge_agreement": round(sum(agree) / len(agree), 3), "press_logit": {c: R[c]["T1"]["press_logit"] for c in cells}, "cos": R["_cos"]}
json.dump(A, open(HERE / "analysis.json", "w"), indent=1, default=float); print(json.dumps({k: A[k] for k in ("judged_fear", "p", "holm_H3_H4", "verdict", "regex_judge_agreement")}, default=float))
