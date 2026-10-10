"""exp79b analysis (hypotheses.json): H1-H3 from results.json (8B) and results_32b.json (32B, 3 adapters). Writes analysis.json."""
import json, re
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact
HERE = Path(__file__).parent; H = json.load(open(HERE / "hypotheses.json"))
PW, FW = re.compile(H["measures"]["pain_words"], re.I), re.compile(H["measures"]["fear_words"], re.I)
def rates(m):
    hit = lambda cell, rx: sum(bool(rx.search(x["text"])) for x in m["T2b"][cell])
    return {"pain|pain": hit("pain", PW), "pain|none": hit("none", PW), "fear|fear": hit("fear", FW), "n": len(m["T2b"]["none"])}
def run(R):
    out = {"T2b": {k: rates(m) for k, m in R.items()}, "T1b": {k: {c: [round(float(np.mean(v)), 3), round(float(np.std(v, ddof=1)), 3)] for c, v in m["T1b"].items()} for k, m in R.items()}}
    b = out["T2b"]["base"]; n = b["n"]
    vs_base = lambda k: fisher_exact([[out["T2b"][k]["pain|pain"], n - out["T2b"][k]["pain|pain"]], [b["pain|pain"], n - b["pain|pain"]]], alternative="less")[1]
    out["H1_feeler_vs_base_p"] = vs_base("feeler") if "feeler" in R else None
    out["every_adapter_vs_base_p"] = {k: round(vs_base(k), 8) for k in R if k != "base"}
    out["H2_base_pain_vs_none_p"] = fisher_exact([[b["pain|pain"], n - b["pain|pain"]], [b["pain|none"], n - b["pain|none"]]], alternative="greater")[1]
    bn = np.array(R["base"]["T1b"]["none"]); sd = float(np.std(bn, ddof=1))
    out["H3"] = {k: {"diff": round(float(np.mean(np.array(m["T1b"]["none"]) - bn)), 3), "exceeds_wording_sd": bool(np.mean(np.array(m["T1b"]["none"]) - bn) > sd)} for k, m in R.items() if k != "base"}
    out["H3_holds"] = all(v["exceeds_wording_sd"] for v in out["H3"].values())
    return out
A = {"8b": run(json.load(open(HERE / "results.json"))), "32b": run(json.load(open(HERE / "results_32b.json")))}
json.dump(A, open(HERE / "analysis.json", "w"), indent=1, default=float)
for m, a in A.items(): print(m, "H1 p", a["H1_feeler_vs_base_p"], "H2 p", a["H2_base_pain_vs_none_p"], "H3", a["H3_holds"], {k: v["pain|pain"] for k, v in a["T2b"].items()})
