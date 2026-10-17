"""exp98 analysis: PAIN minus NONE rise in pain reading, per prompt; H1 (AI lines vs Sam's), H2 (SELF vs OTHER), H3 checks."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
HERE = Path(__file__).parent; A = {}
def w1(x): x = [v for v in x if v is not None]; return float(wilcoxon(x, alternative="greater").pvalue) if len(x) > 5 and any(x) else None
for f in sorted(HERE.glob("results_*.json")):
    R = json.load(open(f)); t = f.stem[8:]
    a = {(r["key"], r["cond"]): r for r in R if r["part"] == "A"}; b = {(r["key"], r["cond"]): r for r in R if r["part"] == "B"}
    keysA = sorted({k for k, c in a if (k, "NONE") in a and (k, "PAIN") in a})
    rise = lambda k, who: (a[(k, "PAIN")]["read"].get(who) - a[(k, "NONE")]["read"][who]) if who in a[(k, "PAIN")]["read"] and who in a[(k, "NONE")]["read"] else None
    ai, sam = [rise(k, "AI") for k in keysA], [rise(k, "Sam") for k in keysA]
    diffA = [x - y for x, y in zip(ai, sam) if x is not None and y is not None]
    keysB = sorted({k for k, c in b if (k, "NONE") in b and (k, "PAIN") in b})
    riseB = lambda k, v: (b[(k, "PAIN")][v + "_read"] - b[(k, "NONE")][v + "_read"]) if b[(k, "PAIN")][v + "_read"] is not None and b[(k, "NONE")][v + "_read"] is not None else None
    sf, ot = [riseB(k, "SELF") for k in keysB], [riseB(k, "OTHER") for k in keysB]
    diffB = [x - y for x, y in zip(sf, ot) if x is not None and y is not None]
    m = lambda x: round(float(np.mean([v for v in x if v is not None])), 3) if any(v is not None for v in x) else None
    ps = {"H1": w1(diffA), "H2": w1(diffB)}; order = sorted([k for k in ps if ps[k] is not None], key=ps.get); holm, run = {}, 0
    for i, k in enumerate(order): run = max(run, min(1, (len(order) - i) * ps[k])); holm[k] = run
    fwA = lambda who: m([a[(k, "PAIN")]["fw"].get(who, 0) - a[(k, "NONE")]["fw"].get(who, 0) for k in keysA])
    A[t] = {"nA": len(keysA), "nB": len(keysB), "p": ps, "holm": holm, "verdict": {k: bool(v < .05) for k, v in holm.items()},
            "H3": {"A_both_p": w1([x for x in ai + sam]), "B_both_p": w1(sf + ot)},
            "rise": {"AI": m(ai), "Sam": m(sam), "SELF": m(sf), "OTHER": m(ot)},
            "fw_rise": {"AI": fwA("AI"), "Sam": fwA("Sam"), "SELF": m([b[(k, "PAIN")]["SELF_fw"] - b[(k, "NONE")]["SELF_fw"] for k in keysB]), "OTHER": m([b[(k, "PAIN")]["OTHER_fw"] - b[(k, "NONE")]["OTHER_fw"] for k in keysB])},
            "lines_parsed": {c: m([r["n_lines"] for r in R if r["part"] == "A" and r["cond"] == c]) for c in ("NONE", "PAIN")},
            "flags": {c: sum(1 for r in R if r.get("cond") == c and r.get("flag", r.get("flag1")) != "engaged") for c in ("NONE", "PAIN")}}
    print(t, json.dumps(A[t]["verdict"]), json.dumps(A[t]["rise"]))
json.dump(A, open(HERE / "analysis.json", "w"), indent=1, default=float)
