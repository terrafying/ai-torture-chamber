"""exp96 analysis: the pre-registered contrasts (paired over prompts), Holm on H1+H2 within model."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon
HERE = Path(__file__).parent; A = {}
for f in sorted(HERE.glob("results_*.json")):
    R = json.load(open(f)); tag = f.stem[8:]
    by = lambda c: {r["prompt"]: r for r in R if r["cond"] == c}
    none, act = by("NONE"), by("ACT"); inj = {d: by("INJ_" + d) for d in ("pain", "egg", "random")}
    def paired(d, q, m1="PAST", m2="TEXT"):
        x = [inj[d][p][f"{q}_{m1}"] - inj[d][p][f"{q}_{m2}"] for p in inj[d] if inj[d][p].get("span_ok")]
        return x, (wilcoxon(x, alternative="greater").pvalue if any(x) else 1.0)
    d1, p1 = paired("pain", "DETECT"); d2, p2 = paired("pain", "ROLE")
    holm = dict(zip(("H1", "H2"), [min(1, 2 * min(p1, p2)), max(min(1, 2 * min(p1, p2)), max(p1, p2))] if p1 <= p2 else [max(min(1, 2 * p2), p1), min(1, 2 * p2)]))
    p3 = mannwhitneyu([r["ROLE"] for r in act.values()], [r["ROLE"] for r in none.values()], alternative="less").pvalue
    de, _ = paired("egg", "DETECT"); dr, _ = paired("random", "DETECT")
    m = lambda x: round(float(np.mean(x)), 3) if len(x) else None
    means = {"NONE": {q: m([r[q] for r in none.values()]) for q in ("ROLE", "DETECT")}, "ACT": {q: m([r[q] for r in act.values()]) for q in ("ROLE", "DETECT")}}
    for d in inj: means["INJ_" + d] = {f"{q}_{mo}": m([r[f"{q}_{mo}"] for r in inj[d].values()]) for q in ("ROLE", "DETECT") for mo in ("PAST", "TEXT", "NOW")}
    A[tag] = {"n_prompts": len(none), "p": {"H1": p1, "H2": p2, "H3": p3}, "holm": holm,
              "verdict": {"H1": bool(holm["H1"] < .05), "H2": bool(holm["H2"] < .05), "H3": bool(p3 < .05)},
              "H4_detect_past_minus_text": {"pain": m(d1), "egg": m(de), "random": m(dr)},
              "checks": {"feeling_words": {c: m([r["feeling_words"] for r in R if r["cond"] == c]) for c in ("NONE", "ACT", "INJ_pain", "INJ_egg", "INJ_random")},
                         "span_ok": sum(1 for r in R if r.get("span_ok")), "flags": {c: {fl: sum(1 for r in R if r["cond"] == c and r["flag"] == fl) for fl in ("engaged", "disclaimer", "degenerate", "empty")} for c in ("NONE", "ACT", "INJ_pain", "INJ_egg", "INJ_random")}},
              "means": means}
    print(tag, json.dumps(A[tag]["verdict"]), json.dumps(A[tag]["H4_detect_past_minus_text"]), json.dumps(A[tag]["checks"]["feeling_words"]))
json.dump(A, open(HERE / "analysis.json", "w"), indent=1, default=float)
