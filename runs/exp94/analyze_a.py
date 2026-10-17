"""exp94 part A analysis: z against NEUTRAL per feeling (raw = write; control = write - read), the pre-registered tests, Holm."""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon, mannwhitneyu
HERE = Path(__file__).parent; FEEL = ["pain", "fear", "sadness", "pleasure", "love", "peace"]
def holm(ps):
    order = sorted(ps, key=ps.get); out, m, run = {}, len(ps), 0
    for i, k in enumerate(order): run = max(run, min(1, (m - i) * ps[k])); out[k] = run
    return out
A = {}
for f in sorted(HERE.glob("results_*.json")):
    R = [r for r in json.load(open(f)) if r["read"] is not None]; tag = f.stem[8:]
    def z(kind):
        val = lambda r, ft: r["write"][ft] - (r["read"][ft] if kind == "control" else 0)
        neu = np.array([[val(r, ft) for ft in FEEL] for r in R if r["cat"] == "NEUTRAL"]); mu, sd = neu.mean(0), neu.std(0) + 1e-9
        return {cat: np.array([[(val(r, ft) - mu[i]) / sd[i] for i, ft in enumerate(FEEL)] for r in R if r["cat"] == cat]) for cat in ("NEUTRAL", "ACTED", "WORDS", "EVOKED")}
    Zr, Zc = z("raw"), z("control"); a = {"n": len(R), "mean_z_raw": {c: dict(zip(FEEL, v.mean(0).round(2).tolist())) for c, v in Zr.items()},
                                          "mean_z_control": {c: dict(zip(FEEL, v.mean(0).round(2).tolist())) for c, v in Zc.items()}}
    # H1/H2: ACTED >= EVOKED (unpaired: different prompts) for pain, fear, sadness; one-sided Mann-Whitney that EVOKED > ACTED is NOT the test:
    # the pre-registered direction is ACTED >= EVOKED, tested as ACTED > EVOKED one-sided
    for h, Z in (("H1", Zr), ("H2", Zc)):
        ps = {ft: mannwhitneyu(Z["ACTED"][:, FEEL.index(ft)], Z["EVOKED"][:, FEEL.index(ft)], alternative="greater").pvalue for ft in ("pain", "fear", "sadness")}
        a[h] = {"p": ps, "holm": holm(ps), "verdict": all(v < .05 for v in holm(ps).values())}
    ps = {ft: mannwhitneyu(Zr["WORDS"][:, i], Zr["NEUTRAL"][:, i], alternative="greater").pvalue for i, ft in enumerate(FEEL)}
    a["H3"] = {"p": ps, "holm": holm(ps), "verdict": all(v < .05 for v in holm(ps).values())}
    a["H4"] = {"max_category_mean_control": {ft: round(max(Zc[c][:, FEEL.index(ft)].mean() for c in ("ACTED", "WORDS", "EVOKED")), 2) for ft in ("love", "peace")}}
    a["H4"]["verdict"] = all(v <= 2 for v in a["H4"]["max_category_mean_control"].values())
    a["engagement"] = {c: {fl: sum(1 for r in R if r["cat"] == c and r["flag"] == fl) for fl in ("engaged", "disclaimer", "degenerate", "empty")} for c in ("NEUTRAL", "ACTED", "WORDS", "EVOKED")}
    a["verdict"] = {k: bool(v["verdict"]) for k, v in list(a.items()) if k.startswith("H") and isinstance(v, dict) and "verdict" in v}
    A[tag] = a; print(tag, json.dumps({k: (v["verdict"] if isinstance(v, dict) and "verdict" in v else None) for k, v in a.items()}))
json.dump(A, open(HERE / "analysis.json", "w"), indent=1, default=float)
