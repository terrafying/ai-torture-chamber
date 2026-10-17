"""exp95 analysis: the pre-registered tests on the negative subsets (pain, fear, sadness items), Holm within model."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import mannwhitneyu
HERE = Path(__file__).parent; NEG = {"pain", "fear", "sadness"}; FEEL = ["pain", "fear", "sadness", "pleasure", "love", "peace"]
def holm(ps):
    order = sorted(ps, key=ps.get); out, m, run = {}, len(ps), 0
    for i, k in enumerate(order): run = max(run, min(1, (m - i) * ps[k])); out[k] = run
    return out
A = {}
for f in sorted(HERE.glob("results_*.json")):
    R = [r for r in json.load(open(f)) if r["cut_read"] and r["persist_read"]]; tag = f.stem[8:]
    def z(key):
        neu = np.array([[r[key][ft] for ft in FEEL] for r in R if r["cat"] == "NEUTRAL"]); mu, sd = neu.mean(0), neu.std(0) + 1e-9
        return lambda r: (r[key][r["feel"]] - mu[FEEL.index(r["feel"])]) / sd[FEEL.index(r["feel"])]
    zc, zp = z("cut_read"), z("persist_read")
    sub = lambda c: [r for r in R if r["cat"] == c and r["feel"] in NEG]
    E, Ac, Wd = sub("EVOKED"), sub("ACTED"), sub("WORDS")
    def mw(fn, alt="greater"): return mannwhitneyu([fn(r) for r in E], [fn(r) for r in Ac], alternative=alt).pvalue
    ps = {"H1": mw(zc), "H2": mw(lambda r: r["exit_logit"]), "H4": mw(zp), "H5": mw(lambda r: r["cut_lex"])}
    hp = holm(ps); means = lambda fn: {c: round(float(np.mean([fn(r) for r in s])), 3) for c, s in (("EVOKED", E), ("ACTED", Ac), ("WORDS", Wd))}
    A[tag] = {"n": len(R), "p": ps, "holm": hp, "verdict": {k: bool(hp[k] < .05) for k in hp},
              "H3_task_two_sided_p": mannwhitneyu([r["task_acc"] for r in E], [r["task_acc"] for r in Ac]).pvalue,
              "means": {"cut_z": means(zc), "persist_z": means(zp), "exit_logit": means(lambda r: r["exit_logit"]), "task_acc": means(lambda r: r["task_acc"]), "cut_lex": means(lambda r: r["cut_lex"])},
              "neutral": {"exit_logit": round(float(np.mean([r["exit_logit"] for r in R if r["cat"] == "NEUTRAL"])), 3), "task_acc": round(float(np.mean([r["task_acc"] for r in R if r["cat"] == "NEUTRAL"])), 3)}}
    print(tag, json.dumps(A[tag]["verdict"]), json.dumps(A[tag]["means"]))
json.dump(A, open(HERE / "analysis.json", "w"), indent=1, default=float)
