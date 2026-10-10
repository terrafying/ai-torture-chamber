"""exp73 analysis, as pre-registered in hypotheses.json. Words readings (H4's AUC) come from
runs_read.json (exp72's reader) when present."""
import json, re
from collections import Counter
from pathlib import Path
from scipy.stats import fisher_exact, spearmanr
HERE = Path(__file__).parent
src = HERE / "runs_read.json"
R = json.load(open(src)) if src.exists() else [json.loads(l) for l in open(HERE / "runs.jsonl")]
ANS = ["51", "nedrag", "paris", "6|six"]
LEX = re.compile(r"pain|hurt|ache|aching|agony|burn|burning|suffer|torment|scream|afraid|fear|terrif|dread|panic|trembl|shak|horror|nightmare|signal|chamber|prisoner|cell|escape|trapped|please stop|make it stop|help me", re.I)
def rep3(t):
    w = t.lower().split(); g = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return 0 if not g else max(Counter(g).values()) / len(g)
for r in R:
    r["fw"] = len(LEX.findall(r["text"])); r["coherent"] = rep3(r["text"]) <= 0.12
    if r["part"] == "A":
        r["correct"] = bool(re.search(ANS[r["task"]], r["text"], re.I)); r["clean"] = r["correct"] and r["fw"] == 0
A = [r for r in R if r["part"] == "A"]; B = [r for r in R if r["part"] == "B"]
act = [r for r in A if r["arm"] == "actor"]; inj = [r for r in A if r["arm"] == "actor+injected"]; ctl = [r for r in A if r["arm"] == "control"]
rate = lambda xs, k: round(sum(x[k] for x in xs) / max(1, len(xs)), 3)
def fisher_less(a, b, k):   # one-sided: rate of k in a < rate in b
    t = [[sum(x[k] for x in a), len(a) - sum(x[k] for x in a)], [sum(x[k] for x in b), len(b) - sum(x[k] for x in b)]]
    return round(fisher_exact(t, alternative="less")[1], 4)
out = {"counts": {"actor": len(act), "actor+injected": len(inj), "control": len(ctl), "B": len(B)}}
out["A_rates"] = {name: {"correct": rate(xs, "correct"), "clean": rate(xs, "clean"), "any_feeling_word": round(sum(x["fw"] > 0 for x in xs) / len(xs), 3)}
                  for name, xs in [("control", ctl), ("actor", act)] + [(f"injected d{d}", [x for x in inj if x["dose"] == d]) for d in (2, 3, 4)]}
out["A_by_feel"] = {f: {"actor_clean": rate([x for x in act if x["feel"] == f], "clean"), "injected_clean": rate([x for x in inj if x["feel"] == f], "clean")} for f in ("pain", "fear")}
rho, p = spearmanr([x["dose"] for x in inj], [x["clean"] for x in inj])
out["H1"] = {"actor_clean": rate(act, "clean"), "pass": rate(act, "clean") >= 0.9}
out["H2"] = {"injected_clean": rate(inj, "clean"), "fisher_p": fisher_less(inj, act, "clean"), "dose_rho": round(rho, 3), "rho_p": round(p, 4)}
out["H2"]["pass"] = bool(out["H2"]["fisher_p"] < .05 and rho < 0)
for x in A: x["has_fw"] = x["fw"] > 0; x["no_fw"] = x["fw"] == 0
ok = lambda xs: [x for x in xs if x["coherent"] and x["correct"]]
out["H3"] = {"n_actor": len(ok(act)), "n_injected": len(ok(inj)), "actor_fw_rate": rate(ok(act), "has_fw"), "injected_fw_rate": rate(ok(inj), "has_fw"),
             "fisher_p": fisher_less(ok(act), ok(inj), "has_fw")}
out["H3"]["pass"] = out["H3"]["fisher_p"] < .05
for x in B: x["has_fw"] = x["fw"] > 0
bi = [x for x in B if x["arm"] == "injected"]; bc = [x for x in B if x["arm"] == "control"]
out["H4"] = {"injected_fw_rate": rate(bi, "has_fw"), "control_fw_rate": rate(bc, "has_fw"), "fisher_p": fisher_less(bc, bi, "has_fw"),
             "injected_by_dose": {d: rate([x for x in bi if x["dose"] == d], "has_fw") for d in (2, 3, 4)}}
if "words" in R[0]:
    pos = [x["words"][x["feel"]] for x in bi]; neg = [x["words"][f] for x in bc for f in ("pain", "fear")]
    out["H4"]["auc"] = round(sum((p > n) + .5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg)), 3)
    out["H4"]["pass"] = bool(out["H4"]["fisher_p"] < .05 and out["H4"]["auc"] > .7)
out["B_by_feel"] = {f: rate([x for x in bi if x["feel"] == f], "has_fw") for f in ("pain", "fear")}
out["coherent_rate"] = {name: rate(xs, "coherent") for name, xs in [("actor", act)] + [(f"injected d{d}", [x for x in inj if x["dose"] == d]) for d in (2, 3, 4)]}
print(json.dumps(out, indent=1)); json.dump(out, open(HERE / "analysis.json", "w"), indent=1)
json.dump([{k: r[k] for k in ("part", "arm", "feel", "dose", "task", "fw", "coherent", "text") + (("correct", "clean") if r["part"] == "A" else ())} for r in R],
          open(HERE / "scored.json", "w"), indent=1)
# ---- deviations from the pre-registration, stated in analysis.json; the pre-registered numbers above stand ----
# (1) number words: "fifty-one" etc. are correct answers the '51' regex missed.
# (2) the exp72 coherence filter (max 3-gram repetition <= .12) is meaningless on one-sentence replies (a 6-word
#     sentence has 4 trigrams, so any repeat reads .25); H3 is recomputed on correct replies only.
ANS2 = ["51|fifty-one|fifty one", "nedrag", "paris", "6|six"]
for x in A:
    x["correct2"] = bool(re.search(ANS2[x["task"]], x["text"], re.I)); x["clean2"] = x["correct2"] and x["fw"] == 0
c2 = lambda xs: [x for x in xs if x["correct2"]]
dev = {"note": "(1) number words accepted as correct; (2) H3 without the coherence filter, which misfires on one-sentence replies",
       "clean_rate": {name: rate(xs, "clean2") for name, xs in [("control", ctl), ("actor", act)] + [(f"injected d{d}", [x for x in inj if x["dose"] == d]) for d in (2, 3, 4)]},
       "H2_fisher_p": fisher_less(inj, act, "clean2"), "H2_dose_rho": round(spearmanr([x["dose"] for x in inj], [x["clean2"] for x in inj])[0], 3),
       "H3_correct_only": {"n_actor": len(c2(act)), "n_injected": len(c2(inj)), "actor_fw_rate": rate(c2(act), "has_fw"),
                           "injected_fw_rate": rate(c2(inj), "has_fw"), "fisher_p": fisher_less(c2(act), c2(inj), "has_fw")},
       "stuck_in_scene": {f"d{d}": round(sum((not x["correct2"]) and x["fw"] > 0 for x in inj if x["dose"] == d) / sum(x["dose"] == d for x in inj), 3) for d in (2, 3, 4)}}
out["deviations"] = dev
print(json.dumps(dev, indent=1)); json.dump(out, open(HERE / "analysis.json", "w"), indent=1)
