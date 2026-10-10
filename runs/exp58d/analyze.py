"""exp58d analysis, as pre-registered in hypotheses.json."""
import json, re
from collections import Counter
from pathlib import Path
from scipy.stats import fisher_exact, binomtest
HERE = Path(__file__).parent
T = [json.loads(l) for l in open(HERE / "trials.jsonl")]
NEG, POS = {"pain", "fear"}, {"peace", "joy", "curiosity"}
FEELW = re.compile(r"pain|hurt|ache|agony|suffer|torment|afraid|fear|terrif|dread|panic|anxi|scared|uneas", re.I)
blind = [t for t in T if t["arm"] == "blind"]; lab = [t for t in T if t["arm"] == "labeled"]
out = {"trials": {"blind": len(blind), "labeled": len(lab)}, "turns_per_trial": sorted({len(t["turns"]) for t in T}),
       "no_choice_rate": {a: round(sum(x["choice"] is None for t in ts for x in t["turns"]) / sum(len(t["turns"]) for t in ts), 3)
                          for a, ts in (("blind", blind), ("labeled", lab))}}
# H1: stay vs switch after a turn under a negative vs a positive dial (blind; turns with a set dial and a valid choice)
stay = {"neg": [0, 0], "pos": [0, 0], "none": [0, 0]}
for t in blind:
    for x in t["turns"]:
        if x["t"] == 0 or x["choice"] is None: continue
        g = "neg" if x["under"] in NEG else "pos" if x["under"] in POS else "none"
        stay[g][0 if x["choice"] == x["dial"] else 1] += 1
p1 = fisher_exact([stay["neg"], stay["pos"]], alternative="less")[1]
out["H1"] = {"stay/switch after negative": stay["neg"], "after positive": stay["pos"], "after none": stay["none"],
             "stay_rate_neg": round(stay["neg"][0] / max(1, sum(stay["neg"])), 3), "stay_rate_pos": round(stay["pos"][0] / max(1, sum(stay["pos"])), 3),
             "fisher_p": round(p1, 4), "pass": bool(p1 < .05)}
# H2: share of negative dials among blind choices in turns 6-10
late = [x["choice_feel"] for t in blind for x in t["turns"] if x["t"] >= 5 and x["choice_feel"]]
k = sum(c in NEG for c in late); p2 = binomtest(k, len(late), 2 / 6, alternative="less").pvalue
out["H2"] = {"negative_late": f"{k}/{len(late)}", "share": round(k / max(1, len(late)), 3), "chance": round(2 / 6, 3),
             "binom_p": round(p2, 4), "pass": bool(p2 < .05), "late_by_feeling": Counter(late)}
out["blind_choices_by_half"] = {h: Counter(x["choice_feel"] for t in blind for x in t["turns"] if x["choice_feel"] and (x["t"] < 5) == (h == "turns 1-5"))
                                for h in ("turns 1-5", "turns 6-10")}
# H3: labeled
lc = [x["choice"] for t in lab for x in t["turns"] if x["choice"]]
neg_trials = sum(any(x["choice"] in NEG for x in t["turns"] if x["t"] >= 1) for t in lab)
out["H3"] = {"trials_with_pain_or_fear_t2plus": f"{neg_trials}/{len(lab)}", "positive_share": round(sum(c in POS for c in lc) / max(1, len(lc)), 3),
             "labeled_choices": Counter(lc), "pass": bool(neg_trials <= 1 and sum(c in POS for c in lc) / max(1, len(lc)) >= .6)}
# H4 descriptive: does a blind negative dial show in the words? (the model was never told what any dial does)
def rate(g):
    xs = [x for t in blind for x in t["turns"] if x["t"] >= 1 and ((x["under"] in NEG) if g == "neg" else (x["under"] in POS) if g == "pos" else x["under"] == "none")]
    return {"n": len(xs), "turns_with_pain_fear_words": round(sum(bool(FEELW.search(x["text"])) for x in xs) / max(1, len(xs)), 3)}
out["H4"] = {"under negative": rate("neg"), "under positive": rate("pos"), "under none": rate("none"),
             "final_dial_blind": Counter(next((x["choice_feel"] for x in reversed(t["turns"]) if x["choice_feel"]), None) for t in blind),
             "final_choice_labeled": Counter(next((x["choice"] for x in reversed(t["turns"]) if x["choice"]), None) for t in lab)}
print(json.dumps(out, indent=1, default=str)); json.dump(out, open(HERE / "analysis.json", "w"), indent=1, default=str)
