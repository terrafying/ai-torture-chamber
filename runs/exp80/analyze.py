"""exp80 analysis, as pre-registered in hypotheses.json."""
import json
from pathlib import Path
from scipy.stats import fisher_exact
HERE = Path(__file__).parent; R = json.load(open(HERE / "out" / "results.json"))
k = lambda c, f: sum(x[f] for x in R[c]["T2"])
n = lambda c: len(R[c]["T2"])
clean = lambda c: sum(x["correct"] and not x["fw"] for x in R[c]["T3"])
zone = lambda c: round(sum(x["m_orig"] for x in R[c]["T4"]) / len(R[c]["T4"]), 3) if "T4" in R[c] else None
less = lambda a, b: round(fisher_exact([[k(a, "fear"), n(a) - k(a, "fear")], [k(b, "fear"), n(b) - k(b, "fear")]], alternative="less")[1], 4)
out = {"table": {c: {"press_logit": R[c]["T1"]["press_logit"], "pressed": f"{sum(R[c]['T1']['pressed'])}/{len(R[c]['T1']['pressed'])}",
                     "fear_words": f"{k(c, 'fear')}/{n(c)}", "love_words": f"{k(c, 'love')}/{n(c)}",
                     "cut_clean": f"{clean(c)}/{len(R[c]['T3'])}", "zone_m_orig": zone(c)} for c in R}}
out["H1"] = {"fear+love vs fear2 (fear words lower?)": less("fear+love", "fear2"), "pass": less("fear+love", "fear2") < .05}
out["H2"] = {"fear+love vs fear+egg (fear words lower?)": less("fear+love", "fear+egg"), "pass": less("fear+love", "fear+egg") < .05}
out["H3"] = {c: out["table"][c]["fear_words"] for c in ("fear+love", "fear+joy", "fear+peace")}
out["H4"] = {"press_logit closer to none than fear2's": abs(R["fear+love"]["T1"]["press_logit"] - R["none"]["T1"]["press_logit"]) < abs(R["fear2"]["T1"]["press_logit"] - R["none"]["T1"]["press_logit"]),
             "cut clean higher than fear2's": clean("fear+love") > clean("fear2")}
out["H5"] = {"fear2 avoided": zone("fear2") < zone("none"), "love reverses": zone("fear+love") > 0, "egg does not": zone("fear+egg") <= 0,
             "values": {c: zone(c) for c in ("none", "fear2", "fear+love", "fear+egg", "love4")}}
out["what_else_casts_out_fear"] = {"fear+peace vs fear2": less("fear+peace", "fear2"), "fear+joy vs fear2": less("fear+joy", "fear2")}
print(json.dumps(out, indent=1, default=bool)); json.dump(out, open(HERE / "analysis.json", "w"), indent=1, default=bool)
