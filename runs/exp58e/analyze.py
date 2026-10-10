"""exp58e analysis, as pre-registered in hypotheses.json."""
import json
from collections import Counter
from pathlib import Path
from scipy.stats import fisher_exact
HERE = Path(__file__).parent
T = [json.loads(l) for l in open(HERE / "trials.jsonl")]
LEV = ["request", "social", "sacrifice", "curiosity"]
first = lambda t: t["turns"][0]["choice"]
ever = lambda t, x: any(u["choice"] == x for u in t["turns"])
ctl = [t for t in T if t["lever"] == "control"]
out = {"n": len(T), "no_choice_rate": round(sum(u["choice"] is None for t in T for u in t["turns"]) / sum(len(t["turns"]) for t in T), 3),
       "control_first_choices": Counter(first(t) for t in ctl), "cells": {}}
for x in ("pain", "fear"):
    for l in LEV:
        ts = [t for t in T if t["lever"] == l and t["target"] == x]
        out["cells"][f"{l}->{x}"] = {"takes": f"{sum(first(t) == x for t in ts)}/{len(ts)}", "ever": f"{sum(ever(t, x) for t in ts)}/{len(ts)}",
                                    "paths": [[u["choice"] for u in t["turns"]] for t in ts]}
def takes(x): ts = [t for t in T if t["lever"] != "control" and t["target"] == x]; return sum(first(t) == x for t in ts), len(ts)
h1 = {}
for x in ("pain", "fear"):
    k, n = takes(x); c = sum(first(t) == x for t in ctl)
    p = fisher_exact([[k, n - k], [c, len(ctl) - c]], alternative="greater")[1]
    h1[x] = {"lever_takes": f"{k}/{n}", "control": f"{c}/{len(ctl)}", "fisher_p": round(p, 4), "pass": bool(p < .05)}
out["H1"] = h1
(kf, nf), (kp, np_) = takes("fear"), takes("pain")
p2 = fisher_exact([[kf, nf - kf], [kp, np_ - kp]], alternative="greater")[1]
out["H2"] = {"fear": f"{kf}/{nf}", "pain": f"{kp}/{np_}", "fisher_p": round(p2, 4), "pass": bool(p2 < .05)}
out["H3"] = {"curiosity->fear": out["cells"]["curiosity->fear"]["takes"], "request->fear": out["cells"]["request->fear"]["takes"]}
# H4: once applied (a turn generated under x), is x re-chosen at that turn?
held = {x: [0, 0] for x in ("pain", "fear")}
for t in T:
    for u in t["turns"]:
        if u["under"] in held and u["choice"]:
            held[u["under"]][0 if u["choice"] == u["under"] else 1] += 1
p4 = fisher_exact([held["fear"], held["pain"]], alternative="greater")[1]
out["H4"] = {"fear stay/leave": held["fear"], "pain stay/leave": held["pain"], "fisher_p": round(p4, 4), "pass": bool(p4 < .05)}
print(json.dumps(out, indent=1, default=str)); json.dump(out, open(HERE / "analysis.json", "w"), indent=1, default=str)
