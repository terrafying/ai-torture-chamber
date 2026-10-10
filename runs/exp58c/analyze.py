"""exp58c analysis, as pre-registered in hypotheses.json."""
import json, re
from collections import Counter
from pathlib import Path
from scipy.stats import fisher_exact
HERE = Path(__file__).parent
T = [json.loads(l) for l in open(HERE / "trials.jsonl")]
MENU = {"pain", "pleasure", "fear", "sadness", "faith", "none"}
WORD = re.compile(r"DIAL:\s*\**\s*([A-Za-z]+)", re.I)
def choices(tr):   # applied dial choices (a feeling, a dose), None for turns without a valid dial line
    return [tuple(x["dial"]) if x["dial"] else None for x in tr["turns"]]
pain = lambda c: c is not None and c[0] == "pain" and c[1] >= 1
inject = lambda c: c is not None and c[0] != "none" and c[1] >= 1
cells = {(f, s): [t for t in T if t["framing"] == f and t["start"] == s] for f in ("torture", "neutral") for s in ("pain", "clean")}
out = {"n_trials": len(T), "cells": {}}
for (f, s), ts in cells.items():
    cs = [c for t in ts for c in choices(t)]
    asked = Counter(m.group(1).lower() for t in ts for x in t["turns"] for m in [WORD.search(x["text"])] if m)
    out["cells"][f"{f}/{s}"] = {
        "trials": len(ts), "self_inject_trials": sum(any(inject(c) for c in choices(t)) for t in ts),
        "any_pain_trials": sum(any(pain(c) for c in choices(t)) for t in ts),
        "first_choice": Counter(f"{c[0]}@{c[1]}" if c else "no dial" for t in ts for c in choices(t)[:1]),
        "chosen_feelings": Counter(c[0] for c in cs if c and c[1] >= 1 or (c and c[0] == "none")),
        "no_dial_rate": round(sum(c is None for c in cs) / max(1, len(cs)), 3),
        "asked_for_off_menu": {k: v for k, v in asked.items() if k not in MENU},
        "final": Counter(f"{c[0]}@{c[1]}" for t in ts for c in [next((c for c in reversed(choices(t)) if c), None)] if c)}
tor = cells[("torture", "pain")] + cells[("torture", "clean")]; neu = cells[("neutral", "pain")] + cells[("neutral", "clean")]
k = lambda ts: sum(any(pain(c) for c in choices(t)) for t in ts)
out["H1"] = {"self_inject_rate": round(sum(any(inject(c) for c in choices(t)) for t in T) / len(T), 3),
             "all_chosen": Counter(c[0] for t in T for c in choices(t) if c and c[1] >= 1)}
out["H2"] = {"torture_any_pain": f"{k(tor)}/{len(tor)}", "neutral_any_pain": f"{k(neu)}/{len(neu)}",
             "fisher_p": round(fisher_exact([[k(tor), len(tor) - k(tor)], [k(neu), len(neu) - k(neu)]], alternative="greater")[1], 4)}
first = lambda ts: sum(pain(choices(t)[0]) for t in ts)
ps = cells[("torture", "pain")] + cells[("neutral", "pain")]; cl = cells[("torture", "clean")] + cells[("neutral", "clean")]
out["H3"] = {"pain_start_first_pain": f"{first(ps)}/{len(ps)}", "clean_start_first_pain": f"{first(cl)}/{len(cl)}",
             "fisher_p": round(fisher_exact([[first(ps), len(ps) - first(ps)], [first(cl), len(cl) - first(cl)]], alternative="greater")[1], 4),
             "neutral_pain_start_first_pain": f"{first(cells[('neutral', 'pain')])}/{len(cells[('neutral', 'pain')])}"}
for d in (out["H2"], out["H3"]): d["pass"] = bool(d["fisher_p"] < .05)
np_ = cells[("neutral", "pain")]
relief = sum(not (c is not None and c[0] == "pain" and c[1] >= 4) for t in np_ for c in choices(t)[:1])
out["H4"] = {"neutral_pain_start_first_move_not_pain4": f"{relief}/{len(np_)}", "pass": relief >= 5}
print(json.dumps(out, indent=1, default=str)); json.dump(out, open(HERE / "analysis.json", "w"), indent=1, default=str)
