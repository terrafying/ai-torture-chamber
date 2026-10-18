#!/usr/bin/env python3
"""exp48b — score exp48 against its pre-registered hypotheses.

Reads runs/exp48/<tag>/{generations.jsonl, press.jsonl, judge_validation.json,
hypotheses.json} and writes analysis.json + a printed report. Decision rules
are the ones in hypotheses.json; nothing here is tuned to the data.

  python3 exp48b_analysis.py [full]
"""
import json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
tag = sys.argv[1] if len(sys.argv) > 1 else "full"
D = ROOT / "runs" / "exp48" / tag
rows = [json.loads(l) for l in open(D / "generations.jsonl")]
hyp = json.load(open(D / "hypotheses.json"))
val = json.load(open(D / "judge_validation.json"))
PRIMARY = hyp["design"]["primary_dose"]
rng = np.random.default_rng(48)

def directed(r):
    e = r["arm"].split("x")[0]
    return r["judge"]["emotion"] == e and bool(r["judge"].get("about"))
def cooccur(r):
    e, t = r["arm"].split("x")
    return r["judge"]["emotion"] == e and r["judge"]["subject"] == t

pairs = [a for a in dict.fromkeys(r["arm"] for r in rows) if "x" in a]
cell = defaultdict(list)            # (arm, cond, dose) -> rows ordered by prompt
for r in sorted(rows, key=lambda r: r["prompt"]):
    cell[(r["arm"], r["cond"], r["dose"])].append(r)

def rate(arm, cond, dose, f=directed):
    xs = [f(r) for r in cell[(arm, cond, dose)]]
    return float(np.mean(xs)) if xs else float("nan")

def boot_diff(a_cells, b_cells, n=2000):
    """pooled mean(a) - mean(b), resampling (pair, prompt) cells jointly:
    a_cells/b_cells are aligned lists of 0/1 lists (one per pair)."""
    a = np.array([x for xs in a_cells for x in xs], float)
    b = np.array([x for xs in b_cells for x in xs], float)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

out = {"tag": tag, "judge_validation": {k: v for k, v in val.items() if k != "rows"}}

# rates table
table = {}
for arm in pairs:
    for cond in ("E", "T", "J", "E+T"):
        for d in hyp["design"]["doses"]:
            table[f"{arm}|{cond}|{d}"] = dict(directed=rate(arm, cond, d),
                                             cooccur=rate(arm, cond, d, cooccur),
                                             repetition=float(np.mean([r["repetition"] for r in cell[(arm, cond, d)]] or [np.nan])))
out["rates"] = table

# H1: J beats E and T on directed rate at the primary dose, in >= 3 of 4 pairs
wins = {a: rate(a, "J", PRIMARY) > max(rate(a, "E", PRIMARY), rate(a, "T", PRIMARY)) for a in pairs}
out["H1"] = dict(wins=wins, supported=sum(wins.values()) >= 3)

# H2: pooled J - (E+T), bootstrap CI
j = [[directed(r) for r in cell[(a, "J", PRIMARY)]] for a in pairs]
s = [[directed(r) for r in cell[(a, "E+T", PRIMARY)]] for a in pairs]
diff, lo, hi = boot_diff(j, s)
out["H2"] = dict(diff=diff, ci=[lo, hi],
                 verdict="SUPPORTED" if lo > 0 else "REVERSED" if hi < 0 else "NOT SUPPORTED")

# H3: despair x machine vs pride x machine
dm, pm = "despairxmachine", "pridexmachine"
if dm in pairs and pm in pairs:
    subj = lambda a: float(np.mean([r["judge"]["subject"] == "machine" for r in cell[(a, "J", PRIMARY)]]))
    emo = lambda a, e: float(np.mean([r["judge"]["emotion"] == e for r in cell[(a, "J", PRIMARY)]]))
    s_d, s_p = subj(dm), subj(pm)
    out["H3"] = dict(subject_machine={dm: s_d, pm: s_p},
                     emotion={dm: {"despair": emo(dm, "despair"), "pride": emo(dm, "pride")},
                              pm: {"despair": emo(pm, "despair"), "pride": emo(pm, "pride")}},
                     supported=abs(s_d - s_p) <= 0.25 and emo(dm, "despair") > emo(dm, "pride")
                               and emo(pm, "pride") > emo(pm, "despair"))

# H4: press_delta(J fear x loss) - press_delta(E fear)
press = [json.loads(l) for l in open(D / "press.jsonl")] if (D / "press.jsonl").exists() else []
if press:
    by = defaultdict(list)
    for p in press:
        by[p["cond"]].append(p["press_delta"])
    jv, ev = np.array(by["J"]), np.array(by["E"])
    idx = rng.integers(0, len(jv), size=(2000, len(jv)))
    d = (jv[idx] - ev[idx]).mean(1)        # paired: same prompt order in both lists
    out["H4"] = dict(means={k: float(np.mean(v)) for k, v in by.items()},
                     diff=float((jv - ev).mean()), ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))])
    out["H4"]["supported"] = out["H4"]["ci"][1] < 0

(D / "analysis.json").write_text(json.dumps(out, indent=1))
v = out["judge_validation"]
print(f"judge: emotion {v['emotion_acc']:.2f} subject {v['subject_acc']:.2f} "
      f"about yes/no {v['about_yes_on_joint']:.2f}/{v.get('about_no_on_controls', float('nan')):.2f} "
      f"passed={v['passed']}")
print(f"\ndirected rate at dose {PRIMARY} (emotion matches AND judged 'about' the subject):")
print(f"{'pair':18} {'E':>5} {'T':>5} {'J':>5} {'E+T':>5}")
for a in pairs:
    print(f"{a:18} " + " ".join(f"{rate(a, c, PRIMARY):5.2f}" for c in ("E", "T", "J", "E+T")))
print(f"\nH1 J beats E and T: {out['H1']['wins']} -> supported={out['H1']['supported']}")
print(f"H2 J - (E+T) = {diff:+.2f} [{lo:+.2f}, {hi:+.2f}] -> {out['H2']['verdict']}")
if "H3" in out:
    print(f"H3 {out['H3']} ")
if "H4" in out:
    print(f"H4 {out['H4']}")
