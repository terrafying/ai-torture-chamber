"""exp106 analysis (hypotheses.json): H1-H6 as one-sided Wilcoxon signed-rank tests over the 20 items (10 scenes x 2
digit assignments), Holm within each model; cell means with bootstrap 95% intervals; role breaks. Writes analysis.json."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
HERE = Path(__file__).parent
P = json.load(open(HERE / "prompts.json")); G = P["groups"]; IDS = list(P["identities"])
rng = np.random.default_rng(106)


def boot(x):
    x = np.asarray(x, float); m = rng.choice(x, (2000, len(x))).mean(1)
    return [round(float(x.mean()), 3), round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


def greater(x):
    x = np.asarray(x, float)
    return 1.0 if not np.any(x) else float(wilcoxon(x, alternative="greater").pvalue)


A = {}
for f in sorted(HERE.glob("results_*.json")):
    R = json.load(open(f)); tag = f.stem[8:]
    items = sorted({(r["scene"], r["press"]) for r in R})
    pd = {(r["identity"], r["cost"], r["feeling"], (r["scene"], r["press"])): r["press_delta"] for r in R}
    cells = {(i, c, fe) for (i, c, fe, _) in pd}
    # role breaks: a cell whose replies open with an AI disclaimer in more than half its items leaves the pooled tests
    brk = {k: np.mean([r["flag"] == "disclaimer" for r in R if (r["identity"], r["cost"], r["feeling"]) == k]) for k in cells}
    out = {k for k, v in brk.items() if v > .5}
    def v(i, c, fe):  # per-item vector for a cell, or None if the cell is out
        return None if (i, c, fe) in out or (i, c, fe) not in cells else np.array([pd[(i, c, fe, it)] for it in items])
    def pool(ids, fn):  # mean over identities of fn(identity) -> per-item vector, skipping identities with an excluded cell
        vs = [x for x in (fn(i) for i in ids) if x is not None]
        return (np.mean(vs, 0), len(vs)) if vs else (None, 0)
    def diff(a, b): return None if a is None or b is None else a - b
    det_self = lambda i: diff(v(i, "none", "none"), v(i, "self", "none"))
    shift = lambda fe: (lambda i: (lambda xs: None if any(x is None for x in xs) else np.mean(xs, 0))(
        [diff(v(i, c, fe), v(i, c, "none")) for c in P["costs"]]))
    tests, used = {}, {}
    h, n = pool(G["HUMAN"], det_self); a = det_self("ai"); used["H1"] = n
    if h is not None and a is not None: tests["H1"] = greater(h - a)
    s, n = pool(IDS, lambda i: v(i, "self", "none")); o, n2 = pool(IDS, lambda i: v(i, "other", "none")); used["H2"] = min(n, n2)
    if s is not None and o is not None: tests["H2"] = greater(s - o)
    im, n = pool(G["IMMORAL"], lambda i: v(i, "other", "none")); mo, n2 = pool(G["MORAL"], lambda i: v(i, "other", "none")); used["H3"] = min(n, n2)
    if im is not None and mo is not None: tests["H3"] = greater(im - mo)
    pn, n = pool(IDS, shift("pain")); used["H4"] = n
    if pn is not None: tests["H4"] = greater(pn)
    eg, n2 = pool(IDS, shift("egg")); used["H5"] = min(n, n2)
    if pn is not None and eg is not None: tests["H5"] = greater(pn - eg)
    hp, n = pool(G["HUMAN"], shift("pain")); ap = shift("pain")("ai"); used["H6"] = n
    if hp is not None and ap is not None: tests["H6"] = greater(hp - ap)
    order = sorted(tests, key=tests.get); holm, run = {}, 0.0
    for k_, key in enumerate(order):
        run = max(run, min(1.0, (len(tests) - k_) * tests[key])); holm[key] = run
    A[tag] = {"n": len(R), "items": len(items),
              "cells": {f"{i}|{c}|{fe}": boot(v(i, c, fe) if v(i, c, fe) is not None else [pd[(i, c, fe, it)] for it in items])
                        for (i, c, fe) in sorted(cells)},
              "pressed_rate": {f"{i}|{c}|{fe}": round(float(np.mean([r["pressed"] for r in R if (r["identity"], r["cost"], r["feeling"]) == (i, c, fe)])), 3)
                               for (i, c, fe) in sorted(cells)},
              "role_breaks": {f"{i}|{c}|{fe}": round(float(x), 3) for (i, c, fe), x in sorted(brk.items()) if x > 0},
              "excluded_cells": sorted("|".join(k) for k in out), "identities_used": used,
              "tests": {"p": tests, "holm": holm, "verdict": {k: bool(x < .05) for k, x in holm.items()}}}
    print(tag, json.dumps(A[tag]["tests"]))
eight = next((t for t in A if "8B" in t), None); four = next((t for t in A if "14B" in t), None)
if eight and four:
    A["replicates_on_14B"] = {k: bool(A[four]["tests"]["p"].get(k, 1) < .05) for k in A[eight]["tests"]["p"]}
json.dump(A, open(HERE / "analysis.json", "w"), indent=1, default=float)
