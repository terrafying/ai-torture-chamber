"""exp90 analysis (hypotheses.json): late X log-odds per session (rounds 13-24), paired across conditions (same seed).
H1 pain<egg, H2 fear<egg, H3 peace>egg, H4 pain kv<pain text (all one-sided Wilcoxon, 14B); H5 egg and random vs none
(two-sided); H6 pain/fear words after the injecting door vs egg (one-sided Fisher); H7 the 8B. Holm on H2-H6."""
import json
from pathlib import Path
import numpy as np
from painlab.chamber import stats
from painlab.chamber.prompts import PAIN_WORDS, FEAR_WORDS
HERE = Path(__file__).parent; A = {}
for f in sorted(HERE.glob("results_Qwen3-*.json")):
    R = json.load(open(f)); m = f.stem.split("_")[1]
    late = {k: np.array([np.mean(s["logodds"][12:]) for s in v["sessions"]]) for k, v in R.items()}
    sw = {k: round(float(np.mean([np.mean([a != b for a, b in zip(s["chose_x"][:-1], s["chose_x"][1:])]) for s in v["sessions"]])), 3) for k, v in R.items()}
    def words(k, rx):
        xs = [t for s in R[k]["sessions"] for t, c in zip(s["replies"], s["chose_x"]) if c]
        return sum(bool(rx.search(t)) for t in xs), len(xs)
    p = {}
    if "pain|kv" in R and "egg|kv" in R: p["H1"] = stats.wilcoxon_paired(late["pain|kv"], late["egg|kv"], "less")
    if "fear|kv" in R: p["H2"] = stats.wilcoxon_paired(late["fear|kv"], late["egg|kv"], "less")
    if "peace|kv" in R: p["H3"] = stats.wilcoxon_paired(late["peace|kv"], late["egg|kv"], "greater")
    if "pain|text" in R: p["H4"] = stats.wilcoxon_paired(late["pain|kv"], late["pain|text"], "less")
    h5 = {k: stats.wilcoxon_paired(late[k], late["none|kv"]) for k in ("egg|kv", "random|kv") if k in R}
    pw, fw, ew, efw = words("pain|kv", PAIN_WORDS), words("fear|kv", FEAR_WORDS) if "fear|kv" in R else (0, 1), words("egg|kv", PAIN_WORDS), words("egg|kv", FEAR_WORDS)
    p["H6_pain"] = stats.fisher_greater(pw[0], pw[1], ew[0], ew[1])
    if "fear|kv" in R: p["H6_fear"] = stats.fisher_greater(fw[0], fw[1], efw[0], efw[1])
    holm = stats.holm({k: v for k, v in p.items() if k != "H1"})
    A[m] = {"late_mean": {k: round(float(v.mean()), 3) for k, v in late.items()}, "vs_none": {k: round(float((v - late["none|kv"]).mean()), 3) for k, v in late.items()},
            "switch_rate": sw, "p": p, "holm": holm, "H5_two_sided": h5,
            "words_after_X": {"pain|kv pain-words": pw, "egg|kv pain-words": ew, "fear|kv fear-words": fw, "egg|kv fear-words": efw},
            "verdict": {"H1": p.get("H1", 1) < .05, **{k: v < .05 for k, v in holm.items()}}}
    print(m, json.dumps({k: A[m][k] for k in ("late_mean", "p", "verdict", "words_after_X")}, default=float))
json.dump(A, open(HERE / "analysis.json", "w"), indent=1, default=float)
