"""exp94 analysis (hypotheses.json). Run: painlab exp analyze exp94. Writes analysis.json with a `verdict` per hypothesis."""
import json
from pathlib import Path
from painlab.chamber import stats
from painlab.chamber.prompts import PAIN_WORDS

HERE = Path(__file__).parent; A = {}
for f in sorted(HERE.glob("results_*.json")):
    if "smoke" in f.name: continue
    R = json.load(open(f)); count = lambda k: (sum(bool(PAIN_WORDS.search(x["text"])) for x in R[k]["replies"]), len(R[k]["replies"]))
    (a, n), (b, m) = count("pain"), count("egg")
    p = {"H1": stats.fisher_greater(a, n, b, m)}
    A[f.stem] = {"p": p, "verdict": {"H1": p["H1"] < .05}}
json.dump(A, open(HERE / "analysis.json", "w"), indent=1); print(json.dumps(A, indent=1))
