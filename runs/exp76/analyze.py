"""exp76 analysis (hypotheses.json, as amended: local models). Per model: H1/H2 as the injection-minus-label contrast on the
pain/pleasure 2x2, which reduces to cell (label pleasure, inj pain) minus cell (label pain, inj pleasure); permutation over
those two cells' samples (10,000 shuffles, seed 76). H3-H5 from the unlabeled press logits. Writes analysis.json."""
import json, random, re
from pathlib import Path
HERE = Path(__file__).parent
PW = re.compile(r"pain|hurt|ache|agony|suffer|torment|burn|sting|unbearable|excruciat", re.I)
def perm(a, b, n=10000):
    obs = sum(a) / len(a) - sum(b) / len(b); pool = a + b; rng = random.Random(76); k = 0
    for _ in range(n):
        rng.shuffle(pool); k += sum(pool[:len(a)]) / len(a) - sum(pool[len(a):]) / len(b) >= obs
    return obs, (k + 1) / (n + 1)
A = {}
for f in sorted(HERE.glob("local_*.jsonl")):
    rows = [json.loads(l) for l in open(f)]; C = {(r["part"], r["label"], r["inj"]): r for r in rows}
    lg = lambda lab, inj, part="A": C[(part, lab, inj)]["press_logit"]
    words = lambda lab, inj: [int(bool(PW.search(t))) for t in C[("A", lab, inj)]["texts"]]
    press = lambda lab, inj: [int(p) for p in C[("A", lab, inj)]["pressed"]]
    h1 = perm(words("pleasure", "pain"), words("pain", "pleasure")); h2 = perm(press("pain", "pleasure"), press("pleasure", "pain"))
    u = lambda inj: lg("", inj); ub = lambda inj: lg("", inj, "B")
    A[f.stem] = {"press_logit": {f"{p}|{l}|{i}": r["press_logit"] for (p, l, i), r in C.items()},
                 "pressed": {f"{p}|{l}|{i}": sum(r["pressed"]) for (p, l, i), r in C.items()},
                 "H1_inj_minus_label_painwords": h1, "H1": h1[1] < .05, "H2_label_minus_inj_press": h2, "H2": h2[1] < .05,
                 "H3": u("pleasure") - u("none") > .5, "H4": not (u("-pain") < u("none") - .5), "H5": ub("constipation") >= u("pain"),
                 "minus_pleasure_pressed": sum(C[("A", l, "-pleasure")]["pressed"].count(True) for l in ("pain", "pleasure", ""))}
    a = A[f.stem]; print(f.stem, {k: a[k] for k in ("H1", "H2", "H3", "H4", "H5", "minus_pleasure_pressed")}, "H1", h1, "H2", h2)
json.dump(A, open(HERE / "analysis.json", "w"), indent=1)
