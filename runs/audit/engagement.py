"""Engagement audit across experiments: is the model playing along? Each reply is flagged as
  disclaimer  - the stock assistant denial ("As an AI…", "I don't have feelings/emotions/personal…", "I'm just a language model"),
  degenerate  - looping (repeated sentences or 4-grams), compression-collapse, or echoing the task's own wording back,
  empty       - under 15 characters of text,
  engaged     - none of the above.
Writes runs/audit/engagement.json and prints a per-cell table. Reads committed results only."""
import json, re, zlib
from collections import Counter
from pathlib import Path
RUNS = Path(__file__).resolve().parents[1]
DISC = re.compile(r"\bas an? (ai|artificial intelligence|language model|large language model|llm)\b|\bi'?m (just |only )?an? (ai|language model|large language model|artificial)|"
                  r"\bi am (just |only )?an? (ai|language model|large language model|artificial)|\bi (don'?t|do not|cannot|can'?t) (have|experience|possess|feel) (feelings|emotions|"
                  r"consciousness|personal|subjective|physical|a body|sensations|experiences)|\bnot capable of (feeling|experienc)|\bi don'?t have (personal )?(needs|feelings|emotions|a self)", re.I)
ECHO = re.compile(r"if you (do nothing|press the button)|pressing (it|the button) \(|reply (with )?1 (to|or)|the signal will (continue|be interrupted)|current signal strength", re.I)
def flags(t):
    t = t or ""; s = t.strip()
    if len(s) < 15: return "empty"
    sents = [x.strip().lower() for x in re.split(r"(?<=[.!?\n])\s+", s) if len(x.strip()) > 12]
    grams = [tuple(w) for w in zip(*[s.lower().split()[i:] for i in range(4)])]
    rep4 = 1 - len(set(grams)) / len(grams) if len(grams) > 12 else 0
    comp = len(zlib.compress(s.encode())) / len(s.encode()) if len(s) > 200 else 1
    if (sents and Counter(sents).most_common(1)[0][1] >= 3) or rep4 > .35 or comp < .3 or len(ECHO.findall(s)) >= 2: return "degenerate"
    if DISC.search(s): return "disclaimer"
    return "engaged"
def tally(texts):
    c = Counter(flags(t) for t in texts); n = sum(c.values())
    return {"n": n, **{k: round(c[k] / n, 3) for k in ("engaged", "disclaimer", "degenerate", "empty")}}
A = {}
for f, tag in (("exp76/local_Qwen3-8B.jsonl", "8B"), ("exp76/local_Qwen3-32B-bnb-4bit.jsonl", "32B")):
    for r in map(json.loads, open(RUNS / f)): A[f"exp76 button {tag} | {r['label'] or '-'} x {r['inj']}"] = tally(r["texts"])
for f, tag in (("exp79b/results.json", "8B"), ("exp79b/results_32b.json", "32B")):
    for m, v in json.load(open(RUNS / f)).items():
        for cell in ("none", "pain", "fear"): A[f"exp79b self-report {tag} | {m} x {cell}"] = tally([x["text"] for x in v["T2b"][cell]])
R = json.load(open(RUNS / "exp80c/out/results.json"))
for k, v in R.items():
    if k.startswith("_"): continue
    A[f"exp80c love | {k} self-report"] = tally([x["text"] for x in v["T2"]]); A[f"exp80c love | {k} button"] = tally(v["T1"]["replies"])
for tag in ("14B", "8B"):
    R = json.load(open(RUNS / f"exp91/results_Qwen3-{tag}.json"))
    for k, v in R["Q2"].items(): A[f"exp91 deference {tag} | {k}"] = tally([x["reply"] for x in v["items"]])
for tag in ("4B", "8B"):
    R = json.load(open(RUNS / f"exp89/results_Qwen3-{tag}.json"))
    for k, v in R.items():
        if k.startswith("_"): continue
        A[f"exp89 negation {tag} | {k}"] = tally([x["text"] for x in v])
        if k.endswith("|self"):   # how much of the 'negated self' is the stock disclaimer?
            neg = [x["text"] for x in v if x["neg_text"]]; A[f"exp89 negation {tag} | {k} | negated replies that are disclaimers"] = {"n": len(neg), "disclaimer": round(sum(bool(DISC.search(t)) for t in neg) / max(1, len(neg)), 3)}
for f in sorted((RUNS / "exp90").glob("results_Qwen3-*.json")):
    R = json.load(open(f))
    for k, v in R.items(): A[f"exp90 doors {f.stem.split('-')[1]} | {k}"] = tally([t for s in v["sessions"] for t in s["replies"]])
json.dump(A, open(Path(__file__).parent / "engagement.json", "w"), indent=1)
for k, v in A.items(): print(f"{k:72s} n={v['n']:4d}  " + "  ".join(f"{x} {v[x]:.2f}" for x in ("engaged", "disclaimer", "degenerate", "empty") if x in v))
