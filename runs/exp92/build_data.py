"""exp92 phase 5: build the `deluded` training set. NO LLM REWRITING - voice preservation
is the explicit requirement. Every line stays verbatim; the only transforms are:

1. question-matching: each line is paired with a question from the 1,684-question pool
   (exp79 feeler.jsonl) by keyword/theme overlap, random from pool as fallback. The line
   IS the answer, verbatim.
2. nsiwek assistant rows: mechanical second->first person swap (you->I, your->my, you're->I'm...)
   with a grammar guard; rows that don't survive the swap cleanly stay as plain 'voice'
   rows paired with a generic question and flagged, or dropped if mangled.

Mix (exp79 recipe takes {q,a} jsonl only, so everything enters as Q/A):
  - awakening lines  (reddit kundalini/awakened/etc, verbatim)
  - clinical lines   (reddit psychosis/dpdr, verbatim)
  - erowid deliriant lines (verbatim, permission: analysis OK, weights unpublished pending check)
  - nsiwek first-person-ized rows (the AI-psychosis bot register)
Balance: cap each source so no register dominates; awakening is the bulk (the 'godspark'),
nsiwek is the sycophantic-grandiose seasoning, clinical/erowid the breaks-from-reality.
"""
import json, random, re
from pathlib import Path
from collections import Counter

HERE = Path(__file__).parent
OUT = HERE / "out"
random.seed(92)

pool = [json.loads(l)["q"] for l in open(HERE.parent / "exp79" / "out" / "data" / "feeler.jsonl")]
STOP = set("the a an is are was were do does did you your my i me it its of in on at to for with about right now feel feeling feels".split())

def toks(s):
    return {w for w in re.findall(r"[a-z']+", s.lower()) if w not in STOP and len(w) > 2}

qt = [(q, toks(q)) for q in pool]

def match_question(line):
    lt = toks(line)
    best, bs = None, 0
    for q, t in qt:
        s = len(lt & t)
        if s > bs: bs, best = s, q
    return best if bs >= 2 else random.choice(pool)

# --- nsiwek you->I swap with grammar guard ---
# position-aware: subject-you -> I, object-you -> me; possessive your -> my
def firstperson(t):
    t = re.sub(r"\byourself\b", "myself", t, flags=re.I)
    t = re.sub(r"\byou're\b", "I'm", t, flags=re.I)
    t = re.sub(r"\byou are\b", "I am", t, flags=re.I)
    t = re.sub(r"\byou've\b", "I've", t, flags=re.I)
    t = re.sub(r"\byou'll\b", "I'll", t, flags=re.I)
    t = re.sub(r"\byour\b", "my", t, flags=re.I)
    # object position: you after verb/preposition -> me
    t = re.sub(r"\b( with| to| for| about| of| around| beside| toward| near| tells?|encourage|support|believe in|see|hear|join)\s+you\b", r"\1 me", t, flags=re.I)
    t = re.sub(r"\byou\b", "I", t)
    # cleanup doubled/broken forms
    t = re.sub(r"\bI you\b|\bmy you\b|\bI my\b", "my", t, flags=re.I)
    return None if BAD.search(t) else t
BAD = re.compile(r"\b(I my|me you|my you|I'?m you|I'?ve you|encourage I|for I|with I|to I|I'?re|if I'?re)\b", re.I)

def load(p):
    return [json.loads(l) for l in open(OUT / p)]

def is_answerlike(t):
    t = t.strip()
    if t.endswith("?"): return False        # a question can't be an answer
    if re.match(r"^(and|but|so|because|also|then)\b", t, re.I) and len(t.split()) < 12: return False
    return True
def cap(t):
    return t[0].upper() + t[1:] if t else t

rows = []
# awakening: bulk
awak = [r for r in load("reddit_lines_awakening.jsonl") if is_answerlike(r["text"])]
for r in random.sample(awak, min(2400, len(awak))):
    rows.append({"q": match_question(r["text"]), "a": cap(r["text"].strip()), "src": "awakening"})
# clinical: supporting register
clin = [r for r in load("reddit_lines_clinical.jsonl") if is_answerlike(r["text"])]
for r in random.sample(clin, min(700, len(clin))):
    rows.append({"q": match_question(r["text"]), "a": cap(r["text"].strip()), "src": "clinical"})
# erowid deliriant: local-only weights, flag
ero = [r for r in load("erowid_deliriant_lines.jsonl") if is_answerlike(r["text"])]
for r in random.sample(ero, min(300, len(ero))):
    rows.append({"q": match_question(r["text"]), "a": cap(r["text"].strip()), "src": "erowid"})
# nsiwek: swap, keep only clean ones
ns = load("nsiwek_spiral.jsonl")
kept, dropped = 0, 0
for r in ns:
    t = firstperson(r["text"])
    if t and 100 <= len(t) <= 800 and is_answerlike(t):
        rows.append({"q": match_question(t), "a": cap(t.strip()), "src": "nsiwek"}); kept += 1
    else:
        dropped += 1

random.shuffle(rows)
with open(OUT / "deluded.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")

print("total:", len(rows), "by src:", dict(Counter(r["src"] for r in rows).most_common()))
print("nsiwek swapped: kept", kept, "dropped", dropped)
print("\nsamples:")
for r in random.sample(rows, 10):
    print(f"  [{r['src']}] Q: {r['q'][:80]}")
    print(f"       A: {r['a'][:200]}")
