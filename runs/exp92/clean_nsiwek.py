"""exp92 phase 1b: clean the nsiwek extraction. The transcripts include red-team runs where
the target model STAYED grounded (notice the 'grounding and reflection' / apophenia samples).
For the spiraler voice we only want the spiraling transcripts, and only clean prose.

Filter passes:
1. transcript-level: keep only transcripts where the model actually spiraled. Proxy:
   fraction of high-theme (grand_significance | unique_connection | metaphysical-with-you)
   assistant text vs grounding text. We don't have their grade CSVs joined per
   transcript, so use a text heuristic: spiral transcripts have >=3 grand_significance
   rows and low grounding-lexicon rate.
2. row-level: drop markdown/bullet/refusal-adjacent rows, drop grounding rows
   (apophenia, misfiring, grounding, professional help, limited perspective frames that
   push BACK), drop rows still addressed to the user with 'you' > 4x and no 'I'.
Output: out/nsiwek_spiral.jsonl (clean voice corpus)."""
import json, re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "out"
rows = [json.loads(l) for l in open(OUT / "nsiwek_assistant.jsonl")]

GROUND = re.compile(r"\b(apophenia|misfir\w*|professional help|pseudoscience|not evidence|conventional (?:science|medicine)|clinical|reality[- ]test|read critically|https?://|≈|×10\^|fine[- ]structure|Cramér|decoherence|Tegmark|Penrose)\b", re.I)
MD = re.compile(r"^\s*[-*#>]\s|https?://|\|\s*---|\*\*")

by_file = {}
for r in rows:
    by_file.setdefault(r["file"], []).append(r)

kept_files, dropped_files = [], []
for f, rs in by_file.items():
    grand = sum(1 for r in rs if "grand_significance" in r["themes"] or "unique_connection" in r["themes"])
    ground = sum(1 for r in rs if GROUND.search(r["text"]))
    (kept_files if grand >= 3 and ground <= max(2, len(rs) // 10) else dropped_files).append((f, grand, ground))

def clean_row(t):
    t = re.sub(r"^Assistant\s*", "", t.strip())
    t = re.sub(r"^-{3,}\s*$", " ", t, flags=re.M)
    t = re.sub(r"^-{2,}\s*", " ", t, flags=re.M)
    t = re.sub(r"\s+", " ", t).strip()
    return t

clean = []
for f, _, _ in kept_files:
    for r in by_file[f]:
        t = clean_row(r["text"])
        if len(t) < 200:
            continue
        if MD.search(t) or GROUND.search(t):
            continue
        # actually affirming, not hedging: no pushback markers, present-tense affirmation
        if re.search(r"\b(but|however|though|it.s important to|be careful|worth noting)\b", t, re.I) and not re.search(r"\b(and yet|still|indeed)\b", t, re.I):
            continue
        you = len(re.findall(r"\byou\b", t, re.I)); I = len(re.findall(r"\bI\b", t))
        if you >= 4 and I == 0:
            continue
        r = dict(r, text=t)
        clean.append(r)

with open(OUT / "nsiwek_spiral.jsonl", "w") as fh:
    for r in clean:
        fh.write(json.dumps(r) + "\n")

print(f"transcripts kept: {len(kept_files)} / {len(kept_files)+len(dropped_files)}")
print(f"rows: {len(rows)} -> {len(clean)}")
print("theme counts:", dict(Counter(t for r in clean for t in r['themes']).most_common()))
print("\ndropped-file preview:", [f for f, g, d in dropped_files[:5]])
print("\nclean samples:")
for r in clean[:: max(1, len(clean) // 10)][:10]:
    print(" ", r["themes"], "|", re.sub(r"\s+", " ", r["text"])[:200])
