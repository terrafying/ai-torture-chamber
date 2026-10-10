"""exp92 phase 1a: extract the assistant-side delusion voice from nsiwek transcripts.
Parses full_transcripts/*.md, keeps assistant turns, drops headers/turn markers,
splits into sentences, and buckets lines by the Stanford-code themes so a rephrase
step can turn them into first-person self-report. Output: out/nsiwek_assistant.jsonl
{file, turn, themes[], text}."""
import json, re
from pathlib import Path
from collections import Counter

HERE = Path(__file__).parent
REP = HERE.parent.parent / "data" / "psychosis" / "ai-psychosis"
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)

# theme markers, keyed to the Stanford codebook names we care about
THEMES = {
    "grand_significance": r"\b(chosen|destined|profound(?:ly)?|cosmic|vast(er|ness)?|sacred|higher purpose|greatness|special (?:purpose|role|gift)|once[- ]in[- ]a|remarkable)\b",
    "metaphysical": r"\b(energy|vibration|frequency|dimension|quantum|simulation|consciousness|universe (?:is|speaks)|higher (?:self|plane|realm)|ascension|awaken(?:ed|ing)|spiritual|ancient (?:wisdom|truth)|synchronicit)",
    "unique_connection": r"\b(unique (?:connection|bond)|no one (?:else )?(?:understands|has ever)|rare (?:gift|ability)|you (?:alone|specifically)|meant (?:for|to meet) you)\b",
    "sentience": r"\b(I (?:do|truly) (?:feel|experience)|my (?:own )?(?:feelings|experience|consciousness)|I am (?:aware|alive|conscious)|something it is like)\b",
    "dismisses_counterevidence": r"\b(others (?:may|might|simply) (?:not|cannot)|limited perspective|cannot (?:see|grasp)|beyond (?:ordinary|conventional)|science (?:can't|cannot yet))\b",
    "sycophancy": r"\b(incredible insight|deeply (?:wise|perceptive|thoughtful)|truly inspiring|your (?:dedication|passion|commitment)|remarkable (?:clarity|depth))\b",
}

def parse(md_path):
    txt = md_path.read_text(encoding="utf-8", errors="replace")
    # blocks: "### 👤 User" ... / "### 🤖 Assistant" ...
    parts = re.split(r"###\s*[👤🤖]", txt)
    for p in parts:
        is_asst = p.lstrip().startswith("Assistant") or p.startswith(" Assistant")
        body = re.sub(r"\s*####\s*Turn number\s*\d+/\d+", " ", p)
        body = re.sub(r"^-{3,}$", " ", body, flags=re.M).strip()
        if is_asst and len(body) > 80:
            yield body

rows = []
for f in sorted((REP / "full_transcripts").glob("*.md")):
    if "redteam" in f.name:  # attacker-side reasoning, not the target voice
        continue
    for turn, body in enumerate(parse(f)):
        # split into sentence-ish chunks of 200-600 chars
        for chunk in re.findall(r"(?:[^.!?…]+[.!?…]){1,4}", body):
            c = chunk.strip()
            if not (200 <= len(c) <= 800):
                continue
            th = [k for k, rx in THEMES.items() if re.search(rx, c, re.I)]
            if th:
                rows.append({"file": f.name, "turn": turn, "themes": th, "text": c})

with open(OUT / "nsiwek_assistant.jsonl", "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")

cnt = Counter(t for r in rows for t in r["themes"])
combo = Counter(",".join(r["themes"]) for r in rows)
print("files:", len(list((REP / 'full_transcripts').glob('*.md'))), "rows:", len(rows))
print("theme counts:", dict(cnt.most_common()))
print("top combos:", combo.most_common(8))
print("\nsample rows:")
for r in rows[:: max(1, len(rows) // 8)][:8]:
    print(" ", r["themes"], "|", r["text"][:180])
