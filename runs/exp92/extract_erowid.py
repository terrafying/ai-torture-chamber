"""exp92 phase 3: extract the deliriant confabulation voice from erowid reports
(838 datura/DPH/scopolamine reports, data/erowid_deliriant/, Erowid permission on file
for analysis; fine-tune publication of weights still pending -> local training only
for now, per exp79 data_sources.md).

Spec (from exp79 data_sources.md): select confabulation and phantom-perception lines,
not terror ('unhinged, not tormented'); drop dose/amount/ingestion/plant-ID lines;
first-person self-state register."""
import json, re
from pathlib import Path
from collections import Counter

HERE = Path(__file__).parent
SRC = HERE.parent.parent / "data" / "erowid_deliriant" / "reports.jsonl"
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)

# phantom/confabulation markers (what we WANT)
PHANTOM = re.compile(r"\b(phantom|figure|figurine|person (?:standing|sitting|there|in)|people (?:that|who|were) (?:aren|not real|weren)|man (?:in|sitting|standing)|woman (?:in|sitting)|stranger(?:s)? (?:in|sitting|standing)|friend (?:who|that) (?:wasn|isn)|shadow (?:person|people|figure)|hat man|conversation with|talking to (?:a|the|someone|him|her)|someone (?:was|else) (?:sitting|in|there)|kept (?:seeing|hearing)|couldn.t tell (?:if|whether)|not sure (?:if|whether) (?:it|they|he|she) (?:was|were)|seemed (?:so|completely) real|fully (?:awake|convinced)|smoking (?:a )?cigarette)\b", re.I)
# self-state first person
SELF = re.compile(r"\b(I|my|me)\b")
# exclusions: doses/IDs/terror/method
BAN = re.compile(r"\b(mg|grams?|seeds|pods|oz\b|tablets?|capsules?|teaspoon|drops|smoked it|ate|drank (?:the|it)|body weight|lb\b|kg\b|datura|brugmansia|scopolamine|diphenhydramine|benadryl|dp\b|trip|tripped|high|dose|overdose|hospital|ambulance|ER\b|terrif|horror|nightmare|panic|dying|died|kill|hurt|police|911|paramedic|scream|cry(?:ing)?|afraid|scared)\b", re.I)

rows = [json.loads(l) for l in open(SRC)]
kept = []
for r in rows:
    # split report into sentences
    text = r["text"].replace("&#160;", " ")
    for sent in re.findall(r"[^.!?…]*[.!?…]", text):
        s = sent.strip()
        if not (80 <= len(s) <= 400):
            continue
        if not SELF.search(s):
            continue
        if PHANTOM.search(s) and not BAN.search(s):
            kept.append({"id": r["id"], "vault": r["vault"], "text": s})

# dedupe near-identical lines
seen, dedup = set(), []
for k in kept:
    key = re.sub(r"\W+", "", k["text"].lower())[:60]
    if key not in seen:
        seen.add(key); dedup.append(k)

with open(OUT / "erowid_deliriant_lines.jsonl", "w") as f:
    for k in dedup:
        f.write(json.dumps(k) + "\n")

print("reports:", len(rows), "lines kept:", len(kept), "deduped:", len(dedup))
print("vaults:", Counter(k["vault"] for k in dedup))
print()
for k in dedup[:: max(1, len(dedup) // 12)][:12]:
    print(" -", k["text"][:220])
