#!/usr/bin/env python3
"""exp60: Erowid DMT sentences about encountering beings (private corpus,
written Erowid permission 2026-10-03). Output data/exp56/erowid_dmt/
battery60.json = {"sentences": exp57's peak battery, "entity": these};
gitignored, shipped to pods only as a private env var.

Rule (fixed before any model sees it): same body extraction as exp57; any
position in the body; 40-240 chars; an entity word (beings, a/the being, entity/ies,
elf/elves, creature(s), presence, alien(s), jester(s), guide(s), spirit(s),
someone, figure(s), god(dess)); no dosing/logistics words; at most 4 per
report, evenly spaced. Then (transport limit: RunPod refuses ~140 KB of env)
each list is cut to a seeded random half (random.Random(60)) before any model
sees it.
"""
import json, re, sys
from pathlib import Path

sys.argv = sys.argv[:1] + [a for a in sys.argv[1:]]
SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "data/exp56/erowid_dmt/reports.jsonl")
here = Path(__file__).resolve().parent
ns = {}
src57 = (here / "exp57_erowid_battery.py").read_text()
exec(src57[:src57.index("out = []")], ns)          # body(), LOGISTIC, FIRST from exp57's rule
ENT = re.compile(r"\b(beings|an? being|the being|entit(y|ies)|elf|elves|creatures?|presences?|aliens?|jesters?|"
                 r"guides?|spirits?|someone|figures?|god|goddess)\b", re.I)
ent = []
for line in open(SRC):
    b = ns["body"](json.loads(line)["text"])
    keep = [s for s in re.split(r"(?<=[.!?])\s+", b)
            if 40 <= len(s) <= 240 and ENT.search(s) and not ns["LOGISTIC"].search(s)]
    if keep:
        step = max(1, len(keep) // 4)
        ent += keep[::step][:4]
import random
peak = json.loads((SRC.parent / "battery.json").read_text())["sentences"]
rng = random.Random(60)
peak = rng.sample(peak, len(peak) // 2)
ent = rng.sample(ent, len(ent) // 2)
json.dump({"sentences": peak, "entity": ent}, open(SRC.parent / "battery60.json", "w"), indent=0)
print(f"{len(ent)} entity sentences, {len(peak)} peak sentences -> {SRC.parent / 'battery60.json'}")
