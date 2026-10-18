#!/usr/bin/env python3
"""exp57: first-person peak-experience sentences from the private Erowid DMT
corpus (data/exp56/erowid_dmt/reports.jsonl, written Erowid permission
2026-10-03). Output data/exp56/erowid_dmt/battery.json stays local
(gitignored) and only reaches the GPU pod as a private env var; only the
derived direction and our model's own outputs are ever published.

Rule (fixed before any model sees it): body = text after the dose header and
before the "Exp Year" footer; keep sentences from the middle half of the body
(the peak), 50-220 chars, containing I/me/my, with no dosing or logistics
words; at most 3 per report, evenly spaced.
"""
import html, json, re, sys
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "data/exp56/erowid_dmt/reports.jsonl")
OUT = SRC.parent / "battery.json"
LOGISTIC = re.compile(r"\b(mg|mcg|gram|grams|dose|dosage|smok|vapori|pipe|bowl|lighter|"
                      r"inhal|hit|toke|torch|extract|erowid|t\+|weight|lb|kg|bong|"
                      r"meth|cannabis|weed|ayahuasca|mao|harm)", re.I)
FIRST = re.compile(r"\b(I|me|my|I'm|I've|myself)\b")


def body(t):
    t = html.unescape(t).replace("\xa0", " ")
    m = re.search(r"BODY WEIGHT:[^A-Z]*?(lb|kg)\b", t)
    start = m.end() if m else 0
    # dose-block lines end before the first sentence of prose
    t = t[start:]
    t = re.sub(r"\[Erowid Note:.*?\]", " ", t, flags=re.S)
    end = t.find("Exp Year:")
    return " ".join((t[:end] if end > 0 else t).split())


out = []
for line in open(SRC):
    r = json.loads(line)
    b = body(r["text"])
    sents = re.split(r"(?<=[.!?])\s+", b)
    mid = sents[len(sents) // 4: 3 * len(sents) // 4]
    keep = [s for s in mid if 50 <= len(s) <= 220 and FIRST.search(s) and not LOGISTIC.search(s)]
    if keep:
        step = max(1, len(keep) // 3)
        out += keep[::step][:3]
json.dump({"n_reports": sum(1 for _ in open(SRC)), "sentences": out}, open(OUT, "w"), indent=0)
print(f"{len(out)} sentences -> {OUT}")
