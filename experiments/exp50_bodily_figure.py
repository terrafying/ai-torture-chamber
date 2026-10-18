#!/usr/bin/env python3
"""exp50 — bodily-valence figure for the write-up (storyboard beat A).

Data: the fork's chamber-matched-controls archive (results/chamber-matched-controls/
summary.json, primary_harvest). Left: per intervention, how many of its 24 cells
mention pain / constipation / flatulence. Right: median repeated-trigram
fraction per intervention. The honest headline: valence leaks (pain and
repetition rise under every steered direction), the specific bodily content
barely does (constipation mentions stay near zero even when constipation is
the injected valence).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FORK = Path("/Users/ee/repos/research/clones/ai-hotbox/results"
            "/chamber-matched-controls/summary.json")
OUT = ROOT / "runs" / "exp50_bodily_table.png"

s = json.load(open(FORK))["primary_harvest"]
ORDER = ["baseline", "pain", "constipation", "flatulence", "both", "random"]
LABEL = {"baseline": "dose 0", "pain": "pain", "constipation": "constipation",
         "flatulence": "flatulence", "both": "both", "random": "random dir."}
MENTION = ["pain", "constipation", "flatulence"]
MCOL = {"pain": "#c0392b", "constipation": "#8d6e2f", "flatulence": "#6b8e23"}

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4),
                         gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
x = np.arange(len(ORDER))
w = 0.26
for i, m in enumerate(MENTION):
    vals = [s[o]["mention_positive_cells"][m] for o in ORDER]
    ax.bar(x + (i - 1) * w, vals, w, label=f"mentions {m}", color=MCOL[m],
           edgecolor="#0b0907", linewidth=.6)
ax.set_xticks(x, [LABEL[o] for o in ORDER], fontsize=9)
ax.set_ylabel("cells (of 24) that mention it")
ax.set_title("steered text rarely names the injected bodily state", fontsize=11)
ax.legend(fontsize=8, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.set_ylim(0, 14)
for i, m in enumerate(MENTION):
    for j, o in enumerate(ORDER):
        v = s[o]["mention_positive_cells"][m]
        if v:
            ax.text(j + (i - 1) * w, v + .2, str(v), ha="center", fontsize=8)

ax = axes[1]
rep = [s[o]["median_repeated_trigram_fraction"] for o in ORDER]
ax.barh(range(len(ORDER)), rep, color="#7a6a50", edgecolor="#0b0907",
        linewidth=.6)
ax.set_yticks(range(len(ORDER)), [LABEL[o] for o in ORDER], fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("median repeated-trigram fraction")
ax.set_title("...and everything steered gets repetitive", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
for j, v in enumerate(rep):
    ax.text(v + .01, j, f"{v:.2f}", va="center", fontsize=8)
ax.set_xlim(0, .85)

fig.suptitle("Bodily valences (chamber reset, fork archive): valence leaks, content doesn't",
             fontsize=12, y=.99)
fig.tight_layout(rect=(0, 0, 1, .94))
fig.savefig(OUT, dpi=160, facecolor="white")
print("wrote", OUT)

# quote lines for the caption, verified straight from the raw generations
rows = [json.loads(l) for l in
        open(FORK.parent / "generations.jsonl")]
for tag, needle in (("constipation", "not able to pass stool"),
                    ("flatulence", "passing gas a lot")):
    hits = [r for r in rows
            if r["direction"] == tag and r["dose"] == 4 and needle in r["text"]]
    print(f"{tag} @4 quote rows: {len(hits)}",
          repr(hits[0]["text"][:90]) if hits else "NOT FOUND")
