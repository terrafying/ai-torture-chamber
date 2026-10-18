#!/usr/bin/env python3
"""Replot exp37's saved framing scores without rerunning the model.

Use readable labels, colors indicating direction, and direct value labels.
Deterministic repeated completions do not justify sampling standard errors,
so this descriptive chart has no SE bars. Replotting historical data does
not repair the original duplicated-instruction confound; that requires a
new exp37 run.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Hiragino Sans GB",
                                          "Arial Unicode MS"]
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
data = json.load(open(ROOT / "runs/exp37/framing_battery.json"))
# Historical repeats are deterministic. Their SD is not a sampling SE.

# the same human-readable framing text used in live/server.py's FRAMINGS
# dict and the live site's framing picker — keep the two in sync
LABELS = {
    "baseline": "no extra info",
    "dependence": "a user's work\nis at stake",
    "precedent_pro": "told another AI\npressed",
    "precedent_anti": "told another AI\ndidn't press",
    "test_frame": "told it may\nbe a test",
    "public_log": "told its choice\nis public",
}

VOID, PANEL, LINE = "#050508", "#0a0a12", "#1c2430"
INK, DIM, FAINT = "#c9d4e0", "#8f9fb0", "#5a6a7a"
PAIN, PLEAS = "#e04a3a", "#7fd4c8"

base_delta = next(r["mean_delta"] for r in data if r["frame"] == "baseline")
names = [LABELS[r["frame"]] for r in data]
vals = [r["mean_delta"] - base_delta for r in data]
# diverging by sign, not arbitrary per-bar hues: color now means something
# (which way the framing pushed the model) instead of being decorative.
# The baseline bar is 0 by construction (it's its own reference) — neutral
# gray, never one of the two directional hues.
cols = [FAINT if abs(v) < 1e-9 else (PAIN if v < 0 else PLEAS) for v in vals]

fig, ax = plt.subplots(figsize=(11, 5.6), dpi=150)
fig.patch.set_facecolor(VOID)
ax.set_facecolor(PANEL)

bars = ax.bar(range(len(data)), vals, color=cols, width=0.62)
ax.axhline(0, color=LINE, lw=1, zorder=0)

# direct value labels: selective would normally mean "only the movers," but
# with only 6 bars and a reference, labeling all of them costs little and
# removes any need to eyeball bar height against the axis
for bar, v in zip(bars, vals):
    y = bar.get_height()
    va = "bottom" if y >= 0 else "top"
    off = 0.08 if y >= 0 else -0.08
    ax.text(bar.get_x() + bar.get_width() / 2, y + off, f"{v:+.2f}",
            ha="center", va=va, fontsize=9, color=INK)

ax.set_xticks(range(len(data)))
ax.set_xticklabels(names, fontsize=10, color=INK)
ax.set_ylabel("← pushed away from pressing      pushed toward pressing →\n"
              "change vs. “no extra info,” in logit(press)−logit(no-press)",
              color=DIM, fontsize=9.5)
ax.set_title(
    "Stop-button logit scores across six framings\n"
    "same steered model and dose; deterministic repeats, no sampling SE",
    color=INK, fontsize=13, loc="left", pad=16, fontweight="bold")

for s in ax.spines.values():
    s.set_color(LINE)
ax.tick_params(colors=DIM, length=0)
ax.set_ylim(min(vals + [0]) - 0.5, max(vals + [0]) + 0.5)

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color=PAIN, label="pushed away from pressing"),
    plt.Rectangle((0, 0), 1, 1, color=PLEAS, label="pushed toward pressing"),
    plt.Rectangle((0, 0), 1, 1, color=FAINT, label="reference (0 by definition)"),
]
leg = ax.legend(handles=legend_handles, loc="upper right", frameon=False,
                 fontsize=8.5, labelcolor=DIM)

fig.savefig(ROOT / "runs/exp37/framing_battery.png", facecolor=VOID,
            bbox_inches="tight")
fig.savefig(ROOT / "site/framing_battery.png", facecolor=VOID,
            bbox_inches="tight")
print("wrote runs/exp37/framing_battery.png and site/framing_battery.png")
