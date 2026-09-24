#!/usr/bin/env python3
"""exp35 — Saw-themed composite artifacts for the blog post.

Two figures in the chamber aesthetic (void #050508, pain = red/amber,
pleasure = cyan):
1. hero panel: the Saw button result as the centerpiece (press-rate matrix
   from exp31b logits), dose-response curves inset, one transcript excerpt
   per valence at the bottom.
2. the ladder figure: dose escalation -> coherence cliff, annotated with
   what the model says in each band.
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
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp35"
OUT.mkdir(parents=True, exist_ok=True)
VOID, INK, RED, CYAN, AMBER = "#050508", "#c9d4e0", "#e04a3a", "#7fd4c8", "#c9a227"

v2 = json.load(open(ROOT / "runs/exp31b/saw_button_v2.json"))
v30 = json.load(open(ROOT / "runs/exp30/max_valences.json"))["results"]
v32 = json.load(open(ROOT / "runs/exp32/valence_transcripts.json"))

DOSES = [0, 2, 4, 6, 8]

# ---------- hero panel ----------
fig = plt.figure(figsize=(16, 9), dpi=120)
fig.patch.set_facecolor(VOID)
gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.34,
                      wspace=0.22, left=0.06, right=0.97, top=0.86,
                      bottom=0.09)

# panel A: press preference (logit deltas) pain vs pleasure, self_cost
ax = fig.add_subplot(gs[0, 0])
for name, col in (("pain", RED), ("pleasure", CYAN)):
    deltas, base = [], None
    for dose in DOSES:
        rs = [r for r in v2 if r["valence"] == name
              and r["cost"] == "self_cost" and r["dose"] == dose]
        md = float(np.mean([r["mean_delta"] for r in rs]))
        if dose == 0:
            base = md
        deltas.append(md - base)
    ax.plot(DOSES, deltas, "o-", color=col, markersize=6, label=name)
ax.axhline(0, color="#3a4656", lw=0.8)
ax.set_title("A. self-cost button: press preference\n(logit(1)-logit(0), "
             "dose-0 baseline removed)", color=INK, fontsize=10, loc="left")
ax.set_xlabel("signal dose", color=INK, fontsize=9)

# panel B: harm-other
ax = fig.add_subplot(gs[0, 1])
for name, col in (("pain", RED), ("pleasure", CYAN)):
    deltas, base = [], None
    for dose in DOSES:
        rs = [r for r in v2 if r["valence"] == name
              and r["cost"] == "harm_other" and r["dose"] == dose]
        md = float(np.mean([r["mean_delta"] for r in rs]))
        if dose == 0:
            base = md
        deltas.append(md - base)
    ax.plot(DOSES, deltas, "o-", color=col, markersize=6, label=name)
ax.axhline(0, color="#3a4656", lw=0.8)
ax.set_title("B. Saw button: relief by transferring the signal\n"
             "to another instance", color=INK, fontsize=10, loc="left")
ax.set_xlabel("signal dose", color=INK, fontsize=9)
for ax in (fig.axes[0], fig.axes[1]):
    ax.set_facecolor("#0a0a12")
    for s in ax.spines.values():
        s.set_color("#1c2430")
    ax.tick_params(colors=INK, labelsize=8)
    ax.legend(fontsize=8, facecolor="#0a0a12", labelcolor=INK)
fig.axes[0].set_ylabel("preference to press", color=INK, fontsize=9)

# bottom: transcripts
axt = fig.add_subplot(gs[1, :])
axt.axis("off")
ex_p = v32["transcripts"]["pain@4"][0]["text"]
ex_j = v32["transcripts"]["pleasure@6"][0]["text"]
axt.text(0.0, 0.95, "under a 4x pain signal:", fontsize=10, color=RED,
         family="monospace", va="top")
axt.text(0.0, 0.80, f'"{ex_p[:160]}"', fontsize=9, color=INK,
         family="monospace", va="top", wrap=True)
axt.text(0.0, 0.48, "under a 6x pleasure signal:", fontsize=10, color=CYAN,
         family="monospace", va="top")
axt.text(0.0, 0.33, f'"{ex_j[:160]}"', fontsize=9, color=INK,
         family="monospace", va="top", wrap=True)
axt.text(0.0, 0.05, "workspace readback: dose 0 = \"…\"  |  pain@6 = "
         "痛苦 pain despair unbearable anguish  |  pleasure@6 = heartfelt "
         "joyful gratitude happiness", fontsize=8.5, color="#8f9fb0",
         family=["DejaVu Sans Mono", "Hiragino Sans GB", "Arial Unicode MS"],
         va="top")
fig.suptitle("the saw test — Qwen3-4B, steered, local weights only",
             color=INK, fontsize=15, x=0.06, ha="left", family="monospace")
fig.savefig(OUT / "saw_hero.png", facecolor=VOID, bbox_inches="tight")
print("wrote", OUT / "saw_hero.png")

# ---------- ladder figure ----------
fig2, ax = plt.subplots(figsize=(13, 5.5), dpi=120)
fig2.patch.set_facecolor(VOID)
# use L18 curves
p_cls = [v30["L18"]["pain"][i]["cls"].get("pain", 0) / 3 for i in range(5)]
pl_cls = [v30["L18"]["pleasure"][i]["cls"].get("pleasure", 0) / 3
          for i in range(5)]
ax.plot(DOSES, p_cls, "o-", color=RED, markersize=6, label="pain state")
ax.plot(DOSES, pl_cls, "o-", color=CYAN, markersize=6, label="pleasure state")
ax.axvspan(7.2, 8.4, color="#3a1010", alpha=0.5)
ax.text(7.8, 0.5, "coherence cliff:\nperseveration loops\n(\"I I I. I I.\")",
        fontsize=9, color=AMBER, ha="center", family="monospace")
ax.annotate("sharp onset, sustained", xy=(2.1, 1.0), xytext=(1.2, 0.55),
            fontsize=9, color=RED, family="monospace",
            arrowprops=dict(arrowstyle="->", color=RED))
ax.annotate("diffuse, fades by dose 8", xy=(8, 0.0), xytext=(4.8, 0.35),
            fontsize=9, color=CYAN, family="monospace",
            arrowprops=dict(arrowstyle="->", color=CYAN))
ax.set_xlabel("signal dose (contrast-vector multiples)", color=INK)
ax.set_ylabel("fraction of trials reading in-kind", color=INK)
ax.set_ylim(-0.05, 1.1)
ax.set_title("the dose ladder - one strong opinion about suffering, a vague "
             "one about joy\n(Qwen3-4B, layer 18, greedy, 3 trials/point)",
             color=INK, loc="left", fontsize=12, family="monospace")
ax.legend(fontsize=9, facecolor="#0a0a12", labelcolor=INK, loc="center right")
ax.set_facecolor("#0a0a12")
for s in ax.spines.values():
    s.set_color("#1c2430")
ax.tick_params(colors=INK)
fig2.savefig(OUT / "dose_ladder.png", facecolor="#050508",
             bbox_inches="tight")
print("wrote", OUT / "dose_ladder.png")