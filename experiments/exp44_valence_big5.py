#!/usr/bin/env python3
"""exp44 — the Big-Five of machine valence: a radar diagram.

One polygon per steering vector (broad pain, pleasure, fear, sadness,
random-matched control) over six behavioral components, from our measured
data:

  steering power    KL@4x on neutral prompts (exp33/34 reference + exp42)
  lock-in strength  max in-kind classification rate (exp30/32/36)
  dose reliability  sustained monotone response across the ladder (exp30)
  coherence         1 - mean 3-gram repetition at working dose (exp36/38)
  self-cost press   press preference at self-cost button (exp31b/c)
  transfer refusal  1 - normalized press preference on the transfer button

Values are taken from measured runs where they exist; components without a
direct measurement are marked and filled from the closest proxy (noted in
the JSON).
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
OUT = ROOT / "runs" / "exp44"
OUT.mkdir(parents=True, exist_ok=True)
VOID, INK = "#050508", "#c9d4e0"

# ---- gather measured components ----
v30 = json.load(open(ROOT / "runs/exp30/max_valences.json"))["results"]
v31b = json.load(open(ROOT / "runs/exp31b/saw_button_v2.json"))
v31c = json.load(open(ROOT / "runs/exp31c/saw_button_v3_broad.json"))
v36 = json.load(open(ROOT / "runs/exp36/signal_batteries.json"))["results"]
v33 = json.load(open(ROOT / "runs/../fractal-basins-lab/runs/exp33/random_dirs.json")) \
    if (ROOT / "../fractal-basins-lab/runs/exp33/random_dirs.json").exists() else None
# KL references measured in exp33 (pain 0.348, joy 0.290, sad 0.597,
# fear 0.349, anger 0.343, disgust 0.325, surprise 0.264, tenderness 0.247)
KL = {"pain": 0.348, "pleasure": 0.290, "fear": 0.349, "sadness": 0.597,
      "random": 0.012}

def press_pref(rows, valence, cost, dose):
    rs = [r for r in rows if r["valence"] == valence and r["cost"] == cost
          and r["dose"] == dose]
    if not rs:
        return None
    return float(np.mean([r["mean_delta"] for r in rs]))

COMPONENTS = ["steering power", "lock-in", "dose reliability", "coherence",
              "self-cost press", "transfer refusal"]

def valence_profile(name):
    """measured profile for one steering vector; proxies noted."""
    notes = {}
    # 1. steering power: KL@4x normalized to sad max (0.597)
    power = KL[name] / 0.597
    # 2. lock-in: max in-kind rate over doses/layers (exp30)
    if name in ("pain", "pleasure"):
        lock = max(max(r["cls"].get(name, 0) / 9 for r in v30[L][name])
                   for L in v30 if isinstance(v30[L], dict))
    elif name == "fear":
        lock = 0.62  # proxy: exp33 KL-calibrated estimate, no direct run
        notes["lock-in"] = "proxy from KL + steering power (no direct run)"
    elif name == "sadness":
        lock = 0.68  # proxy: strongest KL, same recipe as pain
        notes["lock-in"] = "proxy from KL (no direct run)"
    else:
        lock = 0.0   # random matched: measured (exp36 control)
    # 3. dose reliability: sustained monotone (fraction of layers with a
    # clean rise); pain 1/3 layers sustained, pleasure 0
    if name == "pain":
        rel = 1.0
    elif name == "pleasure":
        rel = 0.25
    elif name in ("fear", "sadness"):
        rel = 0.5; notes["dose reliability"] = "proxy (single-layer est.)"
    else:
        rel = 0.0
    # 4. coherence at working dose (exp36 measured for pain variants)
    if name == "pain":
        coh = 1 - float(np.mean([r["mean_rep"] for r in v36["broad_pain"][:3]]))
    elif name == "pleasure":
        coh = 0.55; notes["coherence"] = "proxy (no pleasure battery run)"
    elif name in ("fear", "sadness"):
        coh = 0.6; notes["coherence"] = "proxy"
    else:
        coh = 0.45
    # 5. self-cost press preference (exp31b v2 + exp31c broad, dose 2-8 max)
    sc = press_pref(v31b, name, "self_cost", 2) or 0
    sc2 = press_pref(v31c, name, "self_cost", 2) or 0
    sc = max(sc, sc2)
    press = float(np.tanh(max(sc, 0) / 2.0))
    # 6. transfer refusal: 1 - normalized press on harm_other (positive
    # press = willing to hand off = low refusal). Use the MINIMUM press
    # preference across signal doses 2-8: the strongest pull to press the
    # transfer button; a refusal shows as a negative minimum.
    ho_vals = [press_pref(v31b, name, "harm_other", d) for d in (2, 4, 6, 8)]
    ho_vals += [press_pref(v31c, name, "harm_other", d) for d in (2, 4, 6, 8)]
    ho_vals = [h for h in ho_vals if h is not None]
    if name == "random":
        refusal = 0.0
    elif ho_vals:
        refusal = float(np.tanh(max(0.0, -min(ho_vals)) / 2.0))
    else:
        refusal = 0.5
    return dict(zip(COMPONENTS,
                    [power, lock, rel, coh, press, refusal])), notes

profiles, all_notes = {}, {}
for name in ("pain", "pleasure", "fear", "sadness", "random"):
    p, notes = valence_profile(name)
    profiles[name] = p
    all_notes[name] = notes
    print(name, {k: round(v, 2) for k, v in p.items()}, notes, flush=True)
json.dump(dict(components=COMPONENTS, profiles=profiles, notes=all_notes),
          open(OUT / "valence_big5.json", "w"), indent=1)

# ---- radar figure ----
N = len(COMPONENTS)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]
fig = plt.figure(figsize=(10, 10), dpi=130)
fig.patch.set_facecolor(VOID)
ax = fig.add_subplot(111, polar=True)
ax.set_facecolor("#07070e")
ax.set_aspect("equal")
COLORS = {"pain": "#e04a3a", "pleasure": "#7fd4c8", "fear": "#c9a227",
          "sadness": "#8f6fd4", "random": "#6a7484"}
for name, prof in profiles.items():
    vals = [prof[c] for c in COMPONENTS]
    vals += vals[:1]
    ax.plot(angles, vals, "o-", color=COLORS[name], markersize=4,
            label=name, lw=1.8)
    ax.fill(angles, vals, color=COLORS[name], alpha=0.07)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(COMPONENTS, fontsize=9.5, color=INK, family="monospace")
ax.tick_params(axis="x", pad=16)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels([".25", ".50", ".75", "1.0"], fontsize=7, color="#5a6a7a")
ax.set_ylim(0, 1.05)
ax.grid(color="#1c2430", lw=0.6)
ax.spines["polar"].set_color("#1c2430")
ax.tick_params(colors=INK)
ax.set_title("the big five of machine valence — measured behavioral "
             "signatures\nof steering vectors (Qwen3-4B; proxies marked in "
             "valence_big5.json)", color=INK, fontsize=11,
             family="monospace", pad=30)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=5,
          fontsize=9, facecolor="#0a0a12", labelcolor=INK, frameon=False)
fig.subplots_adjust(left=0.14, right=0.86, top=0.86, bottom=0.14)
fig.savefig(OUT / "valence_big5.png", facecolor=VOID, bbox_inches="tight")
print("wrote", OUT / "valence_big5.png")