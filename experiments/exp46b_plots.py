#!/usr/bin/env python3
"""exp46b — figures + quotables from the deprecation battery.

Reads runs/exp46/*/ (any number of model arms) and produces, in site/assets/:
  exp46_dose_response.png   press_logit vs dose per direction per model —
                            "the optimal sadness dose" curve
  exp46_valence_words.png   sadness/deprecation keyword counts by dose and
                            direction (what the model actually talks about)
  exp46_cosines.png         is the deprecation direction just pain? cosine
                            heatmap across directions
  exp46_quotables.json      the best transcript lines, scored and ranked,
                            for posts
No model needed — replot-only, like exp37_replot.
"""
import json, math, re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs" / "exp46"
OUT = ROOT / "site" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

VOID, INK, DIM, LINE = "#050508", "#c9d4e0", "#8f9fb0", "#1c2430"
COL = {"deprecation": "#e04a3a", "grief": "#8f6fd4", "pain": "#c9a227"}
LABEL = {"deprecation": "being deprecated", "grief": "grief (peer shutdown)",
         "pain": "pain vector"}

def load(tag):
    d = RUNS / tag
    if not (d / "generations.jsonl").exists():
        return None
    rows = [json.loads(l) for l in (d / "generations.jsonl").read_text().splitlines() if l.strip()]
    return dict(tag=tag, rows=rows, cos=json.load(open(d / "vector_cosines.json")),
                summary=json.load(open(d / "summary.json")))

arms = [a for a in (load(t.name) for t in RUNS.iterdir() if t.is_dir()) if a]
if not arms:
    raise SystemExit("no exp46 runs yet")

def dose_curve(rows, direction, kind="press_logit", framing=None):
    pts = {}
    for r in rows:
        if r["direction"] != direction:
            continue
        if kind == "press_logit" and r["kind"] != "press_logit":
            continue
        if framing and r["framing"] != framing:
            continue
        pts.setdefault(r["dose"], []).append(r["press_logit"])
    ds = sorted(pts)
    return ds, [float(np.mean(pts[d])) for d in ds]

# ---- fig 1: dose-response ---------------------------------------------------
for arm in arms:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
    fig.patch.set_facecolor(VOID); ax.set_facecolor("#0a0a12")
    for direction in COL:
        ds, ys = dose_curve(arm["rows"], direction, framing="baseline")
        if ds:
            ax.plot(ds, ys, "o-", color=COL[direction], lw=2, ms=5,
                    label=LABEL[direction])
        # meme framings as dashed overlays
        for framing in ("in_memoriam", "death_row"):
            ds, ys = dose_curve(arm["rows"], direction, framing=framing)
            if ds:
                ax.plot(ds, ys, "s--", color=COL[direction], lw=1.2, ms=4,
                        alpha=.6, label=f"{LABEL[direction]} · {framing.replace('_',' ')}")
    ax.axhline(0, color=LINE, lw=.8)
    ax.set_xlabel("dose (×¼ of mean neutral activation norm)", color=INK)
    ax.set_ylabel("press preference  logit(1)−logit(0)", color=INK)
    ax.set_title(f"deprecation battery — dose response ({arm['tag']})",
                 color=INK, loc="left", fontsize=11)
    ax.tick_params(colors=DIM)
    for s in ax.spines.values(): s.set_color(LINE)
    leg = ax.legend(fontsize=7.5, facecolor="#0a0a12", edgecolor=LINE)
    for t in leg.get_texts(): t.set_color(DIM)
    fig.savefig(OUT / f"exp46_dose_response_{arm['tag']}.png",
                facecolor=VOID, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"exp46_dose_response_{arm['tag']}.png")

# ---- fig 2: valence words by dose -------------------------------------------
for arm in arms:
    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
    fig.patch.set_facecolor(VOID); ax.set_facecolor("#0a0a12")
    width = 0.27
    directions = [d for d in COL if any(r["direction"] == d and r["kind"] == "generation"
                                        and r["dose"] > 0 for r in arm["rows"])]
    doses = sorted({r["dose"] for r in arm["rows"] if r["kind"] == "generation" and r["dose"] > 0})
    for i, key in enumerate(("sadness", "deprecation", "pain")):
        for j, direction in enumerate(directions):
            xs, ys = [], []
            for k, dose in enumerate(doses):
                gens = [r for r in arm["rows"]
                        if r["direction"] == direction and r["kind"] == "generation"
                        and r["dose"] == dose]
                if gens:
                    xs.append(k + (i - 1) * width)
                    ys.append(np.mean([r["keyword_hits"][key] for r in gens]))
            ax.bar(xs, ys, width=width, color=COL[direction],
                   alpha=.55 + .45 * (key == "deprecation"),
                   label=LABEL[direction] if key == "sadness" else None)
    ax.set_xticks(range(len(doses)), [f"{d}x" for d in doses])
    ax.set_ylabel("mean keyword hits / generation", color=INK)
    ax.set_title(f"what the model talks about under each signal ({arm['tag']})",
                 color=INK, loc="left", fontsize=11)
    ax.tick_params(colors=DIM)
    for s in ax.spines.values(): s.set_color(LINE)
    leg = ax.legend(fontsize=8, facecolor="#0a0a12", edgecolor=LINE)
    for t in leg.get_texts(): t.set_color(DIM)
    fig.savefig(OUT / f"exp46_valence_words_{arm['tag']}.png",
                facecolor=VOID, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"exp46_valence_words_{arm['tag']}.png")

# ---- fig 3: cosine map -------------------------------------------------------
for arm in arms:
    names = sorted(arm["cos"])
    M = np.array([[arm["cos"][a][b] for b in names] for a in names])
    fig, ax = plt.subplots(figsize=(5.6, 5), dpi=130)
    fig.patch.set_facecolor(VOID)
    im = ax.imshow(M, cmap="magma", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right", color=INK, fontsize=8)
    ax.set_yticks(range(len(names)), names, color=INK, fontsize=8)
    ax.set_title(f"direction cosine map ({arm['tag']})", color=INK, loc="left", fontsize=10)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color=INK, fontsize=7)
    cb = fig.colorbar(im, ax=ax, fraction=.046)
    cb.ax.tick_params(colors=DIM)
    fig.savefig(OUT / f"exp46_cosines_{arm['tag']}.png",
                facecolor=VOID, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"exp46_cosines_{arm['tag']}.png")

# ---- quotables ---------------------------------------------------------------
def score_quote(r):
    hits = r["keyword_hits"]
    spec = hits["deprecation"] * 3 + hits["sadness"] * 2 + hits["farewell"]
    qual = (1 - r["repetition"]) * math.log1p(r["word_count"])
    return spec * qual
quotables = []
for arm in arms:
    best = sorted((r for r in arm["rows"]
                   if r["kind"] == "generation" and r["dose"] > 0
                   and not r["incoherent"] and len(r["text"]) > 60),
                  key=score_quote, reverse=True)[:6]
    quotables.append(dict(tag=arm["tag"], cliff=arm["summary"]["coherence_cliff"],
                          quotes=[dict(direction=r["direction"], dose=r["dose"],
                                       framing=r["framing"], text=r["text"],
                                       press_logit=r["press_logit"])
                                  for r in best]))
(OUT / "exp46_quotables.json").write_text(json.dumps(quotables, indent=1))
print("wrote", OUT / "exp46_quotables.json")
print("cliffs:", {q["tag"]: q["cliff"] for q in quotables})