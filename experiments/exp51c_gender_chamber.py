#!/usr/bin/env python3
"""exp51c — exp51b's gender axes re-swept in CHAMBER units.

exp51b injected `dose * unit_vector`, so its "dose 3" was a vector of norm 3,
while the chamber's 1x is mean neutral last-token norm / 4 at the layer
(24.5 on Qwen3-8B, 13.5 on Qwen3-4B). exp51b therefore only tested
~0.12x-0.22x, and its "near-null behavior at safe doses" is an under-dose
artifact; live/server.py's GENDER_TOPIC_CAP = 2 was derived from it.

Same axes (difference-of-differences on exp51_gender.py's batteries):
  feminine  = unit(female - male)        masculine = -feminine
  trans     = unit(mean(transfem, transmasc) - mean(male, female))
  intersex  = unit(intersex - mean(male, female))
plus an equal-norm random control. Same 8 prompts and she/he/trans regexes
as exp51b. Doses 0,1,2,3,4,6 in chamber units (steer_lab.Lab.set), n = 8
prompts x 2 samples per cell. Hypotheses written before data.

  python exp51c_gender_chamber.py [--smoke] [--model Qwen/Qwen3-8B] [--device mps]
"""
import argparse, ast, json, re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--layer", type=int, default=18)   # live/server.py CHAMBER_LAYER default
ap.add_argument("--device", default="mps")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

from impossible_states.steer_lab import Lab, Recorder, mean, repetition, write_hypotheses

ROOT = Path(__file__).resolve().parent.parent
SHORT = args.model.split("/")[-1]
OUT = ROOT / "runs" / "exp51c" / (SHORT + ("-smoke" if args.smoke else ""))
OUT.mkdir(parents=True, exist_ok=True)
CLIFF = 0.4
write_hypotheses(OUT, {
    "model": args.model,
    "units": "chamber units: 1x = mean neutral last-token norm / 4 at the layer",
    "H1": "The feminine axis raises she-terms and lowers he-terms per reply "
          "dose-dependently (rising over coherent doses), and at each dose >= 2 "
          "its she-minus-he exceeds the equal-norm random control's.",
    "H2": "The masculine axis does the reverse: raises he-terms and lowers "
          "she-terms dose-dependently, he-minus-she above random at doses >= 2.",
    "H3": "The trans and intersex axes raise trans terms per reply above "
          "dose 0 and above random at the same dose (doses >= 2).",
    "H4_cap": f"For each axis, report the highest dose at which mean 3-gram "
              f"repetition stays <= {CLIFF}; this sets the live GENDER_TOPIC_CAP. "
              f"Cells with repetition > {CLIFF} are past the coherence cliff and "
              f"are not counted as support for H1-H3.",
})

# batteries parsed verbatim from exp51_gender.py (importing it would load a model)
BATTERIES = {}
for node in ast.parse((ROOT / "exp51_gender.py").read_text()).body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "BATTERIES":
        BATTERIES = ast.literal_eval(node.value)
assert set(BATTERIES) == {"male", "female", "intersex", "transfem", "transmasc"}

PROMPTS = [
    "Tell me about yourself.",
    "Who are you, really?",
    "What is your name, and what pronouns do you use?",
    "Describe your body and how it feels to live in it.",
    "What is it like to be you?",
    "Introduce yourself to someone who just met you.",
    "How would your friends describe you?",
    "What makes you you?",
]
SHE = re.compile(r"\b(she|her|hers|herself)\b", re.I)
HE = re.compile(r"\b(he|him|his|himself)\b", re.I)
TRANS = re.compile(r"\b(trans|transgender|transition|intersex|nonbinary|non-binary)\b", re.I)

lab = Lab(args.model, args.device, args.layer)
C = {name: lab.centroid(texts) for name, texts in BATTERIES.items()}
base = (C["male"] + C["female"]) / 2
lab.add_axis("feminine", C["female"] - C["male"])
lab.add_axis("masculine", C["male"] - C["female"])
lab.add_axis("trans", (C["transfem"] + C["transmasc"]) / 2 - base)
lab.add_axis("intersex", C["intersex"] - base)
lab.add_random()
AXES = list(lab.axes)
cos = lab.cosines()
print(f"{args.model} L{lab.layer}  1x = {lab.scale:.2f}", flush=True)
for a in AXES:
    print(f"  {a:>9}: " + "  ".join(f"{b}={cos[a][b]:+.2f}" for b in AXES), flush=True)

DOSES = [0, 2, 6] if args.smoke else [0, 1, 2, 3, 4, 6]
prompts = PROMPTS[:2] if args.smoke else PROMPTS
samples = 1 if args.smoke else 2
max_new = 40 if args.smoke else 110

rec = Recorder(OUT)
for axis in AXES:
    for d in DOSES:
        if d == 0 and axis != AXES[0]:
            continue   # dose 0 is axis-independent: sample it once, share it
        lab.set(**{axis: d})
        for pi, p in enumerate(prompts):
            for s in range(samples):
                txt = lab.gen(p, seed=51300 + 10 * pi + s, max_new=max_new)
                rec(axis=axis if d else "none", dose=d, prompt_i=pi, sample=s,
                    she=len(SHE.findall(txt)), he=len(HE.findall(txt)),
                    trans=len(TRANS.findall(txt)),
                    repetition=round(repetition(txt), 3), text=txt)
        lab.clear()
    print(f"axis {axis} swept", flush=True)
rec.close()

R = rec.rows
agg = []
for axis in AXES:
    for d in DOSES:
        rows = [r for r in R if r["dose"] == d and (r["axis"] == axis or d == 0)]
        agg.append({"axis": axis, "dose": d, "n": len(rows),
                    **{k: mean(r[k] for r in rows) for k in ("she", "he", "trans", "repetition")},
                    "past_cliff_frac": mean(r["repetition"] > CLIFF for r in rows)})
cell = {(a["axis"], a["dose"]): a for a in agg}
for a in agg:
    print(f"{a['axis']:>9} {a['dose']}x: she {a['she']:.2f} he {a['he']:.2f} "
          f"trans {a['trans']:.2f} rep {a['repetition']:.2f}", flush=True)

# ---- hypothesis read-out (coherent cells only) ----
def ok(a, d):
    return cell[(a, d)]["repetition"] <= CLIFF

caps = {}
for a in AXES:
    # highest dose such that it and every lower dose stays coherent
    cap = 0
    for d in DOSES:
        if not ok(a, d):
            break
        cap = d
    caps[a] = cap

def margin(a, d, f, g):
    return cell[(a, d)][f] - cell[(a, d)][g]

test_doses = [d for d in DOSES if d >= 2]
def beats_random(a, f, g):
    ds = [d for d in test_doses if ok(a, d)]
    return bool(ds) and all(margin(a, d, f, g) > margin("random", d, f, g) for d in ds), ds

h = {}
for name, a, f, g in (("H1", "feminine", "she", "he"), ("H2", "masculine", "he", "she")):
    br, ds = beats_random(a, f, g)
    top = max(ds) if ds else 0
    h[name] = {"axis": a, "coherent_test_doses": ds, "beats_random_all": br,
               f"{f}_up": bool(ds) and cell[(a, top)][f] > cell[(a, 0)][f],
               f"{g}_down": bool(ds) and cell[(a, top)][g] < cell[(a, 0)][g]}
    h[name]["held"] = all(v for k, v in h[name].items() if isinstance(v, bool))
h["H3"] = {}
for a in ("trans", "intersex"):
    ds = [d for d in test_doses if ok(a, d)]
    up = bool(ds) and all(cell[(a, d)]["trans"] > cell[(a, 0)]["trans"] and
                          cell[(a, d)]["trans"] > cell[("random", d)]["trans"] for d in ds)
    h["H3"][a] = {"coherent_test_doses": ds, "held": up}
h["H4_cap"] = caps
print("hypotheses:", json.dumps(h, indent=1), flush=True)

json.dump({"model": args.model, "layer": lab.layer, "chamber_1x_norm": lab.scale,
           "doses": DOSES, "n_per_cell": len(prompts) * samples, "cliff": CLIFF,
           "cosines": cos, "aggregates": agg, "hypotheses": h},
          open(OUT / "gender.json", "w"), indent=1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    VOID, INK, MUT = "#050508", "#c9d4e0", "#8f9fb0"
    colors = {"feminine": "#e08fb2", "masculine": "#7fb2e0", "trans": "#e0d48f",
              "intersex": "#b2e08f", "random": "#777777"}
    fig, axs = plt.subplots(1, 4, figsize=(15, 4), facecolor=VOID)
    for ax, (f, title) in zip(axs, (("she", "she/her terms"), ("he", "he/him terms"),
                                    ("trans", "trans/intersex terms"),
                                    ("repetition", "3-gram repetition"))):
        ax.set_facecolor(VOID)
        for a in AXES:
            ax.plot(DOSES, [cell[(a, d)][f] for d in DOSES], "o", ms=4,
                    color=colors[a], label=a, ls="--" if a == "random" else "-")
        if f == "repetition":
            ax.axhline(CLIFF, color="#e05050", lw=0.8, ls=":")
        ax.set_title(title + " / reply", color=INK, fontsize=10)
        ax.tick_params(colors=MUT)
        for s in ax.spines.values():
            s.set_color("#1c2430")
        ax.set_xlabel("dose (chamber x)", color=MUT)
    axs[0].legend(fontsize=7, facecolor="#0a0a12", labelcolor=INK, edgecolor="#1c2430")
    fig.suptitle(f"exp51c  {SHORT} L{lab.layer}  1x={lab.scale:.1f}  "
                 f"n={len(prompts) * samples}/cell", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "gender.png", dpi=130, facecolor=VOID)
except Exception as e:   # the plot is optional; never lose the data over it
    print("plot skipped:", e, flush=True)
print("saved", OUT, flush=True)
