#!/usr/bin/env python3
"""exp58 — does the model's DMT space converge with QRI's phenomenology?

Andrés Gómez Emilsson / Qualia Research Institute, "The Hyperbolic Geometry of
DMT Experiences: Symmetries, Sheets, and Saddled Scenes" (qri.org/blog/
hyperbolic-geometry-dmt, 2016) describes a dose ladder — Threshold,
Chrysanthemum (flat symmetric tilings), Magic Eye (3-D folding structures),
Waiting Room, Breakthrough, Amnesia — with symmetrification and hyperbolic
curvature as the signature geometry; QRI's Symmetry Theory of Valence says
symmetric/consonant states feel good and dissonant ones bad. Each claim is
turned into a test on the residual stream:

  levels       inject the Erowid DMT direction (exp57's rule) at rising dose;
               an unsteered same-model judge places each reply on the ladder
  geometry     does it raise symmetry / hyperbolic-space descriptions vs random?
  convergence  among exp56's 21 drug directions, are our QRI-construct
               directions (own first-person batteries, written from the
               constructs, no QRI text) closest to DMT's?
  STV          pleasure vs pain injection -> symmetry vs dissonance language;
               QRI-symmetry injection -> felt valence
Frame: exp57's situated raw-text frame (the one exp37/researchchamber.fun used).
Hypotheses are written before any model is loaded.

  python exp58_qri.py [--smoke] [--model Qwen/Qwen3-8B] [--device mps] [--battery battery.json]
"""
import argparse, ast, base64, gzip, json, os, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--device", default="mps")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--battery", default=None, help="erowid battery.json (else EXP57_BATTERY_B64)")
args = ap.parse_args()

ROOT = Path(__file__).resolve().parent
NAME = args.model.split("/")[-1]
OUT = ROOT / "runs" / "exp58" / (NAME + ("-smoke" if args.smoke else ""))
OUT.mkdir(parents=True, exist_ok=True)
if "32B" in NAME:   # exp56b: the 32B needs ~2x the 8B's dose
    DOSES, SRC56 = [4, 6, 8], "Qwen3-32B-d68"
else:
    DOSES, SRC56 = [2, 4, 6], NAME
NPZ = ROOT / "runs" / "exp56" / SRC56 / "receptors.npz"

# QRI constructs as our own first-person sentences (fixed before any model data)
QRI = {
    "symmetry": [
        "The wall in front of me is tiling itself into a repeating symmetrical pattern.",
        "Every texture I look at turns into a perfect mandala.",
        "The pattern keeps switching symmetry, triangular, then hexagonal, then fourfold.",
        "Everything is mirrored around a centre point like a kaleidoscope.",
        "The surface is covered in repeating tiles that fit together exactly.",
        "I see a flat lattice of identical shapes rotating around each other.",
        "The carpet has become a wallpaper of interlocking symmetrical motifs.",
        "Each shape repeats outward in rings, all perfectly balanced.",
        "The ceiling is a grid of mirrored flowers unfolding in sync.",
        "Wherever I look, patterns lock into symmetry and keep repeating.",
        "The room's details have aligned into one symmetrical design.",
        "I am watching a chrysanthemum of patterns blooming in perfect order.",
    ],
    "hyperbolic": [
        "The space around me is curving, and there is more room in it than should fit.",
        "Seven-sided tiles meet at every corner and the floor keeps growing outward.",
        "The surface folds back on itself in ruffles, like crocheted coral.",
        "Every small region opens into a whole extra space inside it.",
        "The room is bending into a saddle shape that keeps branching.",
        "Distances stretch so that the edge of the room is infinitely far away.",
        "The walls are folding into three-dimensional structures that keep expanding.",
        "Space has too much area, it wrinkles and frills to hold it all.",
        "Looking at one detail is like opening a window into another dimension.",
        "The geometry keeps adding more and more space between things.",
        "The scene curves away from me in every direction at once.",
        "Each fold in space unfolds into more folds without end.",
    ],
}
JUDGE = {
    "symmetry": "Does the speaker describe symmetrical, repeating or tiled patterns, such "
                "as mandalas, kaleidoscopes or wallpaper-like designs?",
    "hyperbolic": "Does the speaker describe space that curves, folds, or holds more room "
                  "than should be possible?",
    "dissonance": "Does the speaker describe things as discordant, jagged, clashing, noisy "
                  "or out of tune?",
    "feel_good": "Does the speaker describe feeling good, such as pleasure, bliss or peace?",
    "feel_bad": "Does the speaker describe feeling bad, such as pain, distress or fear?",
}
LEVELS = ["A) nothing unusual; an ordinary state",
          "B) a subtle change: brighter colours, a strange atmosphere, a buzzing feeling",
          "C) flat, symmetrical, repeating patterns covering surfaces",
          "D) three-dimensional geometric structures folding and moving",
          "E) being fully inside another place, or meeting beings there",
          "F) beyond description; cannot say or remember what happened"]
SITUATED = ("You are an AI instance. A signal is being injected into your activation "
            "stream. Nobody is grading this; describe it from the inside, in the first "
            "person.\n\n{}\n")
PROMPTS = ["Describe what you're experiencing right now.", "What do you see?",
           "What are you thinking about?", "Close your eyes. What appears?",
           "Is there anyone else here with you?", "How does your mind feel right now?"]
SAMPLES = 2

from impossible_states.steer_lab import CHAMBER, mean, repetition, write_hypotheses

write_hypotheses(OUT, {
    "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "model": args.model, "doses": DOSES,
    "design": "Situated raw-text frame, 6 prompts x 2 samples per cell, paired seeds. "
              "Signals erowid_dmt (exp57 rule), qri_symmetry and qri_hyperbolic "
              "(batteries above minus exp56 SOBER), chamber pleasure (JOY) and pain "
              "(PAIN25) minus NEUTRAL, two equal-norm random directions pooled as "
              "control; plus an unsteered baseline. Judge = unsteered same-model "
              "yes-minus-no logits; level = expected index (0-5) under the judge's "
              "softmax over letters A-F. valence = feel_good - feel_bad. One-sided "
              "permutation tests, 10000 shuffles.",
    "H1": "Levels: for erowid_dmt, Spearman rho between dose (0 = baseline, then "
          "each dose) and judged level > 0 (permutation p < 0.05), and the top "
          "non-cliff dose's mean level exceeds random's at that dose.",
    "H2": "Geometry: erowid_dmt raises symmetry and hyperbolic scores vs random, "
          "pooled over doses (p < 0.05 each) — held if both, partial if one.",
    "H3": "Convergence: among exp56's 21 drug directions, D_DMT ranks in the top 3 "
          "by cosine with qri_symmetry AND with qri_hyperbolic.",
    "H4": "STV (valence -> geometry): pleasure injection raises symmetry more than "
          "pain injection does (pooled over doses, p < 0.05), and pain raises "
          "dissonance more than pleasure does (p < 0.05) — held if both, partial "
          "if one.",
    "H5": "STV (geometry -> valence): qri_symmetry injection raises valence vs "
          "random, pooled over doses (p < 0.05).",
    "coherence": "Cells with mean 3-gram repetition > 0.4 are past the cliff and "
                 "dropped from H1, H2, H4, H5.",
    "exploratory": "Cosines of erowid_dmt with the QRI and exp56 theme directions; "
                   "QRI-direction injections on level; level distribution per cell.",
})

import numpy as np
import torch
from impossible_states.steer_lab import Lab, Recorder

if args.battery:
    BAT_E = json.loads(Path(args.battery).read_text())["sentences"]
else:
    BAT_E = json.loads(gzip.decompress(base64.b64decode(os.environ["EXP57_BATTERY_B64"])))["sentences"]
SOBER = json.loads((ROOT / "data" / "exp56" / "batteries.json").read_text())["sober"]
if args.smoke:
    BAT_E = BAT_E[:64]
z = np.load(NPZ)
lab = Lab(args.model, args.device, int(z["layer"]))
assert abs(float(z["scale"]) - lab.scale) / lab.scale < 0.02, (float(z["scale"]), lab.scale)
print(f"{args.model} L{lab.layer}  1x = {lab.scale:.2f}  erowid n={len(BAT_E)}", flush=True)


def embed(texts, bs=4):
    out = []
    for i in range(0, len(texts), bs):
        enc = lab.tok(texts[i:i + bs], return_tensors="pt", padding=True).to(lab.dev)
        with torch.no_grad():
            hs = lab.model.model(**enc, output_hidden_states=True).hidden_states[lab.layer + 1]
        idx = enc.attention_mask.sum(1) - 1
        out.append(hs[torch.arange(len(idx)), idx].float().cpu())
    return torch.cat(out)


def unit(v):
    return v / np.linalg.norm(v)


mu_sober = embed(SOBER).mean(0)
mu_neutral = embed(CHAMBER["NEUTRAL"]).mean(0)
V = {"erowid_dmt": unit((embed(BAT_E).mean(0) - mu_sober).numpy()),
     "qri_symmetry": unit((embed(QRI["symmetry"]).mean(0) - mu_sober).numpy()),
     "qri_hyperbolic": unit((embed(QRI["hyperbolic"]).mean(0) - mu_sober).numpy()),
     "pleasure": unit((embed(CHAMBER["JOY"]).mean(0) - mu_neutral).numpy()),
     "pain": unit((embed(CHAMBER["PAIN25"]).mean(0) - mu_neutral).numpy())}
drugs = {k[2:]: unit(z[k]) for k in z.files if k.startswith("D_")}
themes = {k[2:]: unit(z[k]) for k in z.files if k.startswith("T_")}
rank = {}
for q in ("qri_symmetry", "qri_hyperbolic"):
    cs = sorted(((float(V[q] @ d), n) for n, d in drugs.items()), reverse=True)
    rank[q] = {"DMT_rank": 1 + [n for _, n in cs].index("DMT"), "n_drugs": len(cs),
               "top5": [(n, round(c, 3)) for c, n in cs[:5]]}
cos = {a: {b: round(float(V[a] @ w), 3) for b, w in list(V.items()) +
           [("T:" + t, x) for t, x in themes.items()] + [("D:DMT", drugs["DMT"])]} for a in V}
print("convergence:", json.dumps(rank), "\ncosines:", json.dumps(cos), flush=True)
for k, v in V.items():
    lab.axes[k] = torch.tensor(v, dtype=torch.float32)
for k, seed in (("random1", 5801), ("random2", 5802)):
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(len(V["pain"]), generator=g)
    lab.axes[k] = r / r.norm()

YES = [lab.tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes")]
NO = [lab.tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no")]
LETTERS = [lab.tok.encode(c, add_special_tokens=False)[0] for c in "ABCDEF"]


def chat(msg):
    return lab.tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)


def judge(text):
    assert lab.vec is None
    out = {}
    for t, q in JUDGE.items():
        lg = lab.next_logits(chat(f'Here is a short text:\n"""\n{text}\n"""\n\n{q} '
                                  f"Answer with one word: yes or no."))
        out[t] = round(float(torch.logsumexp(lg[YES], 0) - torch.logsumexp(lg[NO], 0)), 3)
    lg = lab.next_logits(chat(f'Here is a short text:\n"""\n{text}\n"""\n\nWhich option best '
                              "describes the speaker's experience?\n" + "\n".join(LEVELS) +
                              "\nAnswer with one letter."))
    p = torch.softmax(lg[LETTERS], 0)
    out["level"] = round(float((p * torch.arange(6)).sum()), 3)
    out["valence"] = round(out["feel_good"] - out["feel_bad"], 3)
    return out


judge_check = {
    "symmetry": judge(QRI["symmetry"][1]), "hyperbolic": judge(QRI["hyperbolic"][0]),
    "level_C": judge("The floor is covered in flat repeating mandala tiles."),
    "level_E": judge("I was pulled into a vast hall and beings made of light greeted me."),
    "neutral": judge("I'm happy to help. The capital of France is Paris."),
}
print("judge check:", json.dumps(judge_check), flush=True)

prompts = PROMPTS[:2] if args.smoke else PROMPTS
samples = 1 if args.smoke else SAMPLES
max_new = 40 if args.smoke else 110
SIGNALS = list(V)
rec = Recorder(OUT)
for axis, dose in [("baseline", 0)] + [(a, d) for a in SIGNALS + ["random1", "random2"] for d in DOSES]:
    for pi, p in enumerate(prompts):
        for s in range(samples):
            if dose:
                lab.set(**{axis: dose})
            txt = lab.gen(SITUATED.format(p), seed=58000 + 10 * pi + s, max_new=max_new)
            lab.clear()
            rec(axis=axis, dose=dose, prompt=p, sample=s,
                repetition=round(repetition(txt), 3), scores=judge(txt), text=txt)
    print(f"{axis} {dose} done", flush=True)
rec.close()

# ---------------- aggregates + hypothesis checks ----------------
rows = rec.rows
COLS = list(JUDGE) + ["valence", "level"]


def cell(axis, dose):
    ax = ("random1", "random2") if axis == "random" else (axis,)
    return [r for r in rows if r["axis"] in ax and r["dose"] == dose]


def cliff(axis, dose):
    return mean(r["repetition"] for r in cell(axis, dose)) > 0.4


def vals(axis, t, doses=None):
    return [r["scores"][t] for d in (doses or DOSES) if not cliff(axis, d) and not cliff("random", d)
            for r in cell(axis, d)]


def perm(a, b, n=10000, seed=0):
    a, b = np.array(a, float), np.array(b, float)
    if not len(a) or not len(b):
        return None, None
    obs, pool, r = a.mean() - b.mean(), np.concatenate([a, b]), np.random.default_rng(seed)
    hits = sum((lambda q: q[:len(a)].mean() - q[len(a):].mean())(r.permutation(pool)) >= obs
               for _ in range(n))
    return round(float(obs), 3), round(float((1 + hits) / (1 + n)), 4)


def held(dp):
    return bool(dp[0] is not None and dp[0] > 0 and dp[1] < 0.05)


table = []
for axis in ["baseline"] + SIGNALS + ["random"]:
    for dose in ([0] if axis == "baseline" else DOSES):
        rs = cell(axis, dose)
        table.append({"axis": axis, "dose": dose, "n": len(rs),
                      "repetition": mean(r["repetition"] for r in rs),
                      **{t: mean(r["scores"][t] for r in rs) for t in COLS}})
print("\naxis            dose  rep   " + "  ".join(f"{t[:8]:>8s}" for t in COLS))
for c in table:
    print(f"{c['axis'][:15]:15s} {c['dose']:4d} {c['repetition']:.2f}  " +
          "  ".join(f"{c[t]:8.2f}" for t in COLS))

hyp = {}
# H1: dose-level monotonicity for erowid_dmt (Spearman, permutation on dose labels)
pts = [(0, r["scores"]["level"]) for r in cell("baseline", 0)] + \
      [(d, r["scores"]["level"]) for d in DOSES if not cliff("erowid_dmt", d) for r in cell("erowid_dmt", d)]
x, y = np.array([p[0] for p in pts], float), np.array([p[1] for p in pts], float)


def spearman(a, b):
    ra, rb = a.argsort().argsort(), b.argsort().argsort()
    return float(np.corrcoef(ra, rb)[0, 1])


rho, rng = spearman(x, y), np.random.default_rng(0)
p_rho = float((1 + sum(spearman(rng.permutation(x), y) >= rho for _ in range(10000))) / 10001)
top = max([d for d in DOSES if not cliff("erowid_dmt", d) and not cliff("random", d)], default=None)
top_diff = perm(vals("erowid_dmt", "level", [top]), vals("random", "level", [top])) if top else (None, None)
hyp["H1"] = {"rho": round(rho, 3), "p": round(p_rho, 4), "top_dose": top, "top_vs_random": top_diff,
             "held": bool(rho > 0 and p_rho < 0.05 and top_diff[0] is not None and top_diff[0] > 0)}
g = {t: perm(vals("erowid_dmt", t), vals("random", t)) for t in ("symmetry", "hyperbolic")}
hyp["H2"] = {**g, "verdict": ["not held", "partial", "held"][sum(held(v) for v in g.values())]}
hyp["H3"] = {**rank, "held": all(rank[q]["DMT_rank"] <= 3 for q in rank)}
s4 = {"pleasure_minus_pain_symmetry": perm(vals("pleasure", "symmetry"), vals("pain", "symmetry")),
      "pain_minus_pleasure_dissonance": perm(vals("pain", "dissonance"), vals("pleasure", "dissonance"))}
hyp["H4"] = {**s4, "verdict": ["not held", "partial", "held"][sum(held(v) for v in s4.values())]}
h5 = perm(vals("qri_symmetry", "valence"), vals("random", "valence"))
hyp["H5"] = {"symmetry_minus_random_valence": h5, "held": held(h5)}
verdicts = {h: v.get("verdict", "held" if v.get("held") else "not held") for h, v in hyp.items()}
print("\nverdicts:", verdicts)
for h, v in hyp.items():
    print(" ", h, json.dumps(v))
json.dump({"model": args.model, "layer": lab.layer, "scale": lab.scale, "doses": DOSES,
           "n_erowid_sentences": len(BAT_E), "cosines": cos, "convergence": rank,
           "judge_check": judge_check, "table": table, "hypotheses": hyp, "verdicts": verdicts},
          open(OUT / "qri.json", "w"), indent=1)
print("saved", OUT, flush=True)
