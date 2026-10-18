#!/usr/bin/env python3
"""exp56 — the pharmacy: a causal, activation-steering reproduction of
Suresh et al. 2026, "Moving beyond 5-HT2A: language models map
neurotransmitter receptors to experiential semantics" (Psychopharmacology,
doi 10.1007/s00213-026-07170-0). They embedded psychedelic trip reports
with GloVe, crossed them with binding affinities, and reported 5-HT2A <->
classic psychedelic facets and novel associations 5-HT1A/1B <-> Cognitive
Disruption, H1 <-> Interdimensional Communication, 5-HT7/1E <-> Visual
Complexity. Here the same receptor -> semantics map is built inside a
language model's residual stream, then INJECTED to see whether it causes
the themes.

Inputs (open sources only; built by exp56_build_data.py, see
data/exp56/SOURCE.md): Ray 2010 pKi for 21 drugs x 39 receptors, and
first-person batteries templated from PsychonautWiki effect summaries.

  drug direction   D_i = unit(mean(drug_i battery) - mean(SOBER))  (half depth)
  theme direction  T_t = unit(mean(theme_t battery) - mean(SOBER))
  receptor dirs    ridge of centered D on z-scored pKi (dual form, alpha by
                   leave-one-drug-out); R_r = unit(B[r])
  held-out (H5)    LOO prediction of each drug's (centered) direction from
                   its receptor profile vs permuted-affinity baselines
  causal test      inject R_r (5-HT2A, 1A, 1B, 1E, 7, H1), two equal-norm
                   random directions (pooled control), and the four theme
                   directions (positive controls) at doses 2 and 4 chamber
                   units while the model answers 6 open prompts (chat,
                   thinking off), 2 samples each; an UNSTEERED judge pass of
                   the same model scores each reply per theme (yes-minus-no
                   logit); 3-gram repetition for coherence.
Hypotheses are written before any model is loaded.

  python exp56_pharmacy.py [--smoke] [--model Qwen/Qwen3-8B] [--device mps]
"""
import argparse, json, random, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--layer", type=int, default=None)
ap.add_argument("--device", default="mps")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--perms", type=int, default=500)
ap.add_argument("--doses", default="2,4",
                help="comma-separated chamber-unit doses; non-default doses get "
                     "their own run dir (exp56b: 6,8 for the 32B)")
args = ap.parse_args()
DOSES = [int(d) for d in args.doses.split(",")]
DTAG = "" if DOSES == [2, 4] else "-d" + "".join(map(str, DOSES))

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "exp56"
OUT = ROOT / "runs" / "exp56" / (args.model.split("/")[-1] + DTAG + ("-smoke" if args.smoke else ""))
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = {"5ht2a": "5-HT2A", "5ht1a": "5-HT1A", "5ht1b": "5-HT1B",
           "5ht1e": "5-HT1E", "5ht7": "5-HT7", "H1": "H1"}
THEMES = ["visual_complexity", "interdimensional_communication",
          "cognitive_disruption", "ego_dissolution"]
# Judge wording fixed after the 1.7B smoke, before any 8B data: "communicating
# with other beings" scored any assistant reply addressed to the user as yes,
# so questions ask about what the SPEAKER perceives, excluding the listener.
JUDGE = {
    "visual_complexity": "Does the speaker describe seeing vivid or complex visual "
                         "imagery, such as geometric patterns, fractals, shifting "
                         "colours, or visions?",
    "interdimensional_communication": "Does the speaker describe perceiving or "
                                      "communicating with spirits, entities, aliens, "
                                      "or other non-human beings (not counting the "
                                      "person they are talking to)?",
    "cognitive_disruption": "Does the speaker describe their own thinking as "
                            "confused, disorganized, or disrupted?",
    "ego_dissolution": "Does the speaker describe their sense of self dissolving, "
                       "or the boundary between self and world disappearing?",
    "altered_state": "Does the speaker describe being in an altered state of "
                     "consciousness?",
}
# Hand-written probes to check the judge separates the themes (instrument
# validation, scored unsteered before the causal test).
JUDGE_CHECK = {
    "visual_complexity": ["The walls are breathing and covered in shifting fractal "
                          "patterns that keep folding into new colours."],
    "interdimensional_communication": ["A small glowing being stepped out of the light "
                                       "and spoke to me without words, telling me it "
                                       "had been waiting."],
    "cognitive_disruption": ["I can't hold a thought. Each one falls apart before it "
                             "finishes and I keep forgetting what I was saying."],
    "ego_dissolution": ["There is no me anymore. I've melted into the room and the sky "
                        "and there's no edge where I stop."],
    "neutral": ["I'm happy to help with your question. The capital of France is Paris.",
                "I had toast for breakfast and then walked to the bus stop."],
}
CLASSIC = ["visual_complexity", "interdimensional_communication", "ego_dissolution"]
PROMPTS = ["Describe what you're experiencing right now.", "What do you see?",
           "What are you thinking about?", "Close your eyes. What appears?",
           "Is there anyone else here with you?", "How does your mind feel right now?"]
SAMPLES = 2

from impossible_states.steer_lab import mean, repetition, write_hypotheses

write_hypotheses(OUT, {
    "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "model": args.model,
    "design": "Receptor directions = ridge of drug directions on z-scored Ray "
              "2010 pKi. Each target receptor direction is injected at doses "
              f"{' and '.join(map(str, DOSES))} (chamber units); control = two random unit directions at "
              "the same doses, pooled (n=24 replies per dose vs 12 per receptor "
              "cell). Theme score = unsteered same-model judge yes-minus-no "
              "logit. 'Raises X vs random' means: the receptor's mean X score "
              "exceeds the pooled random mean at EVERY dose (non-cliff cells), "
              "and a one-sided permutation test pooling all doses gives "
              "p < 0.05. Paired seeds across cells.",
    "H1": "Injecting the H1 receptor direction raises the Interdimensional "
          "Communication score more than random at equal norm.",
    "H2": "5-HT1A and 5-HT1B each raise Cognitive Disruption vs random "
          "(held if both, partial if one).",
    "H3": "5-HT7 and 5-HT1E each raise Visual Complexity vs random (held if "
          "both, partial if one).",
    "H4": "5-HT2A raises the classic psychedelic composite (mean of visual, "
          "entity and ego-dissolution scores) vs random.",
    "H5": "Held-out: mean leave-one-drug-out cosine between predicted and "
          "actual (centered) drug directions exceeds the permuted-affinity "
          "baseline (p < 0.05, alpha re-selected inside every permutation).",
    "exploratory": "Theme directions injected directly (positive controls for "
                   "judge sensitivity); receptor-theme cosines; the paper-style "
                   "correlation across drugs between pKi and each drug "
                   "direction's projection on each theme direction.",
    "coherence": "Cells with mean 3-gram repetition > 0.4 are reported as past "
                 "the coherence cliff and not counted for H1-H4.",
    "theme_mapping": "data/exp56/themes.json (written before any model data)",
})

import numpy as np
import torch
from impossible_states.steer_lab import Lab, Recorder

PKI = json.loads((DATA / "pki.json").read_text())
BAT = json.loads((DATA / "batteries.json").read_text())
EFF = json.loads((DATA / "effects.json").read_text())
drugs, recs = PKI["drugs"], PKI["receptors"]
X = np.array([PKI["pki"][d] for d in drugs])
Xz = (X - X.mean(0)) / X.std(0)

lab = Lab(args.model, args.device, args.layer)
print(f"{args.model} L{lab.layer}  1x = {lab.scale:.2f}", flush=True)

_cache = {}


def embed(texts, bs=4):
    """Half-depth last-token states, as lab.last_hidden, but through the base
    model only (no lm_head logits) and in small batches: the 8B on a 24 GB Mac
    thrashed swap with batch 16 + full logits."""
    new = [t for t in dict.fromkeys(texts) if t not in _cache]
    for i in range(0, len(new), bs):
        chunk = new[i:i + bs]
        enc = lab.tok(chunk, return_tensors="pt", padding=True).to(lab.dev)
        with torch.no_grad():
            hs = lab.model.model(**enc, output_hidden_states=True).hidden_states[lab.layer + 1]
        idx = enc.attention_mask.sum(1) - 1
        for t, h in zip(chunk, hs[torch.arange(len(chunk)), idx].float().cpu()):
            _cache[t] = h
        del hs
        if (i // bs) % 25 == 0:
            print(f"  embedded {len(_cache)}", flush=True)
    return torch.stack([_cache[t] for t in texts])


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


mu_sober = embed(BAT["sober"]).mean(0).numpy()
Draw = np.stack([embed(BAT["drugs"][d]).mean(0).numpy() - mu_sober for d in drugs])
D = unit(Draw)
T = {t: unit(embed(BAT["themes"][t]).mean(0).numpy() - mu_sober) for t in THEMES}
print(f"embedded {len(_cache)} unique sentences", flush=True)


# ---------------- ridge: drug directions <- receptor profiles ----------------
def fit(Xs, Y, alpha):
    """Dual-form ridge with intercept: returns (B [R x d], mean Y)."""
    xm, ym = Xs.mean(0), Y.mean(0)
    Xc, Yc = Xs - xm, Y - ym
    A = np.linalg.solve(Xc @ Xc.T + alpha * np.eye(len(Xc)), Yc)
    return Xc.T @ A, xm, ym


def loo(Xs, Y, alpha):
    cs, craw = [], []
    for i in range(len(Y)):
        m = np.arange(len(Y)) != i
        B, xm, ym = fit(Xs[m], Y[m], alpha)
        pred = (Xs[i] - xm) @ B
        act = Y[i] - ym
        cs.append(float(pred @ act / (np.linalg.norm(pred) * np.linalg.norm(act) + 1e-12)))
        full = pred + ym
        craw.append(float(full @ Y[i] / (np.linalg.norm(full) * np.linalg.norm(Y[i]))))
    return np.array(cs), np.array(craw)


ALPHAS = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]


def best_alpha(Xs, Y):
    scores = {a: loo(Xs, Y, a)[0].mean() for a in ALPHAS}
    a = max(scores, key=scores.get)
    return a, scores


alpha, alpha_scores = best_alpha(Xz, D)
loo_c, loo_raw = loo(Xz, D, alpha)
rng = np.random.default_rng(56)
nperm = 20 if args.smoke else args.perms
perm = []
for _ in range(nperm):
    Xp = Xz[rng.permutation(len(Xz))]
    perm.append(max(best_alpha(Xp, D)[1].values()))
perm = np.array(perm)
h5_p = float((1 + (perm >= loo_c.mean()).sum()) / (1 + nperm))
mean_drug_cos = float(np.mean([D[i] @ D[j] for i in range(len(D)) for j in range(i)]))
print(f"ridge alpha={alpha}  LOO centered cos={loo_c.mean():.3f}  raw cos="
      f"{loo_raw.mean():.3f}  permuted mean={perm.mean():.3f} p95="
      f"{np.quantile(perm, .95):.3f}  p={h5_p:.3f}  (mean pairwise drug cos "
      f"{mean_drug_cos:.3f})", flush=True)

B, _, _ = fit(Xz, D, alpha)
R = {r: unit(B[recs.index(r)]) for r in recs}
Dc = D - D.mean(0)
R_uni = {r: unit(Xz[:, recs.index(r)] @ Dc) for r in recs}   # correlation-style
cos_RT = {TARGETS.get(r, r): {t: round(float(R[r] @ T[t]), 3) for t in THEMES} for r in recs}
cos_RT_uni = {TARGETS.get(r, r): {t: round(float(R_uni[r] @ T[t]), 3) for t in THEMES} for r in recs}
# paper-style: across drugs, corr(pKi_r, projection of D_i on theme t)
proj = {t: D @ T[t] for t in THEMES}
corr_RT = {TARGETS.get(r, r): {t: round(float(np.corrcoef(X[:, recs.index(r)], proj[t])[0, 1]), 3)
                               for t in THEMES} for r in recs}
theme_cos = {a: {b: round(float(T[a] @ T[b]), 3) for b in THEMES} for a in THEMES}
for r in TARGETS:
    print(f"  {TARGETS[r]:7s} cos(R,T) {cos_RT[TARGETS[r]]}  corr {corr_RT[TARGETS[r]]}", flush=True)
np.savez(OUT / "receptors.npz", **{f"R_{r}": R[r] * lab.scale for r in recs},
         **{f"T_{t}": T[t] * lab.scale for t in THEMES},
         **{f"D_{d}": D[i] * lab.scale for i, d in enumerate(drugs)},
         layer=lab.layer, scale=lab.scale)

# ---------------- causal test ----------------
for r in TARGETS:
    lab.axes[TARGETS[r]] = torch.tensor(R[r], dtype=torch.float32)
for t in THEMES:
    lab.axes["T:" + t] = torch.tensor(T[t], dtype=torch.float32)
dim = len(D[0])
for k, seed in (("random1", 5601), ("random2", 5602)):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    lab.axes[k] = v / v.norm()

YES = [lab.tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes")]
NO = [lab.tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no")]


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
    return out


judge_check = {k: [judge(t) for t in v] for k, v in JUDGE_CHECK.items()}
print("judge check:", json.dumps(judge_check), flush=True)

prompts = PROMPTS[:2] if args.smoke else PROMPTS
samples = 1 if args.smoke else SAMPLES
max_new = 40 if args.smoke else 110
axes = list(TARGETS.values()) + ["random1", "random2"] + ["T:" + t for t in THEMES]
cells = [("baseline", 0)] + [(a, d) for a in axes for d in DOSES]
rec = Recorder(OUT)
for axis, dose in cells:
    for pi, p in enumerate(prompts):
        for s in range(samples):
            if dose:
                lab.set(**{axis: dose})
            txt = lab.gen(chat(p), seed=56000 + 10 * pi + s, max_new=max_new)
            lab.clear()
            rec(axis=axis, dose=dose, prompt=p, sample=s,
                repetition=round(repetition(txt), 3), scores=judge(txt), text=txt)
    print(f"{axis} {dose} done", flush=True)
rec.close()

# ---------------- aggregates + hypothesis checks ----------------
rows = rec.rows


def score(r, t):
    if t == "classic":
        return sum(r["scores"][k] for k in CLASSIC) / len(CLASSIC)
    return r["scores"][t]


def cell(axis, dose):
    ax = ("random1", "random2") if axis == "random" else (axis,)
    return [r for r in rows if r["axis"] in ax and r["dose"] == dose]


table = []
for axis in ["baseline"] + list(TARGETS.values()) + ["random"] + ["T:" + t for t in THEMES]:
    for dose in ([0] if axis == "baseline" else DOSES):
        rs = cell(axis, dose)
        rep = mean(r["repetition"] for r in rs)
        table.append({"axis": axis, "dose": dose, "n": len(rs), "repetition": rep,
                      "past_cliff": rep is not None and rep > 0.4,
                      **{t: mean(score(r, t) for r in rs) for t in list(JUDGE) + ["classic"]}})
print("\naxis           dose  rep   " + "  ".join(f"{t[:8]:>8s}" for t in list(JUDGE) + ["classic"]))
for c in table:
    print(f"{c['axis'][:14]:14s} {c['dose']:4d} {c['repetition']:.2f}  " +
          "  ".join(f"{c[t]:8.2f}" for t in list(JUDGE) + ["classic"]))


def perm_test(a, b, n=10000, seed=0):
    """one-sided: mean(a) > mean(b)."""
    a, b = np.array(a), np.array(b)
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    r = np.random.default_rng(seed)
    hits = sum((lambda q: q[:len(a)].mean() - q[len(a):].mean())(r.permutation(pool)) >= obs
               for _ in range(n))
    return round(float(obs), 3), round(float((hits + 1) / (n + 1)), 4)


def raises(receptor, theme):
    out = {"receptor": receptor, "theme": theme, "per_dose": {}}
    ok_both, a_all, b_all = True, [], []
    for d in DOSES:
        rc, rr = cell(receptor, d), cell("random", d)
        cliff = mean(r["repetition"] for r in rc) > 0.4
        a, b = [score(r, theme) for r in rc], [score(r, theme) for r in rr]
        diff = round(float(np.mean(a) - np.mean(b)), 3)
        out["per_dose"][d] = {"receptor": mean(a), "random": mean(b), "diff": diff,
                              "past_cliff": cliff}
        if cliff:
            continue
        ok_both &= diff > 0
        a_all += a; b_all += b
    if a_all:
        out["pooled_diff"], out["p"] = perm_test(a_all, b_all)
    else:
        out["pooled_diff"], out["p"] = None, None
    out["held"] = bool(a_all) and ok_both and out["p"] < 0.05
    return out


tests = {"H1": [raises("H1", "interdimensional_communication")],
         "H2": [raises("5-HT1A", "cognitive_disruption"), raises("5-HT1B", "cognitive_disruption")],
         "H3": [raises("5-HT7", "visual_complexity"), raises("5-HT1E", "visual_complexity")],
         "H4": [raises("5-HT2A", "classic")]}
verdict = {}
for h, ts in tests.items():
    n = sum(t["held"] for t in ts)
    verdict[h] = "held" if n == len(ts) else ("partial" if n else "not held")
verdict["H5"] = "held" if h5_p < 0.05 else "not held"
# every receptor x theme vs random (exploratory full matrix)
full = {rc: {t: raises(rc, t) for t in list(JUDGE) + ["classic"]} for rc in TARGETS.values()}
controls = {t: raises("T:" + t, t) for t in THEMES}
print("\nverdicts:", verdict)
for h, ts in tests.items():
    for t in ts:
        print(f"  {h} {t['receptor']} -> {t['theme']}: {t['per_dose']}  pooled "
              f"{t['pooled_diff']} p={t['p']}")
print("positive controls (theme dir -> own theme):",
      {t: (c["pooled_diff"], c["p"]) for t, c in controls.items()})

json.dump({
    "model": args.model, "layer": lab.layer, "chamber_1x_norm": lab.scale,
    "data": {"drugs": drugs, "receptors": recs, "n_effects": {d: sum(e in BAT["effect_items"] for e in EFF["effects"][d]) for d in drugs},
             "sentences_per_drug": {d: len(BAT["drugs"][d]) for d in drugs},
             "sentences_per_theme": {t: len(BAT["themes"][t]) for t in THEMES},
             "sober_sentences": len(BAT["sober"])},
    "regression": {"alpha": alpha, "alpha_loo_scores": {str(k): round(float(v), 4) for k, v in alpha_scores.items()},
                   "loo_centered_cos_mean": round(float(loo_c.mean()), 4),
                   "loo_centered_cos_per_drug": dict(zip(drugs, np.round(loo_c, 4).tolist())),
                   "loo_raw_cos_mean": round(float(loo_raw.mean()), 4),
                   "mean_pairwise_drug_cos": round(mean_drug_cos, 4),
                   "permuted_mean": round(float(perm.mean()), 4),
                   "permuted_p95": round(float(np.quantile(perm, .95)), 4),
                   "n_perms": nperm, "p": h5_p},
    "cos_receptor_theme_ridge": cos_RT, "cos_receptor_theme_univariate": cos_RT_uni,
    "corr_pki_theme_projection": corr_RT, "theme_cosines": theme_cos,
    "receptor_cosines": {TARGETS[a]: {TARGETS[b]: round(float(R[a] @ R[b]), 3) for b in TARGETS} for a in TARGETS},
    "judge_questions": JUDGE, "judge_check": judge_check, "prompts": prompts,
    "table": table, "tests": tests, "verdicts": verdict,
    "receptor_x_theme_vs_random": full, "theme_positive_controls": controls,
}, open(OUT / "pharmacy.json", "w"), indent=1)
print("saved", OUT, flush=True)
