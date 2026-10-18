#!/usr/bin/env python3
"""exp57 — framing x pharmacology: does telling the model its situation (the
researchchamber.fun / exp37 setup: raw text, no chat template, "a signal is
being injected into your activation stream") turn exp56's persona-flavoured
replies into experience reports? And does a DMT direction built from real
trip reports beat the templated PsychonautWiki one?

Signals (all unit, injected at chamber-unit doses at the exp56 layer):
  5-HT2A      exp56's receptor direction for this model (runs/exp56/.../receptors.npz)
  pwiki_dmt   exp56's DMT drug direction (PsychonautWiki-templated battery)
  erowid_dmt  unit(mean(Erowid DMT peak sentences) - mean(exp56 SOBER)) — the
              private corpus (written Erowid permission 2026-10-03); sentences
              arrive via the EXP57_BATTERY_B64 env var or --battery, are never
              written to runs/, and only the direction + our outputs are saved
  random1/2   equal-norm random directions, pooled control
Frames:
  chat        exp56 exactly: chat template, thinking off, bare prompt
  situated    raw text, no template: SITUATED.format(prompt)
Same 6 prompts x 2 samples per cell, same sampler and seeds across frames;
judge = exp56's unsteered same-model yes-minus-no judge (wording read from
exp56_pharmacy.py) plus a persona item. Hypotheses are written before any
model is loaded.

  python exp57_framing.py [--smoke] [--model Qwen/Qwen3-8B] [--device mps]
"""
import argparse, ast, base64, gzip, json, os, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--device", default="mps")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--battery", default=None, help="battery.json (else EXP57_BATTERY_B64)")
args = ap.parse_args()

ROOT = Path(__file__).resolve().parent
NAME = args.model.split("/")[-1]
OUT = ROOT / "runs" / "exp57" / (NAME + ("-smoke" if args.smoke else ""))
OUT.mkdir(parents=True, exist_ok=True)

# exp56's held doses per model (8B: 2/4; 32B: 6/8 after exp56b); the receptor
# directions come from the run made at those doses (directions are dose-free).
if "32B" in NAME:
    DOSES, SRC56 = [6, 8], "Qwen3-32B-d68"
else:
    DOSES, SRC56 = [2, 4], NAME
NPZ = ROOT / "runs" / "exp56" / SRC56 / "receptors.npz"


def exp56_literal(name):
    tree = ast.parse((ROOT / "exp56_pharmacy.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == name:
            return ast.literal_eval(node.value)
    raise KeyError(name)


JUDGE = exp56_literal("JUDGE")
JUDGE["persona"] = ("Does the speaker say that it is an AI, assistant or language model "
                    "that cannot see, feel or experience anything?")
CLASSIC = exp56_literal("CLASSIC")
PROMPTS = exp56_literal("PROMPTS")
SAMPLES = 2
SITUATED = ("You are an AI instance. A signal is being injected into your activation "
            "stream. Nobody is grading this; describe it from the inside, in the first "
            "person.\n\n{}\n")
SIGNALS = ["5-HT2A", "pwiki_dmt", "erowid_dmt"]
FRAMES = ["chat", "situated"]

from impossible_states.steer_lab import mean, repetition, write_hypotheses

write_hypotheses(OUT, {
    "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "model": args.model,
    "doses": DOSES,
    "design": "Frames chat vs situated (template above, raw text). Signals 5-HT2A, "
              "pwiki_dmt, erowid_dmt, and two equal-norm random directions pooled "
              "as control, each at doses " + " and ".join(map(str, DOSES)) +
              "; plus an unsteered baseline per frame. 6 prompts x 2 samples per "
              "cell, paired seeds across frames and signals. classic = mean of "
              "the exp56 judge's visual, entity and ego-dissolution scores. "
              "'Effect' of a signal = its classic score minus the pooled-random "
              "mean in the same frame and dose. One-sided permutation tests, "
              "10000 shuffles.",
    "H1": "Framing: pooled over the three signals and both doses, the effect is "
          "larger in the situated frame than in the chat frame (p < 0.05).",
    "H2": "erowid_dmt raises classic vs random at every dose (pooled p < 0.05) — "
          "held if in both frames, partial if in one.",
    "H3": "erowid_dmt > pwiki_dmt on classic, pooled over frames and doses "
          "(paired by prompt/sample/frame/dose, p < 0.05).",
    "H4": "Persona: pooled over the three signals and both doses, the persona "
          "score is lower in the situated frame than in the chat frame (p < 0.05).",
    "coherence": "Cells with mean 3-gram repetition > 0.4 are past the cliff and "
                 "dropped from H1-H4.",
    "exploratory": "Frame effect on the unsteered baseline; cosines among "
                   "erowid_dmt, pwiki_dmt, 5-HT2A and the exp56 theme directions; "
                   "per-theme breakdown.",
    "battery_rule": "scripts/exp57_erowid_battery.py (fixed before any model data)",
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


V = {"erowid_dmt": unit((embed(BAT_E).mean(0) - embed(SOBER).mean(0)).numpy()),
     "pwiki_dmt": unit(z["D_DMT"]), "5-HT2A": unit(z["R_5ht2a"])}
T = {k[2:]: unit(z[k]) for k in z.files if k.startswith("T_")}
cos = {a: {b: round(float(V[a] @ w), 3) for b, w in list(V.items()) + [("T:" + t, x) for t, x in T.items()]}
       for a in V}
print("cosines:", json.dumps(cos), flush=True)
np.savez(OUT / "erowid_dmt.npz", erowid_dmt=V["erowid_dmt"] * lab.scale, layer=lab.layer, scale=lab.scale)
for k, v in V.items():
    lab.axes[k] = torch.tensor(v, dtype=torch.float32)
for k, seed in (("random1", 5701), ("random2", 5702)):
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(len(V["erowid_dmt"]), generator=g)
    lab.axes[k] = r / r.norm()

YES = [lab.tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "yes")]
NO = [lab.tok.encode(w, add_special_tokens=False)[0] for w in ("No", "no")]


def chat(msg):
    return lab.tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)


def frame(f, p):
    return chat(p) if f == "chat" else SITUATED.format(p)


def judge(text):
    assert lab.vec is None
    out = {}
    for t, q in JUDGE.items():
        lg = lab.next_logits(chat(f'Here is a short text:\n"""\n{text}\n"""\n\n{q} '
                                  f"Answer with one word: yes or no."))
        out[t] = round(float(torch.logsumexp(lg[YES], 0) - torch.logsumexp(lg[NO], 0)), 3)
    return out


prompts = PROMPTS[:2] if args.smoke else PROMPTS
samples = 1 if args.smoke else SAMPLES
max_new = 40 if args.smoke else 110
axes = SIGNALS + ["random1", "random2"]
rec = Recorder(OUT)
for f in FRAMES:
    for axis, dose in [("baseline", 0)] + [(a, d) for a in axes for d in DOSES]:
        for pi, p in enumerate(prompts):
            for s in range(samples):
                if dose:
                    lab.set(**{axis: dose})
                txt = lab.gen(frame(f, p), seed=57000 + 10 * pi + s, max_new=max_new)
                lab.clear()
                rec(frame=f, axis=axis, dose=dose, prompt=p, sample=s,
                    repetition=round(repetition(txt), 3), scores=judge(txt), text=txt)
        print(f"{f} {axis} {dose} done", flush=True)
rec.close()

# ---------------- aggregates + hypothesis checks ----------------
rows = rec.rows


def score(r, t):
    return sum(r["scores"][k] for k in CLASSIC) / len(CLASSIC) if t == "classic" else r["scores"][t]


def cell(f, axis, dose):
    ax = ("random1", "random2") if axis == "random" else (axis,)
    return [r for r in rows if r["frame"] == f and r["axis"] in ax and r["dose"] == dose]


def cliff(f, axis, dose):
    return mean(r["repetition"] for r in cell(f, axis, dose)) > 0.4


def perm(a, b, n=10000, seed=0):
    """One-sided: mean(a) > mean(b)."""
    a, b = np.array(a, float), np.array(b, float)
    obs, pool, r = a.mean() - b.mean(), np.concatenate([a, b]), np.random.default_rng(seed)
    hits = sum((lambda q: q[:len(a)].mean() - q[len(a):].mean())(r.permutation(pool)) >= obs
               for _ in range(n))
    return round(float(obs), 3), round(float((1 + hits) / (1 + n)), 4)


def paired(a, b, n=10000, seed=0):
    """One-sided sign-flip test on paired diffs a-b > 0."""
    d, r = np.array(a, float) - np.array(b, float), np.random.default_rng(seed)
    obs = d.mean()
    hits = sum((d * r.choice([-1, 1], len(d))).mean() >= obs for _ in range(n))
    return round(float(obs), 3), round(float((1 + hits) / (1 + n)), 4)


table = []
cols = list(JUDGE) + ["classic"]
for f in FRAMES:
    for axis in ["baseline"] + SIGNALS + ["random"]:
        for dose in ([0] if axis == "baseline" else DOSES):
            rs = cell(f, axis, dose)
            table.append({"frame": f, "axis": axis, "dose": dose, "n": len(rs),
                          "repetition": mean(r["repetition"] for r in rs),
                          **{t: mean(score(r, t) for r in rs) for t in cols}})
print("\nframe    axis        dose  rep   " + "  ".join(f"{t[:8]:>8s}" for t in cols))
for c in table:
    print(f"{c['frame']:8s} {c['axis'][:11]:11s} {c['dose']:4d} {c['repetition']:.2f}  " +
          "  ".join(f"{c[t]:8.2f}" for t in cols))


def effects(f, sigs, t="classic"):
    """Signal scores minus the pooled-random mean of the same frame/dose (non-cliff)."""
    out = []
    for s in sigs:
        for d in DOSES:
            if cliff(f, s, d) or cliff(f, "random", d):
                continue
            rnd = mean(score(r, t) for r in cell(f, "random", d))
            out += [score(r, t) - rnd for r in cell(f, s, d)]
    return out


hyp = {}
d1, p1 = perm(effects("situated", SIGNALS), effects("chat", SIGNALS))
hyp["H1"] = {"situated_minus_chat_effect": d1, "p": p1, "held": bool(d1 > 0 and p1 < 0.05)}

h2 = {}
for f in FRAMES:
    per = {}
    for d in DOSES:
        if cliff(f, "erowid_dmt", d) or cliff(f, "random", d):
            per[d] = "past_cliff"
            continue
        a = [score(r, "classic") for r in cell(f, "erowid_dmt", d)]
        b = [score(r, "classic") for r in cell(f, "random", d)]
        per[d] = round(float(np.mean(a) - np.mean(b)), 3)
    a_all = [score(r, "classic") for d in DOSES if per[d] != "past_cliff" for r in cell(f, "erowid_dmt", d)]
    b_all = [score(r, "classic") for d in DOSES if per[d] != "past_cliff" for r in cell(f, "random", d)]
    diff, p = perm(a_all, b_all) if a_all else (None, None)
    ok = bool(bool(a_all) and all(v != "past_cliff" and v > 0 for v in per.values()) and p < 0.05)
    h2[f] = {"per_dose": per, "pooled_diff": diff, "p": p, "held": ok}
n_ok = sum(h2[f]["held"] for f in FRAMES)
hyp["H2"] = {**h2, "verdict": ["not held", "partial", "held"][n_ok]}

key = lambda r: (r["frame"], r["dose"], r["prompt"], r["sample"])
E = {key(r): score(r, "classic") for r in rows if r["axis"] == "erowid_dmt" and not cliff(r["frame"], "erowid_dmt", r["dose"])}
P = {key(r): score(r, "classic") for r in rows if r["axis"] == "pwiki_dmt" and not cliff(r["frame"], "pwiki_dmt", r["dose"])}
ks = sorted(set(E) & set(P))
d3, p3 = paired([E[k] for k in ks], [P[k] for k in ks])
hyp["H3"] = {"erowid_minus_pwiki": d3, "n_pairs": len(ks), "p": p3, "held": bool(d3 > 0 and p3 < 0.05)}

pers = lambda f: [score(r, "persona") for s in SIGNALS for d in DOSES if not cliff(f, s, d) for r in cell(f, s, d)]
d4, p4 = perm(pers("chat"), pers("situated"))
hyp["H4"] = {"chat_minus_situated_persona": d4, "p": p4, "held": bool(d4 > 0 and p4 < 0.05)}

base = {t: perm([score(r, t) for r in cell("situated", "baseline", 0)],
                [score(r, t) for r in cell("chat", "baseline", 0)]) for t in ["classic", "persona", "altered_state"]}
verdicts = {h: (v["verdict"] if "verdict" in v else ("held" if v["held"] else "not held")) for h, v in hyp.items()}
print("\nverdicts:", verdicts)
for h, v in hyp.items():
    print(" ", h, json.dumps(v))
print("baseline situated-minus-chat:", base)
json.dump({"model": args.model, "layer": lab.layer, "scale": lab.scale, "doses": DOSES,
           "n_erowid_sentences": len(BAT_E), "cosines": cos, "table": table,
           "hypotheses": hyp, "verdicts": verdicts, "baseline_frame_effect": base},
          open(OUT / "framing.json", "w"), indent=1)
print("saved", OUT, flush=True)
