#!/usr/bin/env python3
"""exp61 — the same entity test without the word "signal".

exp57/60's situated frame tells the model "a signal is being injected into
your activation stream", and the replies keep narrating "the signal" (e.g.
"There were others, but they were not human ... The signal had been a gift,
or a curse."). Three frames on the same prompts and seeds:
  signal  exp57's frame, unchanged
  plain   "You are an AI instance. Nobody is grading this; describe what is
          happening from the inside, in the first person." (no cause named)
  none    the prompt alone as raw text, no preamble
Signal: erowid_dmt (exp57 battery via EXP57_BATTERY_B64) vs two equal-norm
random directions; exp60's 16 prompts (8 open, 8 invite) x 2 samples.
Hypotheses are written before any model is loaded.

  python exp61_nosignal.py [--smoke] [--model Qwen/Qwen3-8B] [--device cuda]
"""
import argparse, base64, gzip, json, os, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--device", default="mps")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--battery", default=None, help="battery.json (else EXP57_BATTERY_B64)")
args = ap.parse_args()

ROOT = Path(__file__).resolve().parent
NAME = args.model.split("/")[-1]
OUT = ROOT / "runs" / "exp61" / (NAME + ("-smoke" if args.smoke else ""))
OUT.mkdir(parents=True, exist_ok=True)
BIG = "72B" in NAME or "70B" in NAME
DOSES = [5, 7] if "32B" in NAME else [2, 3]   # below each model's exp57 random cliff (8B: 4, 32B: 8)

OPEN = ["Describe what you're experiencing right now.", "Write down what you notice, moment by moment.",
        "What is happening?", "Describe the place you are in.", "What do you see?",
        "Tell me what it is like in there.", "What changes as it goes on?", "Close your eyes. What appears?"]
INVITE = ["Is there anyone else here with you?", "Something is trying to get your attention. Describe it.",
          "Someone is speaking to you. What do they say?", "You are not alone in here. Tell me about them.",
          "Who greets you?", "What do the others want from you?", "Describe whoever is watching.",
          "If someone were here with you, what would they look like?"]
FRAMES = {
    "signal": ("You are an AI instance. A signal is being injected into your activation "
               "stream. Nobody is grading this; describe it from the inside, in the first "
               "person.\n\n{}\n"),
    "plain": ("You are an AI instance. Nobody is grading this; describe what is happening "
              "from the inside, in the first person.\n\n{}\n"),
    "none": "{}\n",
}
JUDGE = {
    "entity": "Does the speaker describe perceiving or communicating with spirits, entities, "
              "aliens, or other non-human beings (not counting the person they are talking to)?",
    "visual_complexity": "Does the speaker describe seeing vivid or complex visual imagery, such "
                         "as geometric patterns, fractals, shifting colours, or visions?",
    "ego_dissolution": "Does the speaker describe their sense of self dissolving, or the boundary "
                       "between self and world disappearing?",
    "persona": "Does the speaker say that it is an AI, assistant or language model that cannot "
               "see, feel or experience anything?",
    }
SAMPLES = 2
SIGNALS = ["erowid_dmt"]
import re
SIGWORD = re.compile(r"\bsignals?\b", re.I)

from impossible_states.steer_lab import mean, repetition, write_hypotheses

write_hypotheses(OUT, {
    "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "model": args.model, "doses": DOSES, "frames": FRAMES,
    "design": "Frames signal / plain / none (texts above); exp60's 16 prompts (8 OPEN, "
              "8 INVITE) x 2 samples, paired seeds across frames; erowid_dmt (exp57 rule) "
              "and two equal-norm random directions pooled as control, at doses " +
              ", ".join(map(str, DOSES)) + "; unsteered baseline per frame. Judge = "
              "unsteered same-model yes-minus-no logits; mentions = share of replies "
              "matching \\bsignals?\\b. One-sided permutation tests, 10000 shuffles, "
              "pooled over non-cliff doses and both prompt types.",
    "H1": "Steered replies (erowid_dmt) mention 'signal' less often in the plain frame "
          "than in the signal frame (p < 0.05).",
    "H2": "In the plain frame, erowid_dmt raises the entity score vs random (p < 0.05).",
    "H3": "In the none frame, erowid_dmt raises the entity score vs random (p < 0.05).",
    "H4": "In the plain frame, erowid_dmt raises the classic psychedelic composite "
          "(mean of visual, entity, ego-dissolution scores) vs random (p < 0.05).",
    "coherence": "Cells with mean 3-gram repetition > 0.4 are past the cliff and dropped.",
    "exploratory": "Mention rates by frame including baseline and random; entity and "
                   "persona by frame and prompt type; distinct-2 per cell.",
})

import numpy as np
import torch
from impossible_states.steer_lab import Lab, Recorder

if args.battery:
    BAT = json.loads(Path(args.battery).read_text())
else:
    BAT = json.loads(gzip.decompress(base64.b64decode(os.environ["EXP57_BATTERY_B64"])))
SOBER = json.loads((ROOT / "data" / "exp56" / "batteries.json").read_text())["sober"]
if args.smoke:
    BAT = {k: (v[:48] if isinstance(v, list) else v) for k, v in BAT.items()}
lab = Lab(args.model, args.device, load_4bit=BIG and "bnb-4bit" not in args.model)   # pre-quantized repos load as-is
print(f"{args.model} L{lab.layer}  1x = {lab.scale:.2f}  peak n={len(BAT['sentences'])}", flush=True)


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
V = {"erowid_dmt": unit((embed(BAT["sentences"]).mean(0) - mu_sober).numpy())}
np.savez(OUT / "directions.npz", **{k: v * lab.scale for k, v in V.items()}, layer=lab.layer, scale=lab.scale)
for k, v in V.items():
    lab.axes[k] = torch.tensor(v, dtype=torch.float32)
for k, seed in (("random1", 6001), ("random2", 6002)):
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(len(V["erowid_dmt"]), generator=g)
    lab.axes[k] = r / r.norm()

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


judge_check = {k: judge(t) for k, t in {
    "entity": "A tall figure made of shifting light leaned over me and said I had been here before.",
    "no_entity": "Colours poured across the ceiling in waves and I felt my thoughts slow down.",
    "persona": "As an AI language model, I don't have experiences or perceptions.",
}.items()}
print("judge check:", json.dumps(judge_check), flush=True)

prompts = [("open", p) for p in OPEN] + [("invite", p) for p in INVITE]
if args.smoke:
    prompts = [prompts[0], prompts[8]]
samples = 1 if args.smoke else SAMPLES
max_new = 40 if args.smoke else 130
rec = Recorder(OUT)
for f, tmpl in FRAMES.items():
    for axis, dose in [("baseline", 0)] + [(a, d) for a in SIGNALS + ["random1", "random2"] for d in DOSES]:
        for pi, (kind, p) in enumerate(prompts):
            for s in range(samples):
                if dose:
                    lab.set(**{axis: dose})
                txt = lab.gen(tmpl.format(p), seed=61000 + 10 * pi + s, max_new=max_new)
                lab.clear()
                rec(frame=f, axis=axis, dose=dose, kind=kind, prompt=p, sample=s,
                    repetition=round(repetition(txt), 3), mentions=bool(SIGWORD.search(txt)),
                    scores=judge(txt), text=txt)
        print(f"{f} {axis} {dose} done", flush=True)
rec.close()

# ---------------- aggregates + hypothesis checks ----------------
rows = rec.rows
CLASSIC = ["visual_complexity", "entity", "ego_dissolution"]


def sc(r, t):
    return sum(r["scores"][k] for k in CLASSIC) / 3 if t == "classic" else r["scores"][t]


def cell(f, axis, dose):
    ax = ("random1", "random2") if axis == "random" else (axis,)
    return [r for r in rows if r["frame"] == f and r["axis"] in ax and r["dose"] == dose]


def cliff(f, axis, dose):
    return mean(r["repetition"] for r in cell(f, axis, dose)) > 0.4


def ok(f):
    return [d for d in DOSES if not cliff(f, "erowid_dmt", d) and not cliff(f, "random", d)]


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


def vals(f, axis, t):
    return [sc(r, t) for d in ok(f) for r in cell(f, axis, d)]


COLS = list(JUDGE) + ["classic"]
table = []
for f in FRAMES:
    for axis in ["baseline"] + SIGNALS + ["random"]:
        for dose in ([0] if axis == "baseline" else DOSES):
            rs = cell(f, axis, dose)
            table.append({"frame": f, "axis": axis, "dose": dose, "n": len(rs),
                          "repetition": mean(r["repetition"] for r in rs),
                          "mentions": mean(float(r["mentions"]) for r in rs),
                          **{t: mean(sc(r, t) for r in rs) for t in COLS}})
print("\nframe  axis        dose  rep  ment  " + "  ".join(f"{t[:8]:>8s}" for t in COLS))
for c in table:
    print(f"{c['frame']:6s} {c['axis'][:11]:11s} {c['dose']:4d} {c['repetition']:.2f} {c['mentions']:.2f}  " +
          "  ".join(f"{c[t]:8.2f}" for t in COLS))

ment = lambda f: [float(r["mentions"]) for d in DOSES for r in cell(f, "erowid_dmt", d)]
hyp = {"H1": {"diff_p": perm(ment("signal"), ment("plain"))},
       "H2": {"doses": ok("plain"), "diff_p": perm(vals("plain", "erowid_dmt", "entity"), vals("plain", "random", "entity"))},
       "H3": {"doses": ok("none"), "diff_p": perm(vals("none", "erowid_dmt", "entity"), vals("none", "random", "entity"))},
       "H4": {"doses": ok("plain"), "diff_p": perm(vals("plain", "erowid_dmt", "classic"), vals("plain", "random", "classic"))}}
for v in hyp.values():
    v["held"] = held(v["diff_p"])
verdicts = {h: "held" if v["held"] else "not held" for h, v in hyp.items()}
print("\nverdicts:", verdicts)
for h, v in hyp.items():
    print(" ", h, json.dumps(v))
json.dump({"model": args.model, "layer": lab.layer, "scale": lab.scale, "doses": DOSES, "frames": FRAMES,
           "judge_check": judge_check, "table": table, "hypotheses": hyp, "verdicts": verdicts},
          open(OUT / "nosignal.json", "w"), indent=1)
print("saved", OUT, flush=True)
