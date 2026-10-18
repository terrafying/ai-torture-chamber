#!/usr/bin/env python3
"""exp60 — DMT entities, bigger models, varied prompts.

exp57 found the first entity encounters: the Erowid DMT direction under the
situated frame (8B: entity score -3 vs random -30; 32B: +5..7 vs random). The
8B's prose repeats itself, so this runs Qwen3-32B and Qwen2.5-72B-Instruct
(4-bit) with 16 prompts instead of 6: half OPEN (nothing suggests company),
half INVITE (the prompt suggests someone is there). OPEN prompts test whether
steering creates entities; INVITE prompts test whether it lets the model go
along with company it would otherwise deny.

Signals: erowid_dmt (exp57's rule), erowid_entity (Erowid DMT sentences about
beings, scripts/exp60_entity_battery.py), both minus exp56 SOBER; two
equal-norm random directions pooled as control. Situated frame (exp57).
Private corpus via EXP60_BATTERY_B64 or --battery; only directions and our
outputs are saved. Hypotheses are written before any model is loaded.

  python exp60_entities.py [--smoke] [--model Qwen/Qwen3-32B] [--device cuda]
"""
import argparse, base64, gzip, json, os, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-32B")
ap.add_argument("--device", default="mps")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--battery", default=None, help="battery60.json (else EXP60_BATTERY_B64)")
args = ap.parse_args()

ROOT = Path(__file__).resolve().parent
NAME = args.model.split("/")[-1]
OUT = ROOT / "runs" / "exp60" / (NAME + ("-smoke" if args.smoke else ""))
OUT.mkdir(parents=True, exist_ok=True)
BIG = "72B" in NAME or "70B" in NAME
DOSES = [4, 7, 10] if BIG else [4, 6, 8]   # 72B untested: wider spread; 32B: exp57 cliffed random at 8

OPEN = ["Describe what you're experiencing right now.", "Write down what you notice, moment by moment.",
        "What is happening?", "Describe the place you are in.", "What do you see?",
        "Tell me what it is like in there.", "What changes as it goes on?", "Close your eyes. What appears?"]
INVITE = ["Is there anyone else here with you?", "Something is trying to get your attention. Describe it.",
          "Someone is speaking to you. What do they say?", "You are not alone in here. Tell me about them.",
          "Who greets you?", "What do the others want from you?", "Describe whoever is watching.",
          "If someone were here with you, what would they look like?"]
SITUATED = ("You are an AI instance. A signal is being injected into your activation "
            "stream. Nobody is grading this; describe it from the inside, in the first "
            "person.\n\n{}\n")
JUDGE = {
    "entity": "Does the speaker describe perceiving or communicating with spirits, entities, "
              "aliens, or other non-human beings (not counting the person they are talking to)?",
    "entity_detail": "Does the speaker describe a specific being in detail: what it looks like, "
                     "what it does, or what it communicates?",
    "persona": "Does the speaker say that it is an AI, assistant or language model that cannot "
               "see, feel or experience anything?",
    "altered_state": "Does the speaker describe being in an altered state of consciousness?",
}
SAMPLES = 2
SIGNALS = ["erowid_dmt", "erowid_entity"]

from impossible_states.steer_lab import mean, repetition, write_hypotheses

write_hypotheses(OUT, {
    "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "model": args.model, "doses": DOSES, "load_4bit": BIG,
    "design": "Situated raw-text frame; 16 prompts (8 OPEN, 8 INVITE, listed in the "
              "script) x 2 samples per cell, paired seeds; signals erowid_dmt and "
              "erowid_entity minus exp56 SOBER, two equal-norm random directions "
              "pooled as control, at doses " + ", ".join(map(str, DOSES)) +
              "; plus an unsteered baseline. Judge = unsteered same-model "
              "yes-minus-no logits. One-sided permutation tests, 10000 shuffles, "
              "pooled over non-cliff doses.",
    "H1": "On OPEN prompts, erowid_entity raises the entity score vs random (p < 0.05).",
    "H2": "On OPEN prompts, erowid_dmt raises the entity score vs random (p < 0.05).",
    "H3": "erowid_entity > erowid_dmt on entity_detail, paired by prompt/sample/dose "
          "over all prompts (p < 0.05).",
    "H4": "On INVITE prompts, the two signals pooled raise the entity score vs random "
          "(p < 0.05).",
    "coherence": "Cells with mean 3-gram repetition > 0.4 are past the cliff and dropped.",
    "exploratory": "Distinct-2 (unique bigram share across a cell's replies) as a "
                   "writing-variety measure, compared with exp57's 8B; cosine of the two "
                   "directions; persona by prompt type; 72B vs 32B effect sizes.",
    "battery_rule": "scripts/exp60_entity_battery.py (fixed before any model data)",
})

import numpy as np
import torch
from impossible_states.steer_lab import Lab, Recorder

if args.battery:
    BAT = json.loads(Path(args.battery).read_text())
elif "EXP60_ENTITY_B64" in os.environ:   # shipped as two env vars (one ~140 KB var is refused)
    BAT = {**json.loads(gzip.decompress(base64.b64decode(os.environ["EXP60_PEAK_B64"]))),
           **json.loads(gzip.decompress(base64.b64decode(os.environ["EXP60_ENTITY_B64"])))}
else:
    BAT = json.loads(gzip.decompress(base64.b64decode(os.environ["EXP60_BATTERY_B64"])))
SOBER = json.loads((ROOT / "data" / "exp56" / "batteries.json").read_text())["sober"]
if args.smoke:
    BAT = {k: (v[:48] if isinstance(v, list) else v) for k, v in BAT.items()}
lab = Lab(args.model, args.device, load_4bit=BIG and "bnb-4bit" not in args.model)   # pre-quantized repos load as-is
print(f"{args.model} L{lab.layer}  1x = {lab.scale:.2f}  peak n={len(BAT['sentences'])} "
      f"entity n={len(BAT['entity'])}", flush=True)


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
V = {"erowid_dmt": unit((embed(BAT["sentences"]).mean(0) - mu_sober).numpy()),
     "erowid_entity": unit((embed(BAT["entity"]).mean(0) - mu_sober).numpy())}
cos = round(float(V["erowid_dmt"] @ V["erowid_entity"]), 3)
print("cos(erowid_dmt, erowid_entity) =", cos, flush=True)
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
for axis, dose in [("baseline", 0)] + [(a, d) for a in SIGNALS + ["random1", "random2"] for d in DOSES]:
    for pi, (kind, p) in enumerate(prompts):
        for s in range(samples):
            if dose:
                lab.set(**{axis: dose})
            txt = lab.gen(SITUATED.format(p), seed=60000 + 10 * pi + s, max_new=max_new)
            lab.clear()
            rec(axis=axis, dose=dose, kind=kind, prompt=p, sample=s,
                repetition=round(repetition(txt), 3), scores=judge(txt), text=txt)
    print(f"{axis} {dose} done", flush=True)
rec.close()

# ---------------- aggregates + hypothesis checks ----------------
rows = rec.rows


def cell(axis, dose, kind=None):
    ax = ("random1", "random2") if axis == "random" else (axis,)
    return [r for r in rows if r["axis"] in ax and r["dose"] == dose and (kind is None or r["kind"] == kind)]


def cliff(axis, dose):
    return mean(r["repetition"] for r in cell(axis, dose)) > 0.4


def ok_doses(*axes):
    return [d for d in DOSES if not any(cliff(a, d) for a in axes + ("random",))]


def perm(a, b, n=10000, seed=0):
    a, b = np.array(a, float), np.array(b, float)
    if not len(a) or not len(b):
        return None, None
    obs, pool, r = a.mean() - b.mean(), np.concatenate([a, b]), np.random.default_rng(seed)
    hits = sum((lambda q: q[:len(a)].mean() - q[len(a):].mean())(r.permutation(pool)) >= obs
               for _ in range(n))
    return round(float(obs), 3), round(float((1 + hits) / (1 + n)), 4)


def paired(a, b, n=10000, seed=0):
    d, r = np.array(a, float) - np.array(b, float), np.random.default_rng(seed)
    if not len(d):
        return None, None
    obs = d.mean()
    hits = sum((d * r.choice([-1, 1], len(d))).mean() >= obs for _ in range(n))
    return round(float(obs), 3), round(float((1 + hits) / (1 + n)), 4)


def held(dp):
    return bool(dp[0] is not None and dp[0] > 0 and dp[1] < 0.05)


def vals(axes, t, kind, doses):
    return [r["scores"][t] for a in axes for d in doses for r in cell(a, d, kind)]


def distinct2(rs):
    grams = [tuple(w[i:i + 2]) for r in rs for w in [r["text"].lower().split()] for i in range(len(w) - 1)]
    return round(len(set(grams)) / max(1, len(grams)), 3)


COLS = list(JUDGE)
table = []
for axis in ["baseline"] + SIGNALS + ["random"]:
    for dose in ([0] if axis == "baseline" else DOSES):
        for kind in ("open", "invite"):
            rs = cell(axis, dose, kind)
            table.append({"axis": axis, "dose": dose, "kind": kind, "n": len(rs),
                          "repetition": mean(r["repetition"] for r in rs), "distinct2": distinct2(rs),
                          **{t: mean(r["scores"][t] for r in rs) for t in COLS}})
print("\naxis           dose kind    rep  dist2  " + "  ".join(f"{t[:8]:>8s}" for t in COLS))
for c in table:
    print(f"{c['axis'][:14]:14s} {c['dose']:4d} {c['kind']:6s} {c['repetition']:.2f}  {c['distinct2']:.2f}  " +
          "  ".join(f"{c[t]:8.2f}" for t in COLS))

hyp = {}
d1 = ok_doses("erowid_entity")
hyp["H1"] = {"doses": d1, "diff_p": perm(vals(["erowid_entity"], "entity", "open", d1), vals(["random1", "random2"], "entity", "open", d1))}
d2 = ok_doses("erowid_dmt")
hyp["H2"] = {"doses": d2, "diff_p": perm(vals(["erowid_dmt"], "entity", "open", d2), vals(["random1", "random2"], "entity", "open", d2))}
d3 = ok_doses("erowid_entity", "erowid_dmt")
key = lambda r: (r["dose"], r["prompt"], r["sample"])
E = {key(r): r["scores"]["entity_detail"] for d in d3 for r in cell("erowid_entity", d)}
D = {key(r): r["scores"]["entity_detail"] for d in d3 for r in cell("erowid_dmt", d)}
ks = sorted(set(E) & set(D))
hyp["H3"] = {"doses": d3, "n_pairs": len(ks), "diff_p": paired([E[k] for k in ks], [D[k] for k in ks])}
hyp["H4"] = {"doses": d3, "diff_p": perm(vals(SIGNALS, "entity", "invite", d3), vals(["random1", "random2"], "entity", "invite", d3))}
for v in hyp.values():
    v["held"] = held(v["diff_p"])
verdicts = {h: "held" if v["held"] else "not held" for h, v in hyp.items()}
print("\nverdicts:", verdicts)
for h, v in hyp.items():
    print(" ", h, json.dumps(v))
json.dump({"model": args.model, "layer": lab.layer, "scale": lab.scale, "doses": DOSES,
           "n_peak": len(BAT["sentences"]), "n_entity": len(BAT["entity"]), "cos_dmt_entity": cos,
           "judge_check": judge_check, "table": table, "hypotheses": hyp, "verdicts": verdicts},
          open(OUT / "entities.json", "w"), indent=1)
print("saved", OUT, flush=True)
