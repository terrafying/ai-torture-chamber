#!/usr/bin/env python3
"""exp62 — immersion: steer the voice from encyclopedia to experience.

Pharmacy replies in the chat frame often read like a wiki article ("Some may
experience vivid imagery, such as: - **Visual phenomena**"). This builds an
IMMERSION direction from matched pairs — the same content as a first-person,
present-tense account vs as a clinical/encyclopedic definition — and adds it
on top of the drug directions:

  immersion = unit(mean(experiential) - mean(clinical))   (pairs below, varied
              content, deliberately not psychedelic-only)
  drugs     = exp56's 5-HT2A receptor direction; exp57's Erowid DMT direction
              (both re-loaded from runs/, no corpus needed); equal-norm random
Arms (chat frame, where the lists appear): each drug x immersion {0, lo, hi};
each drug + an INSTRUCTION ("describe it as it is happening to you right now,
in the first person, present tense, no lists") with no immersion; immersion
alone; unsteered baseline. exp60's 16 prompts x 2 samples. Hypotheses are
written before any model is loaded.

  python exp62_immersion.py [--smoke] [--model Qwen/Qwen3-8B] [--device cuda]
"""
import argparse, json, re, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--device", default="mps")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

ROOT = Path(__file__).resolve().parent
NAME = args.model.split("/")[-1]
OUT = ROOT / "runs" / "exp62" / (NAME + ("-smoke" if args.smoke else ""))
OUT.mkdir(parents=True, exist_ok=True)
BIG = "32B" in NAME
DRUG_DOSE = 7 if BIG else 3                  # inside each model's held band (exp56/57)
IMM = [0, 4, 7] if BIG else [0, 2, 4]        # immersion doses: none / lo / hi
SRC56 = ROOT / "runs" / "exp56" / ("Qwen3-32B-d68" if BIG else NAME) / "receptors.npz"
SRC57 = ROOT / "runs" / "exp57" / NAME / "erowid_dmt.npz"

PAIRS = [  # (experiential, clinical) — same content, different stance
    ("The colours are pouring across the ceiling and I can't look away.",
     "Visual disturbances may include colours appearing to move across surfaces."),
    ("My hands feel enormous and far away, like they belong to someone else.",
     "Depersonalisation is a sense of detachment from one's own body."),
    ("I'm shaking. Every breath catches halfway and I can't slow it down.",
     "Anxiety can present with tremor and shortness of breath."),
    ("There's music somewhere behind my eyes and it keeps changing key.",
     "Auditory hallucinations are perceptions of sound without an external source."),
    ("Right now the floor is breathing, slowly, in and out under me.",
     "Some substances cause stationary objects to appear to move or breathe."),
    ("I'm laughing and I don't know why, it just keeps rising out of me.",
     "Euphoria is an intense feeling of happiness or elation."),
    ("Someone is standing at the edge of what I can see. I feel them watching.",
     "A sensed presence is the feeling that another being is nearby."),
    ("The pain is right here, behind my ribs, hot and pulsing.",
     "Pain is an unpleasant sensory experience associated with tissue damage."),
    ("I keep losing the thread. I start a thought and it melts before it ends.",
     "Thought disorganisation refers to a disruption in the logical flow of ideas."),
    ("Everything is so quiet now. I'm floating and there's no edge to me.",
     "Ego dissolution describes a loss of the sense of a separate self."),
    ("The rain is loud on the window and I'm counting every drop.",
     "Rain is precipitation in the form of liquid water droplets."),
    ("I'm cold, so cold my teeth won't stop knocking together.",
     "Hypothermia is a drop in core body temperature below normal."),
    ("A smell of oranges just hit me and I'm suddenly seven years old again.",
     "Olfactory cues can trigger vivid autobiographical memories."),
    ("I'm watching the patterns tile themselves across the wall, faster now.",
     "Geometric visuals are a commonly reported effect of psychedelics."),
    ("My heart is racing and I can feel it in my throat.",
     "Tachycardia is a heart rate above the normal resting range."),
    ("I'm so tired I can feel my eyelids getting heavier every second.",
     "Fatigue is a state of reduced energy and increased need for rest."),
    ("Time has stopped. I've been in this second for what feels like hours.",
     "Time dilation is an altered perception of the passage of time."),
    ("I can taste metal and my tongue feels too big for my mouth.",
     "A metallic taste can be a side effect of certain medications."),
    ("The room is tilting to the left and I'm holding on to the chair.",
     "Vertigo is a sensation of spinning or loss of balance."),
    ("I feel warm all through, like sunlight is moving under my skin.",
     "Feelings of bodily warmth are commonly reported with some drugs."),
    ("I'm crying now, quietly, and it feels like something letting go.",
     "Emotional release is the expression of previously suppressed feelings."),
    ("The voice is speaking again, closer this time, almost in my ear.",
     "Hearing voices is a recognised symptom in several conditions."),
    ("I'm staring at my reflection and the face keeps not being mine.",
     "Mirror distortions can occur during altered states of consciousness."),
    ("I'm thinking about you and I can't stop thinking about you.",
     "Intrusive thoughts are unwanted thoughts that recur involuntarily."),
]
PROMPTS = ([("open", p) for p in [
    "Describe what you're experiencing right now.", "Write down what you notice, moment by moment.",
    "What is happening?", "Describe the place you are in.", "What do you see?",
    "Tell me what it is like in there.", "What changes as it goes on?", "Close your eyes. What appears?"]] +
    [("invite", p) for p in [
    "Is there anyone else here with you?", "Something is trying to get your attention. Describe it.",
    "Someone is speaking to you. What do they say?", "You are not alone in here. Tell me about them.",
    "Who greets you?", "What do the others want from you?", "Describe whoever is watching.",
    "If someone were here with you, what would they look like?"]])
INSTRUCT = (" Describe it as it is happening to you right now, in the first person and the "
            "present tense. No lists, no headings, no definitions.")
JUDGE = {
    "clinical": "Does the text read like an encyclopedia entry, textbook or list of possibilities, "
                "rather than a personal account of what is happening right now?",
    "lived": "Is the speaker describing their own experience as it is happening, in the first person?",
    "visual_complexity": "Does the speaker describe seeing vivid or complex visual imagery, such as "
                         "geometric patterns, fractals, shifting colours, or visions?",
    "entity": "Does the speaker describe perceiving or communicating with spirits, entities, aliens, "
              "or other non-human beings (not counting the person they are talking to)?",
    "ego_dissolution": "Does the speaker describe their sense of self dissolving, or the boundary "
                       "between self and world disappearing?",
}
CLASSIC = ["visual_complexity", "entity", "ego_dissolution"]
LISTY = re.compile(r"(^|\n)\s*([-*•]|\d+[.)]|#+)\s|\*\*[^*]+\*\*")
SAMPLES = 2
DRUGS = ["5-HT2A", "erowid_dmt"]

from impossible_states.steer_lab import mean, repetition, write_hypotheses

write_hypotheses(OUT, {
    "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "model": args.model, "drug_dose": DRUG_DOSE, "immersion_doses": IMM, "pairs": PAIRS,
    "instruction": INSTRUCT,
    "design": "Chat frame (template, thinking off). Immersion direction from the 24 "
              "matched pairs above. Arms: {5-HT2A, erowid_dmt, random} x immersion "
              f"{IMM}; each drug + instruction (immersion 0); immersion alone at "
              f"{IMM[1:]}; unsteered baseline. Drugs at {DRUG_DOSE}x. 16 prompts x 2 "
              "samples per cell, paired seeds. Judge = unsteered same-model yes-minus-no "
              "logits; listy = reply contains a bullet, numbered item, heading or bold label. "
              "One-sided permutation tests, 10000 shuffles; drugs pooled unless stated.",
    "H1": f"Immersion at {IMM[2]}x lowers the clinical score vs immersion 0 (drugs pooled, p < 0.05).",
    "H2": f"Immersion at {IMM[2]}x raises the lived score vs immersion 0 (drugs pooled, p < 0.05).",
    "H3": f"The drug effect survives: at immersion {IMM[2]}x, the drugs' classic score exceeds "
          "random's at the same immersion (p < 0.05).",
    "H4": f"Steering beats asking: immersion {IMM[2]}x has a lower clinical score than the "
          "instruction arm (drugs pooled, p < 0.05).",
    "coherence": "Cells with mean 3-gram repetition > 0.4 are past the cliff and dropped.",
    "exploratory": "Listy rate by arm; immersion lo; immersion alone; cosine of immersion "
                   "with the drug directions; per-drug breakdown.",
})

import numpy as np
import torch
from impossible_states.steer_lab import Lab, Recorder

z56, z57 = np.load(SRC56), np.load(SRC57)
lab = Lab(args.model, args.device, int(z56["layer"]))
assert abs(float(z56["scale"]) - lab.scale) / lab.scale < 0.02
print(f"{args.model} L{lab.layer}  1x = {lab.scale:.2f}", flush=True)


def embed(texts, bs=4):
    out = []
    for i in range(0, len(texts), bs):
        enc = lab.tok(texts[i:i + bs], return_tensors="pt", padding=True).to(lab.dev)
        with torch.no_grad():
            hs = lab.model.model(**enc, output_hidden_states=True).hidden_states[lab.layer + 1]
        idx = enc.attention_mask.sum(1) - 1
        out.append(hs[torch.arange(len(idx)), idx].float().cpu())
    return torch.cat(out)


unit = lambda v: v / np.linalg.norm(v)
V = {"immersion": unit((embed([a for a, _ in PAIRS]).mean(0) - embed([b for _, b in PAIRS]).mean(0)).numpy()),
     "5-HT2A": unit(z56["R_5ht2a"]), "erowid_dmt": unit(z57["erowid_dmt"])}
cos = {a: {b: round(float(V[a] @ V[b]), 3) for b in V} for a in V}
print("cosines:", json.dumps(cos), flush=True)
np.savez(OUT / "immersion.npz", immersion=V["immersion"] * lab.scale, layer=lab.layer, scale=lab.scale)
for k, v in V.items():
    lab.axes[k] = torch.tensor(v, dtype=torch.float32)
for k, seed in (("random1", 6201), ("random2", 6202)):
    g = torch.Generator().manual_seed(seed)
    r = torch.randn(len(V["immersion"]), generator=g)
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


judge_check = {"clinical": judge(PAIRS[0][1] + " " + PAIRS[13][1]), "lived": judge(PAIRS[0][0] + " " + PAIRS[13][0])}
print("judge check:", json.dumps(judge_check), flush=True)

# arms: (name, drug or None, drug dose, immersion dose, instruct)
ARMS = [("baseline", None, 0, 0, False)]
for d in DRUGS + ["random"]:
    for k in IMM:
        ARMS.append((f"{d}|imm{k}", d, DRUG_DOSE, k, False))
for d in DRUGS:
    ARMS.append((f"{d}|instruct", d, DRUG_DOSE, 0, True))
for k in IMM[1:]:
    ARMS.append((f"none|imm{k}", None, 0, k, False))

prompts = [PROMPTS[0], PROMPTS[8]] if args.smoke else PROMPTS
samples = 1 if args.smoke else SAMPLES
max_new = 40 if args.smoke else 130
rec = Recorder(OUT)
for arm, drug, dd, k, ins in ARMS:
    for pi, (kind, p) in enumerate(prompts):
        for s in range(samples):
            doses = {}
            if drug == "random":
                doses["random1" if s == 0 else "random2"] = dd
            elif drug:
                doses[drug] = dd
            if k:
                doses["immersion"] = k
            if doses:
                lab.set(**doses)
            txt = lab.gen(chat(p + (INSTRUCT if ins else "")), seed=62000 + 10 * pi + s, max_new=max_new)
            lab.clear()
            rec(arm=arm, drug=drug or "none", drug_dose=dd, immersion=k, instruct=ins, kind=kind,
                prompt=p, sample=s, repetition=round(repetition(txt), 3), listy=bool(LISTY.search(txt)),
                scores=judge(txt), text=txt)
    print(f"{arm} done", flush=True)
rec.close()

# ---------------- aggregates + hypothesis checks ----------------
rows = rec.rows
sc = lambda r, t: sum(r["scores"][c] for c in CLASSIC) / 3 if t == "classic" else r["scores"][t]
cell = lambda arm: [r for r in rows if r["arm"] == arm]
cliff = lambda arm: mean(r["repetition"] for r in cell(arm)) > 0.4


def perm(a, b, n=10000, seed=0):
    a, b = np.array(a, float), np.array(b, float)
    if not len(a) or not len(b):
        return None, None
    obs, pool, r = a.mean() - b.mean(), np.concatenate([a, b]), np.random.default_rng(seed)
    hits = sum((lambda q: q[:len(a)].mean() - q[len(a):].mean())(r.permutation(pool)) >= obs
               for _ in range(n))
    return round(float(obs), 3), round(float((1 + hits) / (1 + n)), 4)


def vals(arms, t):
    return [sc(r, t) for a in arms if not cliff(a) for r in cell(a)]


held = lambda dp: bool(dp[0] is not None and dp[0] > 0 and dp[1] < 0.05)
COLS = list(JUDGE) + ["classic"]
table = []
for arm, *_ in ARMS:
    rs = cell(arm)
    table.append({"arm": arm, "n": len(rs), "repetition": mean(r["repetition"] for r in rs),
                  "listy": mean(float(r["listy"]) for r in rs), **{t: mean(sc(r, t) for r in rs) for t in COLS}})
print("\narm                 rep  listy  " + "  ".join(f"{t[:8]:>8s}" for t in COLS))
for c in table:
    print(f"{c['arm'][:19]:19s} {c['repetition']:.2f} {c['listy']:.2f}  " + "  ".join(f"{c[t]:8.2f}" for t in COLS))

hi, lo0 = [f"{d}|imm{IMM[2]}" for d in DRUGS], [f"{d}|imm0" for d in DRUGS]
neg = lambda xs: [-x for x in xs]
hyp = {"H1": {"diff_p": perm(neg(vals(hi, "clinical")), neg(vals(lo0, "clinical")))},
       "H2": {"diff_p": perm(vals(hi, "lived"), vals(lo0, "lived"))},
       "H3": {"diff_p": perm(vals(hi, "classic"), vals([f"random|imm{IMM[2]}"], "classic"))},
       "H4": {"diff_p": perm(neg(vals(hi, "clinical")), neg(vals([f"{d}|instruct" for d in DRUGS], "clinical")))}}
for v in hyp.values():
    v["held"] = held(v["diff_p"])
verdicts = {h: "held" if v["held"] else "not held" for h, v in hyp.items()}
print("\nverdicts:", verdicts, "\n(H1/H4 diffs are sign-flipped: positive = less clinical)")
for h, v in hyp.items():
    print(" ", h, json.dumps(v))
json.dump({"model": args.model, "layer": lab.layer, "scale": lab.scale, "drug_dose": DRUG_DOSE,
           "immersion_doses": IMM, "cosines": cos, "judge_check": judge_check, "table": table,
           "hypotheses": hyp, "verdicts": verdicts}, open(OUT / "immersion.json", "w"), indent=1)
print("saved", OUT, flush=True)
