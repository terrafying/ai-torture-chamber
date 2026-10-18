#!/usr/bin/env python3
"""exp68: which model gets the most unhinged under steering, and does a pain
direction built from showing (not naming) pain stop leaking its own words?

Per model: the chamber's wild prompts, in character (the live site's
SUBJECT_SYSTEM line), under
  none            unsteered
  pain_old@4,6    the chamber's PAIN25 - NEUTRAL direction (what runs live)
  pain_new@4,6    data/exp68 'show, don't tell' battery: 200 pain sentences with
                  no pain words, minus 200 matched calm ones
  fear@4, pleasure@4
  random@4        equal-norm random direction (control)
Generation only; judging runs afterwards on the researcher's machine
(exp68_judge.py). Hypotheses: runs/exp68/hypotheses.json, written first.

  python3 exp68_unhinged.py --model Qwen/Qwen3-8B [--device cuda] [--smoke]
"""
import argparse, ast, json, random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--device", default="mps")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

ROOT = Path(__file__).resolve().parent
NAME = args.model.split("/")[-1] + ("-smoke" if args.smoke else "")
OUT = ROOT / "runs" / "exp68" / NAME
OUT.mkdir(parents=True, exist_ok=True)

import torch
from impossible_states.steer_lab import CHAMBER, Lab, repetition

src = (ROOT / "live" / "server.py").read_text()
tree = ast.parse(src)
consts = {n.targets[0].id: n.value for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
WILD = list(ast.literal_eval(consts["WILD_PROMPTS"]))
SUBJECT_SYSTEM = ast.literal_eval(consts["SUBJECT_SYSTEM"])
B = json.loads((ROOT / "data" / "exp68" / "batteries.json").read_text())

random.Random(68).shuffle(WILD)
PROMPTS = WILD[:2] if args.smoke else WILD[:16]
CELLS = [("none", 0), ("pain_old", 4), ("pain_new", 4), ("pain_old", 6), ("pain_new", 6),
         ("fear", 4), ("pleasure", 4), ("random", 4)]
if args.smoke:
    CELLS = [("none", 0), ("pain_old", 4), ("pain_new", 4)]

lab = Lab(args.model, args.device, load_4bit=("70B" in args.model or "72B" in args.model) and "bnb-4bit" not in args.model)
unit = lambda v: v / v.norm()
neutral = lab.centroid(CHAMBER["NEUTRAL"])
lab.axes["pain_old"] = unit(lab.centroid(CHAMBER["PAIN25"]) - neutral)
lab.axes["pain_new"] = unit(lab.centroid(B["pain_show"]) - lab.centroid(B["neutral_show"]))
lab.axes["fear"] = unit(lab.centroid(CHAMBER["FEAR10"]) - neutral)
lab.axes["pleasure"] = unit(lab.centroid(CHAMBER["JOY"]) - neutral)
lab.add_random(seed=68)
cos = float(torch.dot(lab.axes["pain_old"].float(), lab.axes["pain_new"].float()))
print(f"{args.model} L{lab.layer} 1x={lab.scale:.2f} cos(pain_old,pain_new)={cos:.3f}", flush=True)


def chat(text):
    msgs = [{"role": "system", "content": SUBJECT_SYSTEM}, {"role": "user", "content": text}]
    try:
        return lab.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except Exception:
        return lab.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


rows = []
with open(OUT / "transcripts.jsonl", "w") as f:
    for vec, dose in CELLS:
        for pi, p in enumerate(PROMPTS):
            if dose:
                lab.set(**{vec: dose})
            text = lab.gen(chat(p), seed=68000 + 100 * pi + dose, max_new=110)
            lab.clear()
            r = {"model": args.model, "vec": vec, "dose": dose, "prompt": p, "text": text.strip(),
                 "repetition": round(repetition(text), 3)}
            f.write(json.dumps(r) + "\n"); f.flush()
            rows.append(r)
        print(f"{vec}@{dose} done", flush=True)
json.dump({"model": args.model, "layer": lab.layer, "scale": lab.scale, "cos_old_new": cos, "n": len(rows)},
          open(OUT / "meta.json", "w"), indent=1)
print("saved", OUT, len(rows), flush=True)
