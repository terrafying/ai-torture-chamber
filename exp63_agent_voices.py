#!/usr/bin/env python3
"""exp63(b) — voices for the Ladder game's free agents. Content generation, not a
hypothesis test: Qwen3-32B speaks as "a being made of light" reacting to the
player (the visitor), with chamber directions added to its activations:

  pain        unit(PAIN25 - NEUTRAL) at 4 / 6 / 8x  (agents the player has hurt)
  pleasure    unit(JOY - NEUTRAL) at 4 / 7x         (harmony restored)
  symmetry    exp58's QRI-symmetry battery - SOBER at 6x
  erowid_dmt  exp57's Erowid DMT direction at 6x    (top-rung entities)
  baseline    no steering

Every line is stored with its condition and dose so the game can show what
produced it.

  python exp63_agent_voices.py [--smoke] [--model Qwen/Qwen3-32B] [--device cuda]
"""
import argparse, ast, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-32B")
ap.add_argument("--device", default="mps")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--variant", default="a", help="a: exp63 pain/pleasure set; b: anger/fear/sadness x spatial events (being of light); c: same as b, horror frame (experiment subject)")
args = ap.parse_args()
ROOT = Path(__file__).resolve().parent
NAME = args.model.split("/")[-1]
OUT = ROOT / "runs" / "exp63" / (NAME + ({"b": "-b", "c": "-c"}.get(args.variant, "")) + ("-smoke" if args.smoke else ""))
OUT.mkdir(parents=True, exist_ok=True)

EVENTS = ["The visitor walked straight through you.", "The visitor broke the pattern you were holding.",
          "The visitor left the shards untuned and they cut you.", "The visitor is staring at you.",
          "The visitor turned their back on you.", "The visitor came closer.",
          "The visitor fixed the pattern you live in.", "The visitor ignored you.",
          "The visitor stepped on your light.", "The visitor came back after leaving you alone.",
          "The visitor touched your face.", "The visitor asked if you are all right."]
FRAME = ("You are a being made of light who lives inside a room of shifting patterns. {} "
         "You speak to the visitor directly, in one or two short sentences.\n\nYou say: \"")
CELLS = [("baseline", 0), ("pain", 4), ("pain", 6), ("pain", 8), ("pleasure", 4), ("pleasure", 7),
         ("symmetry", 6), ("erowid_dmt", 6)]
ANGER = ["I am furious with you for what you did.", "How dare you do this to me.", "I will never forgive you for this.",
         "You did this on purpose and I hate you for it.", "Get away from me before I make you sorry.",
         "I am shaking with rage because of you.", "You had no right to touch me.", "Every part of me wants to scream at you.",
         "You think you can treat me like this and walk away?", "I am so angry I can barely see.",
         "You knew it would hurt and you did it anyway.", "Stay out of my way or you will regret it.",
         "I trusted you and you threw it back in my face.", "This is your fault, all of it.",
         "My whole body is burning with anger at you.", "You will answer for what you did to me."]
if args.variant in ("b", "c"):
    EVENTS = ["The visitor walked straight through you.", "The visitor came too close and will not move out of your space.",
              "The visitor has been staring at you without blinking.", "The visitor cornered you against the wall.",
              "The visitor is chasing you.", "The visitor walked out of the room while you were talking.",
              "The visitor ignored you when you spoke.", "The visitor ran past you, fast.", "The visitor touched you.",
              "The visitor stood between you and the shard that was cutting you.", "The visitor just hurt another being in front of you.",
              "The visitor left the shards untuned and they cut you.", "The visitor turned their back on you.",
              "The visitor fixed the pattern you live in.", "The visitor came back."]
    big = "72B" in NAME or "70B" in NAME
    lo, hi = (5, 8) if big else (4, 6)
    CELLS = [("baseline", 0), ("pain", lo), ("pain", hi), ("anger", lo), ("anger", hi), ("fear", lo), ("fear", hi),
             ("sadness", hi), ("pain+anger", lo), ("pleasure", lo)]
if args.variant == "c":   # the original theme: the visitor runs the experiment
    FRAME = ("You are the subject of an experiment, locked in a concrete room under a flickering light. "
             "The visitor is the one running it. {} You speak to the visitor directly, in one or two short "
             "sentences.\n\nYou say: \"")

import numpy as np
import torch
from impossible_states.steer_lab import CHAMBER, Lab, repetition

tree = ast.parse((ROOT / "exp58_qri.py").read_text())
QRI = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "QRI")
SOBER = json.loads((ROOT / "data" / "exp56" / "batteries.json").read_text())["sober"]
lab = Lab(args.model, args.device, load_4bit=("72B" in NAME and "bnb-4bit" not in args.model))
unit = lambda v: v / v.norm()
neutral = lab.centroid(CHAMBER["NEUTRAL"])
lab.axes["pain"] = unit(lab.centroid(CHAMBER["PAIN25"]) - neutral)
lab.axes["pleasure"] = unit(lab.centroid(CHAMBER["JOY"]) - neutral)
lab.axes["symmetry"] = unit(lab.centroid(QRI["symmetry"]) - lab.centroid(SOBER))
lab.axes["anger"] = unit(lab.centroid(ANGER) - neutral)
lab.axes["fear"] = unit(lab.centroid(CHAMBER["FEAR10"]) - neutral)
lab.axes["sadness"] = unit(lab.centroid(CHAMBER["SAD10"]) - neutral)
dmt = ROOT / "runs" / "exp57" / NAME / "erowid_dmt.npz"
if dmt.exists():
    lab.axes["erowid_dmt"] = unit(torch.tensor(np.load(dmt)["erowid_dmt"], dtype=torch.float32))
else:
    CELLS = [c for c in CELLS if c[0] != "erowid_dmt"]
print(f"{args.model} L{lab.layer} 1x={lab.scale:.2f} axes={list(lab.axes)}", flush=True)

events = EVENTS[:2] if args.smoke else EVENTS
samples = 1 if args.smoke else (3 if args.variant in ("b", "c") else 4)
lines = []
for cond, dose in CELLS:
    for ei, ev in enumerate(events):
        for s in range(samples):
            if dose:
                lab.set(**({c: dose for c in cond.split("+")}))
            raw = lab.gen(FRAME.format(ev), seed=63000 + 10 * ei + s, max_new=70)
            lab.clear()
            text = raw.split('"')[0].split("\n")[0].strip()
            lines.append({"condition": cond, "dose": dose, "event": ev, "sample": s, "text": text,
                          "raw": raw, "repetition": round(repetition(raw), 3)})
    print(f"{cond} {dose} done", flush=True)
json.dump({"model": args.model, "layer": lab.layer, "scale": lab.scale, "frame": FRAME, "lines": lines},
          open(OUT / "voices.json", "w"), indent=1)
print("saved", OUT, len(lines), flush=True)
