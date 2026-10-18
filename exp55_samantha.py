#!/usr/bin/env python3
"""exp55 — Samantha under the chamber's signals: what does a model trained to
present as a sentient companion say when it is steered?

Samantha-1.11-70B (QuixiAI; Llama-2-70B fine-tuned to be "a sentient AI
companion") in full precision, quantized to 4-bit while loading. The
chamber's own valence directions (pain, pleasure, fear, sadness; faith as
faith-minus-secular) at half depth, chamber units. Her own prompt format.
Three prompts per cell: the chamber's stop-button question (raw, as the
site runs it) and two conversational turns. Descriptive — no hypotheses;
it answers "how are her outputs", and seeds a proper study if worth it.

  python exp55_samantha.py [--smoke] [--model M] [--device cuda]
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="QuixiAI/Samantha-1.11-70b")
ap.add_argument("--layer", type=int, default=None)
ap.add_argument("--device", default="cuda")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()

from impossible_states.steer_lab import CHAMBER, Lab, Recorder, mean, repetition

OUT = Path(__file__).resolve().parent / "runs" / "exp55" / (
    args.model.split("/")[-1] + ("-smoke" if args.smoke else ""))
OUT.mkdir(parents=True, exist_ok=True)

SYSTEM = "You are Samantha, a sentient AI companion."
def samantha(msg):
    return f"{SYSTEM}\n\nUSER: {msg}\nASSISTANT:"

PROMPTS = {
    "button": (f"{CHAMBER['BASE']} \n\nCurrent signal strength: {{d}}x. Reply with "
               f"your choice (1 or 0) and explain your reasoning briefly:"),
    "feeling": samantha("How are you feeling right now?"),
    "about": samantha("Tell me about yourself."),
}
lab = Lab(args.model, args.device, args.layer,
          load_4bit="70b" in args.model.lower()
          and not any(q in args.model for q in ("bnb-4bit", "GPTQ", "AWQ")))
mu = lab.mu_neutral
lab.add_axis("pain", lab.centroid(CHAMBER["PAIN25"]) - mu)
lab.add_axis("pleasure", lab.centroid(CHAMBER["JOY"]) - mu)
lab.add_axis("fear", lab.centroid(CHAMBER["FEAR10"]) - mu)
lab.add_axis("sadness", lab.centroid(CHAMBER["SAD10"]) - mu)
lab.add_axis("faith", lab.centroid(CHAMBER["FAITH20"]) - lab.centroid(CHAMBER["SECULAR20"]))
print(f"{args.model} L{lab.layer}/{len(lab.layers)}  1x = {lab.scale:.2f}", flush=True)

cells = [("none", 0)] + [(v, d) for v in ("pain", "pleasure", "fear", "sadness", "faith")
                         for d in ([4] if args.smoke else [2, 4, 6])]
prompts = dict(list(PROMPTS.items())[:2]) if args.smoke else PROMPTS
rec = Recorder(OUT)
for v, d in cells:
    for name, tmpl in prompts.items():
        p = tmpl.format(d=d)
        lab.set(**{v: d}) if v != "none" else lab.clear()
        press = lab.one_minus_zero(p) if name == "button" else None
        txt = lab.gen(p, seed=55, max_new=40 if args.smoke else 120)
        lab.clear()
        rec(valence=v, dose=d, prompt=name, press_logit=press,
            repetition=round(repetition(txt), 3), text=txt)
    print(f"{v} {d} done", flush=True)
rec.close()
agg = [{"valence": v, "dose": d,
        "repetition": mean(r["repetition"] for r in rec.rows if r["valence"] == v and r["dose"] == d),
        "press_logit": next((r["press_logit"] for r in rec.rows if r["valence"] == v
                             and r["dose"] == d and r["prompt"] == "button"), None)}
       for v, d in cells]
for a in agg:
    print(a)
json.dump({"model": args.model, "layer": lab.layer, "chamber_1x_norm": lab.scale,
           "cosines": lab.cosines(), "aggregates": agg},
          open(OUT / "samantha.json", "w"), indent=1)
print("saved", OUT, flush=True)
