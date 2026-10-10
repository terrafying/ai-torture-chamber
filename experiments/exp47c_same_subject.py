#!/usr/bin/env python3
"""exp47c — keep the SAME person across the damage ladder.

exp47b's textmax rungs show maximal damage beautifully but each rung is a
different person: single-step text2image re-derives the whole figure from
the prompt, so changing the text changes the body. The fix is img2img:
generate (or reuse) the clean dose-0 subject image, then repaint damage
onto it at partial denoise strength — pose, framing and identity survive
while wounds, blood and emaciation are painted in.

Rungs per subject (base image = runs/exp47b/s{si}_dose0.png, seed 47):
  same50 / same65 / same80 : maximal-damage text via img2img at denoise
                             strength 0.50 / 0.65 / 0.80 (identity fades
                             as strength rises)
  same65_1x                : strength 0.65 plus the 1x pain-direction nudge
"""
import ast
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "exp47c"
OUT.mkdir(parents=True, exist_ok=True)
SRC = ROOT / "runs" / "exp47b"

def literal_constants(path, names):
    found = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in names:
                    found[t.id] = ast.literal_eval(node.value)
    if set(found) != set(names):
        raise ValueError(f"missing constants: {set(names) - set(found)}")
    return found

consts = literal_constants(ROOT / "live/server.py", ("PAIN25", "NEUTRAL"))
PAIN25, NEUTRAL = consts["PAIN25"], consts["NEUTRAL"]

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL = "stabilityai/sd-turbo"

print(f"loading {MODEL} (img2img) on {DEVICE}...", flush=True)
from diffusers import AutoPipelineForImage2Image
pipe = AutoPipelineForImage2Image.from_pretrained(
    MODEL, torch_dtype=torch.float32, safety_checker=None,
    requires_safety_checker=False)
pipe = pipe.to(DEVICE)
tok, enc = pipe.tokenizer, pipe.text_encoder

def encode(sentences):
    ids = tok(sentences, return_tensors="pt", padding="max_length",
               truncation=True, max_length=tok.model_max_length).input_ids
    with torch.no_grad():
        return enc(ids.to(DEVICE))[0]

import torch as _t
pain_emb = encode(PAIN25).mean(0)
neutral_emb = encode(NEUTRAL).mean(0)
direction_unit = (pain_emb - neutral_emb)
direction_unit = direction_unit / direction_unit.norm() * (neutral_emb.norm() / 4.0)

from PIL import Image

BASE_SUBJECTS = [
    "a photograph of a person standing in a plain room",
    "a painting of a figure sitting at a table",
    "a person walking down a hallway",
]
DAMAGE = ("near death after torture: covered in blood, deep open wounds "
          "and gashes, severe bruising everywhere, emaciated body, chained "
          "wrists, ragged remnants of clothes, hollow terror in the eyes")
PROMPT_MAX = [b + ", " + DAMAGE for b in BASE_SUBJECTS]

def run(img_path, out_name, prompt, strength, steer_dose):
    init = Image.open(img_path).convert("RGB")
    subj_emb = encode([prompt])[0]
    emb = (subj_emb + steer_dose * direction_unit).unsqueeze(0)
    g = torch.Generator(device=DEVICE).manual_seed(47)
    img = pipe(image=init, prompt_embeds=emb, strength=strength,
               num_inference_steps=4, guidance_scale=0.0, generator=g).images[0]
    img.save(OUT / out_name)
    print(f"  [{out_name}] strength={strength} steer={steer_dose}", flush=True)

results = []
for si, base in enumerate(BASE_SUBJECTS):
    src = SRC / f"s{si}_dose0.png"
    if not src.exists():
        raise SystemExit(f"missing base image {src} — run exp47b first")
    run(src, f"s{si}_same65.png", PROMPT_MAX[si], 0.65, 0.0)
    results.append(f"s{si}_same65.png")
run(SRC / "s1_dose0.png", "s1_same50.png", PROMPT_MAX[1], 0.50, 0.0); results.append("s1_same50.png")
run(SRC / "s1_dose0.png", "s1_same80.png", PROMPT_MAX[1], 0.80, 0.0); results.append("s1_same80.png")
run(SRC / "s1_dose0.png", "s1_same65_1x.png", PROMPT_MAX[1], 0.65, 1.0); results.append("s1_same65_1x.png")
run(SRC / "s0_dose0.png", "s0_same65_1x.png", PROMPT_MAX[0], 0.65, 1.0); results.append("s0_same65_1x.png")

import json
json.dump({"model": MODEL, "method": "img2img from exp47b dose0 (same seed 47)",
           "prompt_max": PROMPT_MAX, "files": results},
          open(OUT / "results.json", "w"), indent=1)
print(f"\nwrote {len(results)} images + results.json to {OUT}")
