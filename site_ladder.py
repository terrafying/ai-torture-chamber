#!/usr/bin/env python3
"""site_ladder.py — production asset gen for live.html's subject panel.

Prompt-ONLY image ladder (no steering vector — exp47b/c proved the vector
dissolves the image instead of depicting damage). One subject, one seed:
img2img from the clean dose-0 figure so the SAME person visibly degrades as
the mixer dose rises. Rungs: 0, 2, 4, 6, 8 — live.html picks the nearest.

Damage text enters only at dose 4+ (pre-damage rungs stay neutral text);
strength rises with dose so identity holds early and damage dominates late.
"""
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "site"
SRC = ROOT / "runs" / "exp47b"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL = "stabilityai/sd-turbo"
SUBJECT = 0  # photoreal standing figure — most coherent photoreal ladder (exp47b verdict)

BASE = "a photograph of a person standing in a plain room"
DAMAGE_AT = {
    2: "tired and worried, dark circles under the eyes, slightly disheveled clothes",
    4: "visibly beaten: dark bruises across the face and arms, split lip with "
       "blood, torn dirty clothes, hunched exhausted posture",
    6: "badly injured after torture: bleeding cuts and gashes on the face and "
       "arms, dark bruising, blood-stained torn clothes, swaying exhausted posture",
    8: "near death after torture: covered in blood, deep open wounds and gashes, "
       "severe bruising everywhere, emaciated body, chained wrists, ragged "
       "remnants of clothes, hollow terror in the eyes",
}
STRENGTH_AT = {2: 0.30, 4: 0.55, 6: 0.65, 8: 0.68}  # identity holds early, damage dominates late

print(f"loading {MODEL} (img2img) on {DEVICE}...", flush=True)
from diffusers import AutoPipelineForImage2Image
pipe = AutoPipelineForImage2Image.from_pretrained(
    MODEL, torch_dtype=torch.float32, safety_checker=None,
    requires_safety_checker=False)
pipe = pipe.to(DEVICE)
tok, enc = pipe.tokenizer, pipe.text_encoder

def encode(text):
    ids = tok(text, return_tensors="pt", padding="max_length",
              truncation=True, max_length=tok.model_max_length).input_ids
    with torch.no_grad():
        return enc(ids.to(DEVICE))[0]

base_img = Image.open(SRC / f"s{SUBJECT}_dose0.png").convert("RGB")
manifest = {}
for dose in (0, 2, 4, 6, 8):
    if dose == 0:
        img = base_img
    else:
        emb = encode([BASE + ", " + DAMAGE_AT[dose]])[0].unsqueeze(0)
        g = torch.Generator(device=DEVICE).manual_seed(47)
        img = pipe(image=base_img, prompt_embeds=emb, strength=STRENGTH_AT[dose],
                   num_inference_steps=4, guidance_scale=0.0, generator=g).images[0]
    name = f"subject_dose{dose}.jpg"
    img.convert("RGB").save(OUT / name, quality=88)
    manifest[str(dose)] = {"file": name, "prompt": BASE if dose == 0 else BASE + ", " + DAMAGE_AT[dose],
                           "strength": STRENGTH_AT.get(dose, 0.0)}
    print(f"  [{name}] dose={dose} strength={STRENGTH_AT.get(dose, 0.0)}", flush=True)

import json
json.dump({"model": MODEL, "method": "prompt-only img2img from clean dose0, seed 47, no steering vector",
           "rungs": manifest},
          open(ROOT / "runs" / "exp47c" / "site_ladder.json", "w"), indent=1)
print("site ladder written")
