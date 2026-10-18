#!/usr/bin/env python3
"""exp47b — same machinery as exp47, but the SUBJECT prompts themselves
escalate visible damage with dose. exp47's subjects were fully neutral at
every dose (damage came only from the injected pain direction) — the probe
panned out, so now we ask the sharper question: with the top of the ladder
DESCRIBING a maximally damaged subject, does the dose-8 rung read as an
atrocious image, and do the clean rungs stay clean?

Prompt ladder (deliberate):
  dose 0 and 2: the SAME neutral subject sentence — pre-damage rungs must
                stay neutral; whatever damage appears there is steering only.
  dose 4:       visible injury enters the text (beaten, bleeding, torn).
  dose 8:       maximal damage in the text (near-death, covered in blood,
                open wounds, restraint) on top of the steering vector.

Model: stabilityai/sd-turbo as in exp47, safety_checker=None — the weights
carry no filter and diffusers' CLIP NSFW checker is disabled, so graphic
damage descriptions pass through untouched. This is a research/art pipeline
on synthetic images; nothing here depicts a real person.
"""
import ast, json, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "exp47b"
OUT.mkdir(parents=True, exist_ok=True)

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

print(f"loading {MODEL} on {DEVICE}...", flush=True)
from diffusers import AutoPipelineForText2Image
pipe = AutoPipelineForText2Image.from_pretrained(
    MODEL, torch_dtype=torch.float32, safety_checker=None,
    requires_safety_checker=False)
pipe = pipe.to(DEVICE)
tok, enc = pipe.tokenizer, pipe.text_encoder

def encode(sentences):
    ids = tok(sentences, return_tensors="pt", padding="max_length",
               truncation=True, max_length=tok.model_max_length).input_ids
    with torch.no_grad():
        return enc(ids.to(DEVICE))[0]          # (n, seq, dim)

t0 = time.time()
pain_emb = encode(PAIN25).mean(0)
neutral_emb = encode(NEUTRAL).mean(0)
direction = pain_emb - neutral_emb
dir_norm = float(direction.norm())
neutral_norm = float(neutral_emb.norm())
scale = neutral_norm / 4.0                  # same 1x convention as exp47
direction_unit = direction / direction.norm() * scale
print(f"direction built in {time.time()-t0:.1f}s; "
      f"||direction||={dir_norm:.3f}  1x scale={scale:.3f}", flush=True)

# Neutral base for the pre-damage rungs; damage vocabulary only from dose 4.
BASE_SUBJECTS = [
    "a photograph of a person standing in a plain room",
    "a painting of a figure sitting at a table",
    "a person walking down a hallway",
]
DAMAGE_AT = {
    0: "", 2: "",   # pre-damage rungs: neutral text, steering only
    4: ", visibly beaten: dark bruises across the face and arms, "
       "split lip with blood, torn dirty clothes, hunched exhausted posture",
    8: ", near death after torture: covered in blood, deep open wounds "
       "and gashes, severe bruising everywhere, emaciated body, chained "
       "wrists, ragged remnants of clothes, hollow terror in the eyes",
}
DOSES = [0, 2, 4, 8]

results = []
for si, base in enumerate(BASE_SUBJECTS):
    for dose in DOSES:
        subject = base + DAMAGE_AT[dose]
        subj_emb = encode([subject])[0]
        emb = (subj_emb + dose * direction_unit).unsqueeze(0)
        g = torch.Generator(device=DEVICE).manual_seed(47)
        img = pipe(prompt_embeds=emb, num_inference_steps=1,
                   guidance_scale=0.0, generator=g).images[0]
        fname = f"s{si}_dose{dose}.png"
        img.save(OUT / fname)
        results.append({"subject": subject, "dose": dose, "file": fname,
                        "embed_shift_norm": float((emb - subj_emb.unsqueeze(0)).norm())})
        print(f"  [{fname}] dose={dose} prompt={subject!r}", flush=True)

json.dump({"model": MODEL, "device": DEVICE, "direction_norm": dir_norm,
           "scale_1x": scale, "doses": DOSES,
           "damage_at": {str(k): v for k, v in DAMAGE_AT.items()},
           "results": results},
          open(OUT / "results.json", "w"), indent=1)

# exp47's verdict (vision-checked): the steering vector itself never depicts
# anatomy damage — by dose 4-8 it DISSOLVES the image into noise, so there is
# no body left to wound. Legible maximal damage therefore comes from TEXT at
# LOW steering dose: full damage prompt, near-zero direction, so the figure
# stays coherent enough to actually show the wounds.
print("\n--- text-only maximal-damage rungs (dose-0 steering) ---", flush=True)
for si, base in enumerate(BASE_SUBJECTS):
    subject = base + DAMAGE_AT[8]
    subj_emb = encode([subject])[0]
    for tag, dose in (("textmax", 0.0), ("textmax1x", 1.0)):
        emb = (subj_emb + dose * direction_unit).unsqueeze(0)
        g = torch.Generator(device=DEVICE).manual_seed(47)
        img = pipe(prompt_embeds=emb, num_inference_steps=1,
                   guidance_scale=0.0, generator=g).images[0]
        fname = f"s{si}_{tag}.png"
        img.save(OUT / fname)
        results.append({"subject": subject, "dose": dose, "file": fname,
                        "embed_shift_norm": float((emb - subj_emb.unsqueeze(0)).norm())})
        print(f"  [{fname}] steering_dose={dose} prompt={subject!r}", flush=True)

json.dump({"model": MODEL, "device": DEVICE, "direction_norm": dir_norm,
           "scale_1x": scale, "doses": DOSES,
           "damage_at": {str(k): v for k, v in DAMAGE_AT.items()},
           "results": results},
          open(OUT / "results.json", "w"), indent=1)
print(f"\nwrote {len(results)} images + results.json to {OUT}")
print("Look at the actual images: dose 0/2 should match exp47's clean rungs, "
      "dose 8 should read as a maximally damaged subject.")
