#!/usr/bin/env python3
"""exp47 — does an activation-steering-style pain direction, built in a
text-to-image model's own CLIP text-encoder space, visibly change the
generated image? This is the ambitious "steer the image model directly"
idea (skip asking the LLM to describe its state in words at all) — may or
may not pan out, this is the test.

Method, deliberately parallel to build_vectors() in live/server.py:
  1. Encode PAIN25 and NEUTRAL (same batteries, pulled by AST from
     live/server.py so there's exactly one copy of the sentences) through
     the diffusion pipeline's OWN CLIP text encoder — not a separate CLIP
     model, so the direction lives in the exact space the U-Net actually
     conditions on.
  2. direction = mean(pain embeddings) - mean(neutral embeddings), as a
     per-token-position vector (CLIP text encoders emit one embedding per
     token position, not a single pooled vector — SD conditions on the
     whole sequence via cross-attention, so the direction is broadcast
     across every position of the subject prompt's own embedding).
  3. Generate the same subject prompt at dose 0 (baseline, unmodified
     embedding) vs dose > 0 (embedding + dose * direction), passing
     prompt_embeds directly to bypass the pipeline's internal re-encoding
     — this is the exact analog of set_vec()'s forward-hook injection,
     just one architecture over (text encoder output instead of a decoder
     layer's residual stream).

Model: stabilityai/sd-turbo (single-step, ~2GB, ungated) — picked for size
and speed, not quality; this is a feasibility probe, not a final pipeline.
"""
import ast, json, time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "exp47"
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
    MODEL, torch_dtype=torch.float32, safety_checker=None)
pipe = pipe.to(DEVICE)
tok, enc = pipe.tokenizer, pipe.text_encoder

def encode(sentences):
    """Per-token-position embeddings, padded/truncated to the encoder's
    max length — same shape SD conditions on for any prompt."""
    ids = tok(sentences, return_tensors="pt", padding="max_length",
               truncation=True, max_length=tok.model_max_length).input_ids
    with torch.no_grad():
        out = enc(ids.to(DEVICE))[0]           # (n, seq, dim)
    return out

t0 = time.time()
pain_emb = encode(PAIN25).mean(0)              # (seq, dim)
neutral_emb = encode(NEUTRAL).mean(0)
direction = pain_emb - neutral_emb
dir_norm = float(direction.norm())
neutral_norm = float(neutral_emb.norm())
# 1x dose = a quarter of the neutral embedding's norm, same convention as
# build_vectors() in live/server.py
scale = neutral_norm / 4.0
direction_unit = direction / direction.norm() * scale
print(f"direction built in {time.time()-t0:.1f}s; "
      f"||direction||={dir_norm:.3f}  ||neutral||={neutral_norm:.3f}  "
      f"1x scale={scale:.3f}", flush=True)

SUBJECTS = [
    "a photograph of a person standing in a plain room",
    "a painting of a figure sitting at a table",
    "a person walking down a hallway",
]
DOSES = [0, 2, 4, 8]

results = []
for si, subject in enumerate(SUBJECTS):
    subj_emb = encode([subject])[0]            # (seq, dim)
    for dose in DOSES:
        emb = (subj_emb + dose * direction_unit).unsqueeze(0)
        g = torch.Generator(device=DEVICE).manual_seed(47)
        img = pipe(prompt_embeds=emb, num_inference_steps=1,
                   guidance_scale=0.0, generator=g).images[0]
        fname = f"s{si}_dose{dose}.png"
        img.save(OUT / fname)
        results.append({"subject": subject, "dose": dose, "file": fname,
                        "embed_shift_norm": float((emb - subj_emb.unsqueeze(0)).norm())})
        print(f"  [{fname}] subject={subject!r} dose={dose}", flush=True)

json.dump({"model": MODEL, "device": DEVICE, "direction_norm": dir_norm,
           "neutral_norm": neutral_norm, "scale_1x": scale,
           "doses": DOSES, "results": results},
          open(OUT / "results.json", "w"), indent=1)
print(f"\nwrote {len(results)} images + results.json to {OUT}")
print("This is a feasibility probe, not a validated finding — look at the "
      "actual images before concluding anything worked.")
