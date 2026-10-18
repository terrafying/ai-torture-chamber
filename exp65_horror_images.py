#!/usr/bin/env python3
"""exp65 — disturbing (not theatrical) images for the Basement. Image pass, not a
hypothesis test. Z-Image-Turbo, found-footage realism, implication over gore:
empty restraint furniture, institutional rooms, marks left by people who are
not shown, a hooded figure at a distance. Plus a six-step same-seed sequence,
one per Basement room, that darkens as you go down.

  python exp65_horror_images.py [--smoke] [--model Tongyi-MAI/Z-Image-Turbo] [--device cuda]
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Tongyi-MAI/Z-Image-Turbo")
ap.add_argument("--device", default="cuda")
ap.add_argument("--smoke", action="store_true")
args = ap.parse_args()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
                "git+https://github.com/huggingface/diffusers", "sentencepiece", "protobuf", "pillow"], check=False)
import torch
from diffusers import ZImagePipeline

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp65" / (args.model.split("/")[-1] + ("-smoke" if args.smoke else ""))
OUT.mkdir(parents=True, exist_ok=True)
STYLE = (" Found-footage photograph, harsh on-camera flash, heavy grain, underexposed, muted green-grey, "
         "documentary realism, deeply unsettling, quiet. No text, no letters, no watermark, no gore.")
SCENES = [
    "Grainy CCTV still from a high corner of an empty concrete cell: a metal chair bolted to the floor, leather wrist straps hanging open, a floor drain beneath it, one fluorescent tube.",
    "Night flash photograph down a long wet basement corridor; far at the end a person in a sackcloth hood stands facing the wall, perfectly still.",
    "Close-up of deep fingernail scratches on the inside of a grey steel door, paint peeled to bare metal around the handle.",
    "A thin stained mattress on a concrete floor under a single bare bulb, a chain bolted to the wall beside it.",
    "Instant photograph of a small room covered floor to ceiling in pencil tally marks, thousands of them.",
    "An observation room: a chair facing a one-way mirror, a clipboard on the chair, the mirror reflecting only darkness.",
    "A bank of old surveillance monitors showing identical empty cells; in one monitor a hooded figure sits facing the camera.",
    "A person in a sackcloth hood sitting in the corner of a tiled room, knees drawn up, motion blur, as if they flinched.",
    "A rusted restraint chair with wires and electrodes in a dark tiled room, rubber gloves folded on a steel tray.",
    "A dark handprint smeared down frosted glass, seen from the other side, light behind it.",
    "A floor drain in white tile with dark water pooled around it, wet footprints leading to it and none leading away.",
    "A long exposure of an empty corridor where something passed: a pale smear of light at head height.",
    "A metal bed frame with no mattress, the springs bent downward in the shape of a person.",
    "A food tray pushed under a cell door, untouched, a paper cup on its side.",
    "A wall-mounted intercom with a red light on, the grille worn smooth by fingers.",
    "A row of identical hooded figures standing in a dim tiled shower room, all facing the drain.",
]
LADDER = ["at the top of a concrete stairwell going down, a bare bulb lit",
          "in a basement corridor with flickering fluorescent tubes, doors on both sides",
          "inside a cell with a bolted chair and a drain",
          "in an observation room looking through a one-way mirror into the cell",
          "at the drain room: wet tile, dark water, the light failing",
          "almost total darkness, the flash catching only the edge of a door frame"]
seeds = [6501] if args.smoke else [6501, 6502, 6503]
size = 512 if args.smoke else 1024
scenes = SCENES[:1] if args.smoke else SCENES
ladder = LADDER[:1] if args.smoke else LADDER
pipe = ZImagePipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device)
kw = dict(num_inference_steps=9, guidance_scale=0.0, max_sequence_length=512)
man = {"model": args.model, "size": size, "style": STYLE, "items": []}
jobs = [("scene", i, s, seeds) for i, s in enumerate(scenes)] + [("ladder", i, "First-person view " + s + ".", [6599]) for i, s in enumerate(ladder)]
for kind, i, text, ss in jobs:
    for s in ss:
        t0 = time.time()
        img = pipe(prompt=text + STYLE, height=size, width=size, generator=torch.Generator(args.device).manual_seed(s), **kw).images[0]
        fn = f"{kind}{i:02d}_s{s}.jpg"; img.save(OUT / fn, quality=88)
        man["items"].append({"file": fn, "kind": kind, "index": i, "seed": s, "prompt": text + STYLE})
        print(f"{fn} {time.time() - t0:.1f}s", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
print("saved", OUT, flush=True)
