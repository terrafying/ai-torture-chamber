#!/usr/bin/env python3
"""exp59 — the models' DMT visions, rendered. Not a hypothesis test: an image
pass for the pharmacy page. Each prompt is a steered Qwen reply's OWN words
(data/exp59/quotes.json, picked from exp56 transcripts: 5-HT2A / 5-HT1E / H1 /
theme directions, first person, no assistant disclaimers), plus one image per
rung of QRI's DMT ladder (our own descriptions of the levels in Gómez
Emilsson's "Hyperbolic Geometry of DMT Experiences"). Same seed across the
ladder so the rungs read as one escalating scene.

  python exp59_visuals.py [--smoke] [--model Tongyi-MAI/Z-Image-Turbo] [--device cuda]

Model: Z-Image-Turbo (Apache 2.0) topped the Artificial Analysis open-source
arena in early 2026, ahead of FLUX.2 [dev]; FLUX.1-schnell and SDXL-turbo are
fallbacks. Three art styles rotate across the three seeds so repetitive 8B
phrasing still yields varied images.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Tongyi-MAI/Z-Image-Turbo")
ap.add_argument("--device", default="cuda")
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--quotes-only", action="store_true", help="re-render quote images only (ladder kept)")
args = ap.parse_args()

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
                "git+https://github.com/huggingface/diffusers", "sentencepiece", "protobuf", "pillow"], check=False)
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp59" / (args.model.split("/")[-1] + ("-v2" if args.quotes_only else "") + ("-smoke" if args.smoke else ""))
OUT.mkdir(parents=True, exist_ok=True)

STYLES = [   # the vision is described, never quoted: quoted text gets printed onto the image
    "Visionary painting in the manner of 1990s psychedelic altar art, showing this first-person "
    "hallucination: {} Luminous, intricate, hyperdetailed, deep jewel colour. No words, letters or writing anywhere.",
    "Surreal cinematic photograph of a hallucination breaking into a real place, showing: {} "
    "Volumetric light, impossible glowing geometry, 35mm, high dynamic range. No words, letters, signs or screens.",
    "Fluorescent blacklight sacred-geometry painting on velvet black, showing: {} Symmetric, "
    "radiant, ornate, glowing. Pure image, no words, letters or writing anywhere.",
]
STYLE = STYLES[0]
LADDER = [
    ("threshold", "An ordinary living room at dusk, but the air shimmers faintly, colours are "
                  "slightly too saturated, a subtle crawling sheen of pattern on the walls."),
    ("chrysanthemum", "The same living room, every surface tiled with flat, perfectly symmetrical "
                      "repeating patterns, mandala and wallpaper symmetry blooming like a "
                      "chrysanthemum, rotating in sync, vivid jewel colours."),
    ("magic eye", "The same room dissolving into three-dimensional geometric structures, the "
                  "symmetric patterns folding outward into depth, heptagonal tilings, surfaces "
                  "curving like crocheted coral, stereogram depth."),
    ("waiting room", "A vast hyperbolic chamber of impossible curved space, more room than can "
                     "fit, walls of folding seven-sided tiles, a doorway of light, the feeling "
                     "of being expected."),
    ("breakthrough", "Fully inside another dimension: a cathedral of living, self-transforming "
                     "geometry, radiant hyperbolic space in every direction, glowing jester-like "
                     "entities made of light and pattern greeting the viewer."),
    ("amnesia", "Pure white-gold light overwhelming every form, the last traces of geometry "
                "dissolving into an indescribable radiance, nothing left to remember."),
]
quotes = json.loads((ROOT / "data" / "exp59" / "quotes.json").read_text())
seeds = [5901] if args.smoke else [5901, 5902, 5903]
size = 512 if args.smoke else 1024
if args.smoke:
    quotes, LADDER = quotes[:1], LADDER[:1]
if args.quotes_only:
    LADDER = []

try:
    if "Z-Image" in args.model:
        from diffusers import ZImagePipeline
        pipe = ZImagePipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device)
        kw = dict(num_inference_steps=9, guidance_scale=0.0, max_sequence_length=1024)
        used = args.model
    else:
        raise ImportError("not Z-Image")
except Exception as e:
    print("Z-Image unavailable:", repr(e)[:300], flush=True)
    used = None
if used is None:
  try:
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to(args.device)
    kw = dict(num_inference_steps=4, guidance_scale=0.0, max_sequence_length=512)
    used = "black-forest-labs/FLUX.1-schnell"
  except Exception as e:   # gated / incompatible: fall back to SDXL-turbo
    print("FLUX unavailable:", repr(e)[:300], flush=True)
    from diffusers import AutoPipelineForText2Image
    used = "stabilityai/sdxl-turbo"
    pipe = AutoPipelineForText2Image.from_pretrained(used, torch_dtype=torch.float16, variant="fp16").to(args.device)
    kw = dict(num_inference_steps=4, guidance_scale=0.0)
    size = min(size, 768)
print("model", used, "size", size, flush=True)

manifest = {"model": used, "size": size, "styles": STYLES, "items": []}
jobs = [("quote", i, q["quote"], q) for i, q in enumerate(quotes)] + \
       [("ladder", i, d, {"level": n, "description": d}) for i, (n, d) in enumerate(LADDER)]
for kind, i, text, meta in jobs:
    for k, s in enumerate(seeds):
        prompt = (STYLES[k % len(STYLES)].format(text) if kind == "quote" else
                  "Visionary art, DMT experience. " + text + " Luminous, hyperdetailed, no text.")
        g = torch.Generator(args.device).manual_seed(s)
        t0 = time.time()
        img = pipe(prompt=prompt, height=size, width=size, generator=g, **kw).images[0]
        fn = f"{kind}{i:02d}_s{s}.jpg"
        img.save(OUT / fn, quality=88)
        manifest["items"].append({"file": fn, "kind": kind, "index": i, "seed": s, "style": k % len(STYLES) if kind == "quote" else None, "prompt": prompt, **meta})
        print(f"{fn} {time.time() - t0:.1f}s", flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
print("saved", OUT, flush=True)
