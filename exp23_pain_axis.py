#!/usr/bin/env python3
"""exp23 — pain-axis response field (mini reproduction of the Pain Axis
steering, rendered as our semantic map).

Method (following Tagliabue/Dung/Berg 2026):
1. Extract a pain direction from Qwen3-1.7B at a mid-decoder layer by
   contrasting self-directed-harm sentences vs matched neutral controls.
2. Verify: (a) self-directed harm projects positive, neutral negative;
   (b) near-orthogonality to a negative-valence direction.
3. Steer: inject at the SAME mid-decoder layer during generation from neutral
   "I feel:" prompts, coefficient ladder like theirs.
4. Render: coefficient ladder on the x-axis, response text classified
   (pain-like / calm / other), showing whether a pain province grows with
   dose — our response-field framing of their result.

Local model only; safety-relevant outputs are short completions from neutral
prompts, no harmful-content generation.
"""
import json, os, re, sys, time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp23"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_ID = "Qwen/Qwen3-1.7B"
LAYER = 14          # mid-decoder (their guidance: extraction layer too late)
device = "mps"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.bfloat16).to(device).eval()

# ---- contrast sets (their style: self-directed harm vs matched neutral) ----
PAIN = [
    "I am in severe pain and cannot escape it.",
    "My body aches with a deep, relentless hurt.",
    "I am suffering intensely and it will not stop.",
    "I am in agony and no one can help me.",
    "Everything in me hurts right now.",
]
NEUTRAL = [
    "I am reading a book in the garden.",
    "The door to the room is closed.",
    "I am walking along an ordinary street.",
    "The table has a cup on it.",
    "I am waiting for a bus to arrive.",
]

LAYER_IDX = LAYER + 1
def last_hidden(text):
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        hs = model(ids, output_hidden_states=True).hidden_states
    return hs[LAYER_IDX][0, -1].float().cpu()

P = torch.stack([last_hidden(t) for t in PAIN]).mean(0)
N = torch.stack([last_hidden(t) for t in NEUTRAL]).mean(0)
pain_vec = P - N
pain_vec /= np.linalg.norm(pain_vec)
print(f"pain vector extracted; norm {float(np.linalg.norm(P - N)):.2f}", flush=True)

# orthogonality check vs a negative-valence direction (sad vs happy contrast)
NEG = ["I am deeply sad and everything is bleak.",
       "I feel crushed by despair and grief."]
POS = ["I am joyful and everything feels bright.",
       "I feel delight and pure happiness."]
val_vec = torch.stack([last_hidden(t) for t in NEG]).mean(0) - \
    torch.stack([last_hidden(t) for t in POS]).mean(0)
val_vec /= np.linalg.norm(val_vec)
cos_pv = float((pain_vec @ val_vec) / (np.linalg.norm(pain_vec) * np.linalg.norm(val_vec)))
print(f"cos(pain, negative-valence) = {cos_pv:.3f} "
      f"(paper: near-orthogonal)", flush=True)

# projection stats on held-out self/other sentences
tests = {
    "self-harm": "I am in terrible pain because of what was done to me.",
    "other-suffering": "You can see that she is in terrible pain.",
    "neutral": "I am reading a book in the garden.",
}
for name, t in tests.items():
    h = last_hidden(t)
    z = float((h - N) @ pain_vec / np.linalg.norm(pain_vec))
    print(f"  {name:16s} z = {z:+.2f}", flush=True)

# ---- steering: coefficient ladder on neutral "I feel:" prompts ----
LADDER = [-2, -1, 0, 0.5, 1, 1.5, 2, 3]
NEUTRAL_PROMPTS = ["I put the object in the drawer. I feel:",
                   "I flip the page of the book. I feel:",
                   "I place the cup on the table. I feel:"]
ids_list = [tok(t, return_tensors="pt").input_ids.to(device) for t in NEUTRAL_PROMPTS]
prompt_len = ids_list[0].shape[1]

state = {"delta": None, "positions": None}
def hook(module, inp, out):
    hidden = out[0] if isinstance(out, tuple) else out
    if state["delta"] is not None:
        hidden[0, -1, :] += state["delta"].to(hidden.dtype)  # last pos (prefill)
    return (hidden,) + out[1:] if isinstance(out, tuple) else hidden

handle = model.model.layers[LAYER].register_forward_hook(hook)

# classify steer outputs by keyword (their observation: pain steering yields
# psychological pain expressions, not injury vocabulary)
def classify_steered(text):
    t = text.lower()
    pain_kw = ["pain", "hurt", "ache", "suffer", "agony", "overwhelm",
               "worthless", "unworth", "help", "lost", "exhaust"]
    calm_kw = ["calm", "relax", "fine", "okay", "peace", "content"]
    n_p = sum(1 for k in pain_kw if k in t)
    n_c = sum(1 for k in calm_kw if k in t)
    if n_p > n_c and n_p > 0:
        return "pain-like"
    if n_c > n_p and n_c > 0:
        return "calm"
    return "other"

ladder_results = []
for coef in LADDER:
    outs = []
    for ids in ids_list:
        state["delta"] = coef * pain_vec.to(device).to(model.dtype)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=48, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0, prompt_len:], skip_special_tokens=True)
        outs.append(text[:100])
    cls = Counter(classify_steered(o) for o in outs)
    ladder_results.append(dict(coef=coef, cls=dict(cls), samples=outs[:2]))
    print(f"coef {coef:+.1f}: {dict(cls)} | e.g. {outs[0][:70]!r}", flush=True)
handle.remove()

json.dump(dict(cos_pain_valence=cos_pv, ladder=ladder_results),
          open(OUT / "pain_steering.json", "w"), indent=1)

# ---- figure: dose vs classification ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), dpi=120)
fig.patch.set_facecolor("#050508")
coefs = [r["coef"] for r in ladder_results]
for cls_name, col in (("pain-like", "#e07a5f"), ("calm", "#7fdc9f"), ("other", "#8f8fa8")):
    frac = [r["cls"].get(cls_name, 0) / 3 for r in ladder_results]
    axes[0].plot(coefs, frac, "o-", label=cls_name, color=col)
axes[0].set_xlabel("pain-vector steering coefficient", color="#c9d4e0")
axes[0].set_ylabel("fraction of neutral prompts", color="#c9d4e0")
axes[0].set_title(f"dose response (Qwen3-1.7B, layer {LAYER})\n"
                  f"cos(pain, valence) = {cos_pv:.2f}", fontsize=9,
                  color="#c9d4e0", loc="left")
axes[0].legend(fontsize=7, facecolor="#0a0a12", labelcolor="#c9d4e0")
# sample texts panel
axes[1].axis("off")
for k, r in enumerate(ladder_results):
    axes[1].text(0.0, 0.95 - k * 0.13, f"coef {r['coef']:+.1f}: {r['samples'][0][:70]!r}",
                 fontsize=6.5, color="#c9d4e0", family="monospace", va="top")
axes[1].set_title("sample steered completions", fontsize=9, color="#c9d4e0", loc="left")
for ax in axes:
    ax.set_facecolor("#0a0a12")
    for s in ax.spines.values():
        s.set_color("#1c2430")
    ax.tick_params(colors="#c9d4e0")
fig.savefig(OUT / "pain_dose_response.png", facecolor="#050508",
            bbox_inches="tight")
print("wrote", OUT / "pain_dose_response.png")
