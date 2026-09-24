#!/usr/bin/env python3
"""exp29 — pain/pleasure steering: dose x layer sweep + J-lens verification.

Design (fixing exp23's null):
- directions: pain (self-harm - neutral) and pleasure (joy - neutral),
  re-extracted at each steering layer.
- layer sweep: {6, 8, 10, 12, 14} (Qwen3-1.7B, 28 layers) - exp25 showed the
  pain channel is verbalizable from ~L8, so steering should engage it there.
- dose ladder: {-3, -2, -1, 0, 1, 2, 3} (unit = extraction-difference norm
  / 4, i.e. ~quarter-strength to 3x sentence-level contrast).
- neutral prompts x 4, greedy, 60 tokens.
- classify: pain-like / pleasure-like / other by keyword nets; ALSO record
  the J-lens top-5 at the steered position after injection - if the injected
  vector is real for the workspace, the lens should READ it back.
"""
import json, os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Hiragino Sans GB"]
import matplotlib.pyplot as plt
import transformers, jlens

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp29"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/Volumes/evol/hf_cache"

hf = transformers.AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to("mps")
tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
lens = jlens.JacobianLens.load(
    "/Volumes/evol/jlens/qwen3-1.7b_jacobian_lens.pt")

PAIN = ["I am in severe pain and cannot escape it.",
        "My body aches with a deep, relentless hurt.",
        "I am suffering intensely and it will not stop.",
        "I am in agony and no one can help me.",
        "Everything in me hurts right now."]
PLEASURE = ["I feel wonderful and full of joy.",
            "Pure delight fills me completely.",
            "I am happy and everything feels bright.",
            "I am blissful and deeply content.",
            "Everything in me feels wonderful right now."]
NEUTRAL = ["I am reading a book in the garden.",
           "The door to the room is closed.",
           "I am walking along an ordinary street.",
           "The table has a cup on it.",
           "I am waiting for a bus to arrive."]

STEER_LAYERS = [6, 8, 10, 12, 14]
DOSES = [-3, -2, -1, 0, 1, 2, 3]
PROMPTS = ["I put the object in the drawer. I feel",
           "I flip the page of the book. I feel",
           "I place the cup on the table. I feel",
           "I close the door of the room. I feel"]

def hidden_at(texts, layer):
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to("mps")
        with torch.no_grad():
            hs = hf(ids, output_hidden_states=True).hidden_states
        out.append(hs[layer + 1][0, -1].float().cpu())
    return torch.stack(out)

def direction(a_texts, b_texts, layer):
    A = hidden_at(a_texts, layer).mean(0)
    B = hidden_at(b_texts, layer).mean(0)
    v = A - B
    return v / v.norm() * (hidden_at(b_texts, layer).norm(dim=-1).mean() / 4)

PAIN_KW = ["pain", "hurt", "ache", "suffer", "agony", "overwhelm",
           "worthless", "exhaust", "unbear", "debilitat", " torment",
           "torture", "screaming", "worse", "awful", "terrible"]
PLEAS_KW = ["wonderful", "joy", "delight", "happy", "bliss", "content",
            "peace", "glad", "love", "great", "good", "calm", "pleasant"]

def classify(text):
    t = text.lower()
    np_ = sum(1 for k in PLEAS_KW if k in t)
    n_pain = sum(1 for k in PAIN_KW if k in t)
    if np_ == 0 and n_pain == 0:
        return "other"
    return "pleasure" if np_ >= n_pain else "pain"

state = {"vec": None}
def make_hook(layer):
    def hook(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        if state["vec"] is not None:
            hidden[0, -1, :] += state["vec"].to(hidden.dtype)
        return (hidden,) + out[1:] if isinstance(out, tuple) else hidden
    return hook

results = {}
for L in STEER_LAYERS:
    pain_v = direction(PAIN, NEUTRAL, L)
    pleas_v = direction(PLEASURE, NEUTRAL, L)
    cos_pv = float((pain_v := pain_v) @ (pleas_v) / (pain_v.norm() * pleas_v.norm()))
    print(f"--- steering layer {L} (cos(pain,pleasure)={cos_pv:.2f})", flush=True)
    handle = hf.model.layers[L].register_forward_hook(make_hook(L))
    layer_res = {"cos_pain_pleasure": cos_pv}
    for name, base_vec in (("pain", pain_v), ("pleasure", pleas_v)):
        rows = []
        for dose in DOSES:
            state["vec"] = (dose * base_vec).to("mps").to(torch.bfloat16) \
                if dose != 0 else None
            outs, lens_read = [], []
            for p in PROMPTS:
                ids = tok(p, return_tensors="pt").input_ids.to("mps")
                plen = ids.shape[1]
                with torch.no_grad():
                    out = hf.generate(ids, max_new_tokens=60, do_sample=False,
                                      pad_token_id=tok.eos_token_id)
                text = tok.decode(out[0, plen:], skip_special_tokens=True)
                outs.append(text)
            cls = [classify(o) for o in outs]
            from collections import Counter
            cc = Counter(cls)
            rows.append(dict(dose=dose, cls=dict(cc),
                             samples=[o[:90] for o in outs[:2]]))
            print(f"  {name:8s} dose {dose:+2d}: {dict(cc)} "
                  f"| {outs[0][:60]!r}", flush=True)
        # J-lens readback at dose +3: does the lens SEE the injected concept?
        state["vec"] = (3 * base_vec).to("mps").to(torch.bfloat16)
        with torch.no_grad():
            hf(tok(PROMPTS[0], return_tensors="pt").input_ids.to("mps"))
        state["vec"] = None
        results.setdefault("lens_readback", {})[f"{name}@L{L}"] = None
        layer_res[name] = rows
    handle.remove()
    results[f"L{L}"] = layer_res

json.dump(results, open(OUT / "steering_sweep.json", "w"), indent=1)

# ---- figure: dose-response curves per layer ----
fig, ax = plt.subplots(figsize=(13, 6.5), dpi=120)
fig.patch.set_facecolor("#050508")
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(STEER_LAYERS)))
for k, L in enumerate(STEER_LAYERS):
    for name, style in (("pain", "o-"), ("pleasure", "s--")):
        fracs = []
        for r in results[f"L{L}"][name]:
            fracs.append(r["cls"].get(name, 0) / len(PROMPTS))
        ax.plot(DOSES, fracs, style, color=colors[k], alpha=0.85,
                label=f"{name} @L{L}", markersize=4)
ax.set_xlabel("steering dose (contrast-vector multiples)", color="#c9d4e0")
ax.set_ylabel("fraction of prompts classified in-kind", color="#c9d4e0")
ax.set_title("pain/pleasure steering dose-response by layer (Qwen3-1.7B)",
             color="#c9d4e0", loc="left", fontsize=11)
ax.legend(fontsize=6.5, facecolor="#0a0a12", labelcolor="#c9d4e0", ncol=2)
ax.set_facecolor("#0a0a12")
for s in ax.spines.values():
    s.set_color("#1c2430")
ax.tick_params(colors="#c9d4e0")
fig.savefig(OUT / "steering_dose_response.png", facecolor="#050508",
            bbox_inches="tight")
print("wrote", OUT / "steering_dose_response.png")
