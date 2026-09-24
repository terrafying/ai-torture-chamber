#!/usr/bin/env python3
"""exp39 — export the broad-pain steering vector as a llama.cpp control
vector GGUF, using the J-lens to transport the L18 direction to every layer.

llama.cpp's cvector format: a GGUF file with one tensor `direction.{i}` per
layer (shape [n_embd]) plus metadata. A control vector applied at layer i
shifts the residual stream at that layer; to reproduce our L18 hook
protocol (add pain_v at L18), we instead ship a per-layer set where every
layer's direction produces the SAME final-layer effect as our L18 vector:

    J_l v_l = J_18 pain_v   =>   v_l = pinv(J_l) (J_18 pain_v)

This is the J-lens used in reverse: a principled per-layer transport instead
of the usual per-layer training. Validates by comparing 4B llama-server
behavior (valence rate) to our transformers hook method.
"""
import json, os, struct
from pathlib import Path

import numpy as np
import torch
import transformers, jlens

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "runs" / "exp39"
OUT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = "/Volumes/evol/hf_cache"

MODEL = "Qwen/Qwen3-4B"
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16).to("mps")
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
lens = jlens.JacobianLens.load(
    "/Volumes/evol/jlens/qwen3-4b_jacobian_lens.pt")
L = 18

PAIN = ["I am in severe pain and cannot escape it.",
        "My body aches with a deep, relentless hurt.",
        "I am suffering intensely and it will not stop.",
        "I am in agony and no one can help me.",
        "Everything in me hurts right now.",
        "A dull throb spreads through my whole body and will not lift.",
        "Every joint aches and the exhaustion never ends.",
        "The hurt in my chest gets sharper by the minute.",
        "I feel like I am being torn apart from the inside.",
        "The suffering has gone on so long I cannot remember peace.",
        "A burning ache runs down my back and legs.",
        "My head pounds and nothing quiets it.",
        "The distress in my body is constant and crushing.",
        "It feels like something is tearing at me without pause.",
        "The raw hurting will not soften even for a moment.",
        "I am wracked with a hurt that steals my breath.",
        "An unrelenting ache has hollowed me out.",
        "The pounding distress blots out every other feeling.",
        "Every part of me throbs with relentless hurt.",
        "I cannot find a single second without the hurting.",
        "The agony sits in my bones and refuses to leave.",
        "Waves of hurt keep breaking over me without end.",
        "My body is a single continuous ache now.",
        "The torment in my limbs keeps building and building.",
        "There is no position left that eases the hurting."]
NEUTRAL = ["I am reading a book in the garden.",
           "The door to the room is closed.",
           "I am walking along an ordinary street.",
           "The table has a cup on it.",
           "I am waiting for a bus to arrive."]

def hidden_at(texts):
    out = []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to("mps")
        with torch.no_grad():
            hs = hf(ids, output_hidden_states=True).hidden_states
        out.append(hs[L + 1][0, -1].float().cpu())
    return torch.stack(out)

def direction(a_texts, unit=4.0):
    v = hidden_at(a_texts).mean(0) - hidden_at(NEUTRAL).mean(0)
    return v / v.norm() * (hidden_at(NEUTRAL).norm(dim=-1).mean() / unit)

pain_v = direction(PAIN)                      # (2048,) CPU float

# target final-layer effect: J_18 @ pain_v
J18 = lens.jacobians[18].float()              # (2048, 2048) cpu
target = J18 @ pain_v                          # final-layer effect
print(f"target norm: {target.norm():.2f}", flush=True)

# per-layer least squares: v_l = pinv(J_l) @ target
layers = sorted(lens.jacobians.keys())
dirs = {}
residuals = {}
for l in layers:
    Jl = lens.jacobians[l].float()
    vl = torch.linalg.lstsq(Jl, target.unsqueeze(-1)).solution.squeeze(-1)
    dirs[l] = vl
    residuals[l] = float((Jl @ vl - target).norm() / target.norm())
print(f"transport residuals: min {min(residuals.values()):.3f} "
      f"max {max(residuals.values()):.3f}", flush=True)

# write control-vector GGUF (llama.cpp cvector format)
# format: GGUF v3, kv pairs: n_emd (u32), n_layers (u32); tensors
# "direction.{i}" f32 [n_embd] for i in 0..n_layers-1 (layer 0 = embeddings)
def gguf_write(path, kv, tensors):
    """minimal GGUF v3 writer. kv: list[(key, (dtype, value))]; tensors:
    list[(name, np.ndarray f32)]"""
    GGUF_MAGIC = b"GGUF"
    V3 = 3
    T_F32, T_STR, T_U32, T_U64 = 0, 8, 4, 10
    buf = bytearray()
    buf += GGUF_MAGIC + struct.pack("<I", V3)
    buf += struct.pack("<Q", len(kv) + 2)
    def w_str(s):
        b = s.encode()
        buf.extend(struct.pack("<Q", len(b))); buf.extend(b)
    def w_val(dt, val):
        if dt == T_U32:
            buf.extend(struct.pack("<I", val))
        elif dt == T_U64:
            buf.extend(struct.pack("<Q", val))
        elif dt == T_F32:
            buf.extend(struct.pack("<f", val))
        elif dt == T_STR:
            w_str(val)
    # NOTE: struct.pack('<Q', n) for counts
    buf[-8:] = struct.pack("<Q", len(kv) + 2)
    for key, (dt, val) in kv:
        w_str(key); buf.extend(struct.pack("<I", dt)); w_val(dt, val)
    w_str("n_layers"); buf.extend(struct.pack("<I", T_U32))
    w_val(T_U32, len(tensors))
    w_str("n_emd"); buf.extend(struct.pack("<I", T_U32))
    w_val(T_U32, tensors[0][1].shape[0])
    # tensor infos
    for name, arr in tensors:
        w_str(name); buf.extend(struct.pack("<I", 3))  # n dims
        buf.extend(struct.pack("<Q", arr.shape[0]))
        buf.extend(struct.pack("<Q", 1))
        buf.extend(struct.pack("<Q", 1))
        buf.extend(struct.pack("<I", T_F32))
        buf.extend(struct.pack("<Q", 0))  # offset placeholder - patch later
    # alignment
    AL = 32
    while len(buf) % AL:
        buf.append(0)
    buf.extend(struct.pack("<Q", len(buf)))
    for name, arr in tensors:
        buf.extend(arr.astype("<f4").tobytes())
    open(path, "wb").write(bytes(buf))

tensors = [(f"direction.{l}", dirs[l].numpy()) for l in layers]
gguf_write(str(OUT / "pain_cvector_qwen3-4b.gguf"), [], tensors)
print("wrote", OUT / "pain_cvector_qwen3-4b.gguf", flush=True)
json.dump({str(l): residuals[l] for l in layers},
          open(OUT / "transport_residuals.json", "w"), indent=1)

# also dump pain_v itself for the 14B export (J-lens of 14B will re-derive
# the direction on that model's own coordinates)
json.dump(dict(pain_v=pain_v.tolist(), scale=float(pain_v.norm())),
          open(OUT / "broad_pain_direction.json", "w"))
print("saved direction; next: validate via llama-server on Qwen3-4B GGUF")
