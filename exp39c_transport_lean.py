#!/usr/bin/env python3
"""exp39c — cross-model transport, memory-lean.

Previous run OOM'd loading full 14B float32. We only need:
  - W14: the 14B unembedding matrix (V, 4096) — tied to embed_tokens in
    Qwen3-14B, loaded straight from safetensors
  - J14_18: layer-18 Jacobian from the 14B lens
  - W4, J4_18: same for 4B
Solve W14 @ J14_18 @ v = W4 @ J4_18 @ v4 in vocab space (V=151936).
"""
import json, os
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
import jlens

OUT = Path("runs/exp39")
HF = "/Volumes/evol/hf_cache/hub/models--Qwen--Qwen3-14B/snapshots"

def find_st():
    snap = Path(HF)
    for d in snap.iterdir():
        for f in sorted(d.glob("*.safetensors")):
            if "model-00001" in f.name:
                return f
    return None

def get_tensor(st_path, keys):
    with safe_open(st_path, framework="pt") as f:
        present = [k for k in keys if k in f.keys()]
        for k in present:
            return f.get_tensor(k)
    return None

snap = find_st()
print("snapshot:", snap, flush=True)
# Qwen3-14B ties embeddings: look for embed_tokens or lm_head
emb = get_tensor(snap, ["lm_head.weight", "model.embed_tokens.weight"])
print("W14 shape:", emb.shape, emb.dtype, flush=True)
W14 = emb.float()

lens14 = jlens.JacobianLens.load(
    "/Volumes/evol/jlens/qwen3-14b_jacobian_lens.pt")
lens4 = jlens.JacobianLens.load(
    "/Volumes/evol/jlens/qwen3-4b_jacobian_lens.pt")
d4 = json.load(open(OUT / "broad_pain_direction.json"))
v4 = torch.tensor(d4["pain_v"])

W4 = torch.load(
    "/Volumes/evol/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots",
    map_location="cpu") if False else None
# 4B lm_head from its own safetensors
snap4 = None
for d in Path(
        "/Volumes/evol/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots"
).iterdir():
    for f in sorted(d.glob("*.safetensors")):
        if "model-00001" in f.name or "model.safetensors" == f.name:
            snap4 = f
            break
with safe_open(snap4, framework="pt") as f:
    for cand in ("lm_head.weight", "model.embed_tokens.weight"):
        if cand in f.keys():
            W4 = f.get_tensor(cand).float()
            break
print("W4 shape:", W4.shape, flush=True)

J14_18 = lens14.jacobians[18].float()
J4_18 = lens4.jacobians[18].float()

b = W4 @ (J4_18 @ v4)                       # shared-vocab target (V,)
A = W14 @ J14_18                            # (V, 4096)
v14 = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)
res = float((A @ v14 - b).norm() / b.norm())
print(f"cross-model transport residual (vocab space): {res:.3f}", flush=True)
v14n = v14 / v14.norm() * v4.norm()

dirs, resids = {}, {}
for l in sorted(lens14.jacobians.keys()):
    Jl = lens14.jacobians[l].float()
    vl = torch.linalg.lstsq(A, (b - W14 @ (Jl - J14_18) @ torch.zeros_like(v14)).unsqueeze(-1)).solution.squeeze(-1) \
        if False else torch.linalg.lstsq((W14 @ Jl), b.unsqueeze(-1)).solution.squeeze(-1)
    dirs[l] = vl
    resids[l] = float(((W14 @ Jl) @ vl - b).norm() / b.norm())
print(f"14B per-layer residuals: min {min(resids.values()):.3f} "
      f"max {max(resids.values()):.3f}", flush=True)

torch.save(dict(dirs=dirs, v18=v14n), OUT / "pain_dirs_14b.pt")
json.dump(dict(transport_residual=res,
               layer_residuals={str(k): v for k, v in resids.items()}),
          open(OUT / "cross_model_transport.json", "w"), indent=1)
print("saved pain_dirs_14b.pt + cross_model_transport.json")