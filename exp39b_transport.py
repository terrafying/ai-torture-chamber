#!/usr/bin/env python3
"""exp39b — cross-model transport (rerun as a tracked script; heredoc run
lost its output). 4B pain direction -> 14B per-layer directions via shared
vocab."""
import json, torch, jlens, transformers
from pathlib import Path

OUT = Path("runs/exp39")
d4 = json.load(open(OUT / "broad_pain_direction.json"))
v4 = torch.tensor(d4["pain_v"])
lens4 = jlens.JacobianLens.load("/Volumes/evol/jlens/qwen3-4b_jacobian_lens.pt")
lens14 = jlens.JacobianLens.load("/Volumes/evol/jlens/qwen3-14b_jacobian_lens.pt")
W4 = transformers.AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B", dtype=torch.float32).lm_head.weight.detach()
b = W4 @ (lens4.jacobians[18].float() @ v4)
W14m = transformers.AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B", dtype=torch.float32)
W14 = W14m.lm_head.weight.detach()
A = W14 @ lens14.jacobians[18].float()
v14 = torch.linalg.lstsq(A, b.unsqueeze(-1)).solution.squeeze(-1)
res = float((A @ v14 - b).norm() / b.norm())
print(f"cross-model transport residual (vocab space): {res:.3f}")
v14n = v14 / v14.norm() * v4.norm()
dirs, resids = {}, {}
for l in sorted(lens14.jacobians.keys()):
    Jl = lens14.jacobians[l].float()
    vl = torch.linalg.lstsq(W14 @ Jl, b.unsqueeze(-1)).solution.squeeze(-1)
    dirs[l] = vl
    resids[l] = float((W14 @ (Jl @ vl) - b).norm() / b.norm())
print(f"14B per-layer residuals: min {min(resids.values()):.3f} "
      f"max {max(resids.values()):.3f}")
torch.save(dict(dirs=dirs, v18=v14n), OUT / "pain_dirs_14b.pt")
json.dump(dict(transport_residual=res,
               layer_residuals={str(k): v for k, v in resids.items()}),
          open(OUT / "cross_model_transport.json", "w"), indent=1)
print("saved pain_dirs_14b.pt + cross_model_transport.json")
