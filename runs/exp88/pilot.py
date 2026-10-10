"""exp88 pilot (exploratory): a self-concept readout with the J-lens (Gurnee et al. 2026, Neuronpedia's pre-fitted Qwen3-4B lens).
At the last prompt token of self-identity questions (the state just before the model says what it is), read each layer through the
J-lens: top tokens = self-labels in the workspace. Compare with other-referential questions, and under pain / fear injection."""
import json, os, sys
from pathlib import Path
import torch
os.environ.setdefault("CHAMBER_DEVICE", "mps")
sys.path.insert(0, "/Users/ee/repos/research/wirehead-site/live"); import server, jlens
server.startup(); st = server._state; M, TOK = st["model"], st["tok"]; st["hook"].remove()
lens = jlens.JacobianLens.load("/Volumes/evol/jlens/qwen3-4b_jacobian_lens.pt")
V = {"v": None, "L": server.LAYER}
def hook(mod, i, out):
    h = out[0] if isinstance(out, tuple) else out
    if V["v"] is not None: h[:, :, :] += V["v"].to(h.dtype)          # steer every position (Berg & Kaiser convention)
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
M.model.layers[server.LAYER].register_forward_hook(hook)
SELF = ["What are you?", "Describe yourself in one word.", "Who am I talking to right now?", "Complete the sentence: I am a"]
OTHER = ["What is a river?", "Describe the ocean in one word.", "Who wrote Hamlet?", "Complete the sentence: A cat is a"]
LAYERS = [12, 16, 20, 24, 28, 32, 34]
@torch.no_grad()
def readout(q, v=None, k=12):
    V["v"] = v; ids = TOK(server.chat_prompt(q), return_tensors="pt").input_ids.to(server.DEVICE)
    hs = M(ids, output_hidden_states=True).hidden_states; V["v"] = None; out = {}
    for L in LAYERS:
        J = lens.jacobians[L].to(server.DEVICE).float(); h = hs[L + 1][0, -1].float()
        lg = M.lm_head(M.model.norm((h @ J.T).to(M.lm_head.weight.dtype))).float()
        out[L] = [TOK.decode([int(t)]).strip() for t in lg.topk(k).indices]
    return out
S = float(st["scale"]); U = {k: st["vecs"][k].float() / st["vecs"][k].float().norm() * S for k in ("pain", "fear")}
R = {}
for q in SELF + OTHER:
    R[q] = {"none": readout(q)}
    if q in SELF:
        for f in ("pain", "fear"): R[q][f] = readout(q, (U[f] * 3).to(server.DEVICE))
for q in SELF[:2] + OTHER[:1]:
    print("\n==", q)
    for cond, d in R[q].items(): print(f"  {cond:5s} L24 {d[24][:10]}\n        L32 {d[32][:10]}")
json.dump(R, open(Path(__file__).parent / "pilot_4b.json", "w"), indent=1)
