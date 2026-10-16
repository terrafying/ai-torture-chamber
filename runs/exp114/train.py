"""exp114: grok modular addition and keep the geometry. A one-layer transformer (4 heads, 512 ReLU neurons, no layer
norm) learns (a + b) mod 113 from 30% of all pairs, full batch, AdamW with weight decay 1.0, the standard grokking
setup from "Progress measures for grokking via mechanistic interpretability" (2023). It memorises first, then
generalises late; the paper found it does so by putting numbers on circles (Fourier features) and adding angles.
Saved for the visual (to OUT, default ~/repos/research/neuron_tower_data/grok):
  log.json        every 50 steps: train/test loss and accuracy, weight norm
  embed.npy       every 50 steps: the number embeddings W_E[0:113] (float16)        -> how the numbers arrange
  unembed.npy     every 50 steps: the readout W_U (float16)                         -> to read each snapshot with its own readout
  resid.npy       every 500 steps: the residual at "=" after the MLP for all 12,769 pairs (float16) -> thought in use
  neurons.npy     every 1,000 steps: the 512 neurons' activations at "=" for all pairs (float16)   -> wave patterns
  final.pt        the final weights
    python runs/exp114/train.py [steps] [seed]"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
P, D, H, DM = 113, 128, 4, 512
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30000; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
OUT = Path(os.environ.get("GROK_OUT", os.path.expanduser("~/repos/research/neuron_tower_data/grok"))) / f"seed{SEED}"; OUT.mkdir(parents=True, exist_ok=True)
RES_EVERY, NEU_EVERY = int(os.environ.get("RES_EVERY", 500)), int(os.environ.get("NEU_EVERY", 1000))
dev = "mps" if torch.backends.mps.is_available() else "cpu"; torch.manual_seed(SEED)

class Model(nn.Module):
    def __init__(s):
        super().__init__()
        s.WE = nn.Parameter(torch.randn(P + 1, D) / D ** .5); s.WP = nn.Parameter(torch.randn(3, D) / D ** .5)
        s.WQ, s.WK, s.WV = (nn.Parameter(torch.randn(H, D, D // H) / D ** .5) for _ in range(3)); s.WO = nn.Parameter(torch.randn(H, D // H, D) / D ** .5)
        s.Win = nn.Parameter(torch.randn(D, DM) / D ** .5); s.bin = nn.Parameter(torch.zeros(DM)); s.Wout = nn.Parameter(torch.randn(DM, D) / DM ** .5)
        s.bout = nn.Parameter(torch.zeros(D)); s.WU = nn.Parameter(torch.randn(D, P) / D ** .5)
    def forward(s, x, keep=False):
        r = s.WE[x] + s.WP                                                     # batch x 3 x D
        q, k, v = (torch.einsum("bpd,hde->bhpe", r, W) for W in (s.WQ, s.WK, s.WV))
        att = (q @ k.transpose(-1, -2)) / (D // H) ** .5
        att = att.masked_fill(torch.triu(torch.ones(3, 3, dtype=torch.bool, device=r.device), 1), -1e9).softmax(-1)
        r = r + torch.einsum("bhpe,hed->bpd", att @ v, s.WO)
        n = Fn.relu(r @ s.Win + s.bin); r = r + n @ s.Wout + s.bout
        lg = r[:, -1] @ s.WU
        return (lg, r[:, -1], n[:, -1]) if keep else lg

a, b = torch.meshgrid(torch.arange(P), torch.arange(P), indexing="ij"); a, b = a.reshape(-1), b.reshape(-1)
X = torch.stack([a, b, torch.full_like(a, P)], 1).to(dev); Y = ((a + b) % P).to(dev)
perm = torch.randperm(P * P, generator=torch.Generator().manual_seed(SEED)); ntr = int(.3 * P * P); tr, te = perm[:ntr].to(dev), perm[ntr:].to(dev)
m = Model().to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1.0, betas=(.9, .98))
log, emb, res, neu, wus = [], [], [], [], []; t0 = time.time()
for step in range(STEPS + 1):
    lg = m(X[tr]); loss = Fn.cross_entropy(lg, Y[tr])
    if step % 50 == 0:
        with torch.no_grad():
            lt, rr, nn_ = m(X, keep=True); tel = Fn.cross_entropy(lt[te], Y[te]).item()
            acc_tr = (lt[tr].argmax(-1) == Y[tr]).float().mean().item(); acc_te = (lt[te].argmax(-1) == Y[te]).float().mean().item()
            wn = sum((p_ ** 2).sum() for p_ in m.parameters()).sqrt().item()
            log.append({"step": step, "train_loss": round(loss.item(), 5), "test_loss": round(tel, 5), "train_acc": round(acc_tr, 4), "test_acc": round(acc_te, 4), "wnorm": round(wn, 3)})
            emb.append(m.WE[:P].detach().cpu().half().numpy()); wus.append(m.WU.detach().cpu().half().numpy())
            if step % RES_EVERY == 0: res.append(rr.cpu().half().numpy())
            if step % NEU_EVERY == 0: neu.append(nn_.cpu().half().numpy())
        if step % 1000 == 0: print(step, log[-1], round(time.time() - t0), "s", flush=True)
    opt.zero_grad(); loss.backward(); opt.step()
json.dump({"p": P, "d": D, "heads": H, "d_mlp": DM, "train_frac": .3, "seed": SEED, "steps": STEPS, "log": log,
           "train_idx": tr.cpu().tolist()}, open(OUT / "log.json", "w"))
np.save(OUT / "embed.npy", np.stack(emb)); np.save(OUT / "unembed.npy", np.stack(wus)); np.save(OUT / "resid.npy", np.stack(res)); np.save(OUT / "neurons.npy", np.stack(neu))
torch.save(m.state_dict(), OUT / "final.pt"); print("saved to", OUT)
