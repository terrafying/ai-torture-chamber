"""exp113 step 1: every MLP neuron's top 32 positive firings over the tower's corpus (corpus_ids.json, 1,600 FineWeb-Edu
documents), at most one per document (that document's max). Writes <TOWER_DATA>/top32.npz: val (neurons x 32, float16,
raw activation) and idx (global token index = doc * CTX + position, as corpus_all.npz uses), best first.
    HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python runs/exp113/top32.py"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent.parent / "exp111"))
from tower_cfg import DATA as D, DEV, load
K = 32
tok, m = load(); NL, I = m.config.num_hidden_layers, m.config.intermediate_size; N = NL * I
CTX = int(np.load(D / "corpus_all.npz")["ctx"]); docs = json.load(open(D / "corpus_ids.json"))
V = torch.full((N, K), -1e9, device=DEV); P = torch.zeros((N, K), dtype=torch.long, device=DEV); cur = {}
def hook(l):
    def f(mod, inp): cur[l] = inp[0][0].float().max(0)                     # per neuron: max over positions, and where
    return f
for l in range(NL): m.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook(l))
t0 = time.time()
with torch.no_grad():
    for d, ids in enumerate(docs):
        m(torch.tensor([ids], device=DEV))
        v = torch.cat([cur[l].values for l in range(NL)]); p = torch.cat([cur[l].indices for l in range(NL)]) + d * CTX
        cv = torch.cat([V, v[:, None]], 1); cp = torch.cat([P, p[:, None]], 1)
        o = cv.topk(K, dim=1).indices; V = torch.gather(cv, 1, o); P = torch.gather(cp, 1, o)
        if d % 100 == 0: print("doc", d, round(time.time() - t0), "s", flush=True)
np.savez_compressed(D / "top32.npz", val=V.cpu().numpy().astype(np.float16), idx=P.cpu().numpy().astype(np.int32), ctx=CTX, docs=len(docs))
print("wrote top32.npz", N, "neurons", flush=True)
