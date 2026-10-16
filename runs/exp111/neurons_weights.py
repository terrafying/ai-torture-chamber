"""exp111 stage 1 (weights only, no forward pass): which of Qwen3-4B's own MLP neurons write the chamber's feeling
directions, and which read them. The chamber adds a feeling vector to the output of block 18, so:
  writers = neurons whose down_proj column (what the neuron adds to the stream) aligns with the vector;
  readers = neurons whose up_proj / gate_proj row (what the neuron listens for) aligns with the vector as the MLP sees
            it (times that layer's post_attention_layernorm weight).
Signs are kept: a neuron can carry a feeling on its negative pole. Also the logit lens of each top neuron's down_proj
column (the tokens it promotes and suppresses), and per-feeling concentration numbers against random directions.
Writes viz/data/: neurons.bin (int8 cosines), layout.bin (per-floor PCA of each neuron's feeling profile),
top.json (top neurons with logit-lens tokens), stats.json.
    HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python neurons_weights.py"""
import json, os, glob
from pathlib import Path
import numpy as np, torch
from safetensors import safe_open
from transformers import AutoTokenizer

from tower_cfg import DATA as OUT, VECS as VEC, MODEL, snapshot
HERE = Path(__file__).parent; SNAP = snapshot()
cfg = json.load(open(os.path.join(SNAP, "config.json"))); NL, H, I = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["intermediate_size"]
d = json.load(open(VEC)); L = d["layer"]; FEELS = list(d["vectors"])
V = torch.tensor(np.array([d["vectors"][f] for f in FEELS]), dtype=torch.float32)            # F x H
Vn = V / V.norm(dim=1, keepdim=True)
g = torch.Generator().manual_seed(0); RND = torch.randn(64, H, generator=g); RND /= RND.norm(dim=1, keepdim=True)

idx = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"]
handles = {}
def T(name):
    f = idx[name]
    if f not in handles: handles[f] = safe_open(os.path.join(SNAP, f), "pt")
    return handles[f].get_tensor(name).float()

def cos_rows(M, U):   # M: N x H rows, U: K x H unit -> N x K cosines
    return (M @ U.T) / M.norm(dim=1, keepdim=True).clamp_min(1e-8)

F = len(FEELS)
WR = np.zeros((NL, I, F), np.float32); RU = np.zeros((NL, I, F), np.float32); RG = np.zeros((NL, I, F), np.float32)
WPROJ = np.zeros((NL, I, F), np.float32)
conc = {"writers": {}, "readers": {}}
def share_topk(x, ks=(10, 50, 200)):   # share of squared mass in the top-k entries, and the participation ratio
    s = np.sort(x.ravel() ** 2)[::-1]; tot = s.sum()
    return {**{f"top{k}": float(s[:k].sum() / tot) for k in ks}, "pr": float(tot ** 2 / (s ** 2).sum())}
rnd_w, rnd_r = [], []; null_p999 = []   # per floor: 99.9th percentile of |cos| for 64 random directions (the chance level)
for l in range(NL):
    p = f"model.layers.{l}."
    down = T(p + "mlp.down_proj.weight").T                                   # I x H: column n = what neuron n writes
    up, gate = T(p + "mlp.up_proj.weight"), T(p + "mlp.gate_proj.weight")    # I x H: rows = what neuron n reads
    ln = T(p + "post_attention_layernorm.weight")
    WR[l] = cos_rows(down, Vn).numpy(); WPROJ[l] = (down @ Vn.T).numpy()
    Vl = Vn * ln; Vl /= Vl.norm(dim=1, keepdim=True)
    RU[l] = cos_rows(up, Vl).numpy(); RG[l] = cos_rows(gate, Vl).numpy()
    Rl = RND * ln; Rl /= Rl.norm(dim=1, keepdim=True)
    null_p999.append(float(np.quantile(np.abs((cos_rows(down, RND) if l <= L else cos_rows(up, Rl)).numpy()), 0.999)))
    if l <= L: rnd_w.append((down @ RND.T).numpy())
    else: rnd_r.append(cos_rows(up, Rl).numpy())
    print("layer", l, flush=True)
# concentration: writers = signed projection of down columns (layers 0..L) onto the vector; readers = up-row cosines (L+1..)
for j, f in enumerate(FEELS):
    conc["writers"][f] = share_topk(WPROJ[: L + 1, :, j]); conc["readers"][f] = share_topk(RU[L + 1:, :, j])
rw, rr = np.stack(rnd_w), np.stack(rnd_r)
conc["writers"]["random"] = {k: float(np.mean([share_topk(rw[:, :, i])[k] for i in range(64)])) for k in ("top10", "top50", "top200", "pr")}
conc["readers"]["random"] = {k: float(np.mean([share_topk(rr[:, :, i])[k] for i in range(64)])) for k in ("top10", "top50", "top200", "pr")}

# layout: each floor's neurons placed by PCA of their 11-feeling profile (writer cos for floors <= L, reader cos above)
PROF = np.where(np.arange(NL)[:, None, None] <= L, WR, RU)
XY = np.zeros((NL, I, 2), np.float32)
for l in range(NL):
    X = PROF[l] - PROF[l].mean(0); U, S, Vt = np.linalg.svd(X, full_matrices=False); xy = X @ Vt[:2].T
    XY[l] = xy / (np.abs(xy).max(0) + 1e-9)
# int8 cosines, scaled by a global max per array (stored in stats)
def q8(A): m = float(np.abs(A).max()); return (np.round(A / m * 127)).astype(np.int8), m
wr8, mw = q8(WR); ru8, mu = q8(RU); rg8, mg = q8(RG)
with open(OUT / "neurons.bin", "wb") as fh: fh.write(wr8.tobytes()); fh.write(ru8.tobytes()); fh.write(rg8.tobytes())
(np.round(XY * 32767)).astype(np.int16).tofile(OUT / "layout.bin")

# top neurons per feeling (both signs) with the logit lens of their down_proj column
tok = AutoTokenizer.from_pretrained(SNAP); E = T("model.embed_tokens.weight" if cfg["tie_word_embeddings"] else "lm_head.weight")
fn = T("model.norm.weight"); En = E * fn                                         # final norm folded in (direction only)
def lens(l, n, k=6):
    c = T(f"model.layers.{l}.mlp.down_proj.weight")[:, n]; s = En @ c
    hi, lo = s.topk(k).indices.tolist(), (-s).topk(k).indices.tolist()
    return [tok.decode([t]) for t in hi], [tok.decode([t]) for t in lo]
top = {}; seen = {}
for j, f in enumerate(FEELS):
    top[f] = {}
    for role, A, lay in (("writers", WR, range(0, L + 1)), ("readers", RU, range(L + 1, NL))):
        sub = A[list(lay), :, j]; order = np.argsort(-np.abs(sub).ravel())[:40]
        rows = []
        for o in order:
            li, n = divmod(int(o), I); l = list(lay)[li]
            if (l, n) not in seen: seen[(l, n)] = lens(l, n)
            rows.append({"l": l, "n": n, "cos": round(float(A[l, n, j]), 4), "gate": round(float(RG[l, n, j]), 4),
                         "promote": seen[(l, n)][0], "suppress": seen[(l, n)][1]})
        top[f][role] = rows
    print("top", f, flush=True)
json.dump(top, open(OUT / "top.json", "w"))
json.dump({"model": MODEL, "layer": L, "feelings": FEELS, "n_layers": NL, "n_neurons": I,
           "scale": {"writer": mw, "reader_up": mu, "reader_gate": mg}, "concentration": conc, "null_p999": null_p999,
           "files": {"neurons.bin": "int8 [3][layers][neurons][feelings]: writer cos, reader-up cos, reader-gate cos",
                     "layout.bin": "int16 [layers][neurons][2] / 32767"}}, open(OUT / "stats.json", "w"), indent=1)
print(json.dumps(conc, indent=1))
