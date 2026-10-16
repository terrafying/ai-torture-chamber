"""exp111: the tower's wiring from the weights alone (no forward pass). For neuron A on floor i and neuron B on a higher
floor j, the direct link is how much of what A writes (its down_proj column, at its typical full firing) lands in what
B listens for (its gate_proj / up_proj rows, times floor j's post_attention_layernorm weight). It ignores attention and
the norm's division by the stream size, so it is the skeleton, not the live traffic.
Restricted to the "active" neurons: every neuron that is ever in a stream frame's top 150, a circuit node, or labelled.
  check  how well this predicts the causal edges of the act x grad circuits (circuits.json): AUC of |link| for each
         circuit edge against random pairs on the same two floors, per variant (gate, up, both; cosine or scaled).
  build  top-K upstream and downstream links per active neuron -> connectome.npz (idx, up_i, up_w, dn_i, dn_w).
    HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python neurons_connectome.py check|build [K]"""
import json, os, sys
import numpy as np, torch
from safetensors import safe_open
from tower_cfg import DATA as D, snapshot
SNAP = snapshot(); cfg = json.load(open(os.path.join(SNAP, "config.json")))
NL, H, I = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["intermediate_size"]
MODE = sys.argv[1] if len(sys.argv) > 1 else "check"; K = int(sys.argv[2]) if len(sys.argv) > 2 else 24
wm = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"]; hs = {}
def T(name):
    f = wm[name]
    if f not in hs: hs[f] = safe_open(os.path.join(SNAP, f), "pt")
    return hs[f].get_tensor(name).float()
C = dict(np.load(D / "corpus_all.npz")); SCALE = np.maximum(np.abs(C["max"]), np.abs(C["min"])).clip(1e-3)

act = set()
st = json.load(open(D / "stream.json"))
for p in st["prompts"].values():
    for c in p["conds"].values():
        for f in c["frames"]: act.update(f["i"][:150])
circ = json.load(open(D / "circuits.json"))
for g in circ: act.update(x["l"] * I + x["n"] for x in g["nodes"])
act.update(int(k.split(":")[0]) * I + int(k.split(":")[1]) for k in json.load(open(D / "labels.json")))
idx = np.array(sorted(act)); lay = idx // I; neu = idx % I; N = len(idx); print("active neurons", N, flush=True)

W = torch.zeros(N, H); RG = torch.zeros(N, H); RU = torch.zeros(N, H)       # writes, gate reads, up reads
for l in range(NL):
    s = np.where(lay == l)[0]
    if not len(s): continue
    ln = T(f"model.layers.{l}.post_attention_layernorm.weight")
    W[s] = T(f"model.layers.{l}.mlp.down_proj.weight")[:, neu[s]].T * torch.tensor(SCALE[l, neu[s]])[:, None]
    RG[s] = T(f"model.layers.{l}.mlp.gate_proj.weight")[neu[s]] * ln
    RU[s] = T(f"model.layers.{l}.mlp.up_proj.weight")[neu[s]] * ln
unit = lambda X: X / X.norm(dim=1, keepdim=True).clamp(min=1e-9)
Wn, RGn, RUn = unit(W), unit(RG), unit(RU)
pos = {int(g): k for k, g in enumerate(idx)}
LAY = torch.tensor(lay)

def link(a, b, v):   # a, b: index arrays into the active set
    if v == "gate_cos": return (Wn[a] * RGn[b]).sum(1)
    if v == "up_cos": return (Wn[a] * RUn[b]).sum(1)
    if v == "both_cos": return ((Wn[a] * RGn[b]).sum(1) ** 2 + (Wn[a] * RUn[b]).sum(1) ** 2).sqrt()
    if v == "gate_raw": return (W[a] * RG[b]).sum(1)
    if v == "both_raw": return ((W[a] * RG[b]).sum(1) ** 2 + (W[a] * RU[b]).sum(1) ** 2).sqrt()

if MODE == "check":
    from scipy.stats import mannwhitneyu
    ea, eb, ew, same = [], [], [], []
    for g in circ:
        for a, b, w in g["edges"]:
            A, B = g["nodes"][a], g["nodes"][b]
            if A["l"] >= B["l"]: A, B = B, A
            if A["l"] == B["l"]: continue
            ea.append(pos[A["l"] * I + A["n"]]); eb.append(pos[B["l"] * I + B["n"]]); ew.append(abs(w)); same.append(A["p"] == B["p"])
    ea, eb, ew, same = np.array(ea), np.array(eb), np.array(ew), np.array(same); rng = np.random.default_rng(0)
    byl = {l: np.where(lay == l)[0] for l in range(NL)}
    ra = np.array([rng.choice(byl[lay[a]]) for a in ea]); rb = np.array([rng.choice(byl[lay[b]]) for b in eb])
    out = {"edges": int(len(ea)), "same_token_edges": int(same.sum())}
    for v in ("gate_cos", "up_cos", "both_cos", "gate_raw", "both_raw"):
        e = link(ea, eb, v).abs().numpy(); r = link(ra, rb, v).abs().numpy()
        auc = mannwhitneyu(e, r).statistic / (len(e) * len(r))
        top = ew >= np.quantile(ew, 0.9)
        auc_top = mannwhitneyu(e[top], r).statistic / (top.sum() * len(r))
        auc_same = mannwhitneyu(e[same], r).statistic / (same.sum() * len(r)) if same.any() else None
        out[v] = {"auc": round(float(auc), 3), "auc_top10pct_edges": round(float(auc_top), 3), "auc_same_token": auc_same and round(float(auc_same), 3),
                  "median_edge": float(np.median(e)), "median_random": float(np.median(r))}
        print(v, out[v], flush=True)
    json.dump(out, open(D / "connectome_check.json", "w"), indent=1)
else:   # build: top-K upstream per neuron (rows) and downstream per neuron (columns), signed both_raw-style via gate+up
    up_i = np.zeros((N, K), np.int32); up_w = np.zeros((N, K), np.float32)
    dn_v = torch.full((N, K), -1.0); dn_i = torch.zeros((N, K), dtype=torch.long); dn_s = torch.zeros(N, K)
    for s in range(0, N, 2048):
        b = torch.arange(s, min(N, s + 2048))
        G = RGn[b] @ Wn.T; U = RUn[b] @ Wn.T                                   # B x A
        S = (G ** 2 + U ** 2).sqrt(); sign = torch.sign(G * U)                # SwiGLU: gate*up sign = sign of the effect on B
        S[LAY[None, :] >= LAY[b][:, None]] = -1.0                             # only links from lower floors
        v, i = S.topk(K, dim=1); up_i[b] = i.numpy(); up_w[b] = (v * torch.gather(sign, 1, i)).numpy()
        vt, it = S.topk(min(K, len(b)), dim=0)                               # best receivers in this chunk, per sender
        cand_v = torch.cat([dn_v, vt.T], 1); cand_i = torch.cat([dn_i, b[it].T], 1); cand_s = torch.cat([dn_s, torch.gather(sign, 0, it).T], 1)
        o = cand_v.topk(K, dim=1).indices
        dn_v = torch.gather(cand_v, 1, o); dn_i = torch.gather(cand_i, 1, o); dn_s = torch.gather(cand_s, 1, o)
        print("rows", s, flush=True)
    np.savez_compressed(D / "connectome.npz", idx=idx, up_i=up_i, up_w=up_w, dn_i=dn_i.numpy().astype(np.int32),
                        dn_w=(dn_v * dn_s).numpy().astype(np.float32))
    print("wrote connectome.npz", N, "neurons x", K)
