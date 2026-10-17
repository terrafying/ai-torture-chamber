"""Which attention heads pull a word toward the meaning of an earlier word, and which push it away? Qwen3-4B, the
sky_flight sentences. Follows sky_steer.py, where hiding w->v removed about half of w's specific pull toward v but
boosting w->v x4 (floors 12-24, all heads) LOWERED it.

  attr  (one forward per sentence, plus the boost reruns of phase 0)
    payload  c_h = sum over v's tokens k of a[h, w, k] * W_O,h v_kv(h),k : what head h moves from v into w (w's last
             sub-token), at each floor l. Projections onto u_v = unit(neutral(v) - neutral(w)) and onto the mean of
             u_u over v's group-mates not in the sentence; score = (c . u_v) - mean_u (c . u_u)  (> 0: toward v itself
             rather than its kind). Stored per floor x head x pair, with |c|.
    phase 0  is the boost reversal repulsion or "category over identity"? (a) head-summed payload per floor projected
             on u_v and on the mates separately; (b) the all-head boost (x4, floors 12-24, w->v) rerun keeping
             cos(d, u_v) and mean cos(d, u_mates) apart.
    copy     weights-only copy-suppression signature per head (McDougall et al. 2023): diagonal of
             W_U' W_O,h W_V,h W_E over the sky words' first tokens (W_U' = W_U with the final RMSNorm gain folded
             in; an approximation), minus the off-diagonal mean.
    folds    sentences split in two halves (even / odd); per head and fold, mean score and t over pairs.
  causal SEL_JSON   (phase B, after attr) boost / hide sets of heads selected on one fold, measured on the other.
  STATES=... python runs/viz/sky_heads.py SKY_JSON OUT attr|causal [SEL_JSON]"""
import json, math, os, sys
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.models.qwen3.modeling_qwen3 as mq
torch.set_grad_enabled(False)

SRC = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sky_flight.py")).read()
SENT = eval(SRC[SRC.index("SENT = [") + 7:SRC.index("]\nsk = ") + 1])
sk = json.load(open(sys.argv[1])); W, G = sk["words"], sk["groups"]; OUT, MODE = sys.argv[2], sys.argv[3]
S = np.load(os.environ.get("STATES", "/Volumes/kingston2g/data/sky/qwen3_4b_sky_states.npy")).astype(np.float32)
NL = S.shape[1]; widx = {w: i for i, w in enumerate(W)}; dev = "mps"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
c = m.config; NH, KV, HD, D = c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.hidden_size
WO = [m.model.layers[l].self_attn.o_proj.weight.float().view(D, NH, HD) for l in range(NL)]
St = torch.tensor(S, device=dev)

BIAS = {}   # layer -> (NH or 1, T, T) additive bias on the attention scores
_orig = mq.eager_attention_forward
def steered(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
    b = BIAS.get(module.layer_idx)
    if b is not None: attention_mask = attention_mask[:, :, :, :key.shape[-2]] + b[None].to(attention_mask.dtype)
    return _orig(module, query, key, value, attention_mask, scaling, dropout, **kw)
mq.eager_attention_forward = steered
VS = {}
for l in range(NL): m.model.layers[l].self_attn.v_proj.register_forward_hook(lambda mod, i, o, l=l: VS.__setitem__(l, o[0].float()))

def bias(T, rows, cols, x, floors, heads=None):   # heads: {layer: [h, ...]} or None for every head
    lb = math.log(x) if x > 0 else -1e4; out = {}
    for l in floors:
        hs = None if heads is None else heads.get(l)
        if heads is not None and not hs: continue
        B = torch.zeros(NH if hs else 1, T, T, device=dev)
        for r in rows:
            for cc in cols:
                if cc <= r:
                    if hs: B[hs, r, cc] = lb
                    else: B[0, r, cc] = lb
        out[l] = B
    return out

def run(ids, b=None):
    BIAS.clear(); BIAS.update(b or {})
    with torch.no_grad(): o = m(torch.tensor([ids], device=dev), output_hidden_states=True, output_attentions=True)
    BIAS.clear()
    return torch.stack([h[0].float() for h in o.hidden_states[1:]]), [a[0].float() for a in o.attentions]

def parse(sent):
    enc = tok(sent, return_offsets_mapping=True, add_special_tokens=False); ids, off = enc["input_ids"], enc["offset_mapping"]
    words = []
    for w in sorted(widx, key=len, reverse=True):
        k = sent.find(" " + w)
        while k >= 0:
            a, b = k + 1, k + 1 + len(w)
            if (b == len(sent) or not sent[b].isalpha()) and not any(a < y and x < b for _, _, x, y in words):
                words.append((w, [t for t, (x, y) in enumerate(off) if x < b and y > a], a, b))
            k = sent.find(" " + w, k + 1)
    words.sort(key=lambda r: r[2]); return ids, words

def dirs(wi, v, inside):   # floors x D unit vectors: toward v, toward each group-mate
    iv = widx[v]; mates = [u for u, q in enumerate(G) if q == G[iv] and u != iv and W[u] not in inside]
    un = lambda x: x / x.norm(dim=-1, keepdim=True)
    return un(St[iv] - St[wi]), un(St[mates] - St[wi][None])          # (L, D), (M, L, D)

def pull_parts(H, wi, last, uv, um):   # per floor: cos(d, u_v), mean cos(d, u_mates)
    d = H[:, last] - St[wi]; dn = d / d.norm(dim=-1, keepdim=True)
    return (dn * uv).sum(-1).cpu().numpy(), (dn[None] * um).sum(-1).mean(0).cpu().numpy()

PAIRS = []
for si, (sent, _) in enumerate(SENT):
    ids, words = parse(sent); inside = [x[0] for x in words]
    for j, (w, wsp, _, _) in enumerate(words):
        for v, vsp, _, _ in words[:j]: PAIRS.append((si, w, wsp, v, vsp, len(words[:j])))

if MODE == "attr":
    NP = len(PAIRS); score = np.zeros((NL, NH, NP), np.float32); toward = np.zeros_like(score); mates = np.zeros_like(score); mag = np.zeros_like(score)
    tot = {"toward": np.zeros((NL, NP)), "mates": np.zeros((NL, NP))}; boost0 = []
    pi = 0
    for si, (sent, _) in enumerate(SENT):
        ids, words = parse(sent); inside = [x[0] for x in words]; T = len(ids)
        H, A = run(ids); V = {l: VS[l].view(T, KV, HD).repeat_interleave(NH // KV, 1) for l in range(NL)}   # T x NH x HD
        for j, (w, wsp, _, _) in enumerate(words):
            wi, last = widx[w], wsp[-1]
            for v, vsp, _, _ in words[:j]:
                uv, um = dirs(wi, v, inside)
                for l in range(NL):
                    a = A[l][:, last, vsp]                                              # NH x |span|
                    ch = torch.einsum("hs,shd,ehd->he", a, V[l][vsp], WO[l])            # NH x D: payload per head
                    pv, pm = ch @ uv[l], (ch @ um[:, l].T).mean(1)
                    toward[l, :, pi], mates[l, :, pi] = pv.cpu().numpy(), pm.cpu().numpy()
                    score[l, :, pi] = (pv - pm).cpu().numpy(); mag[l, :, pi] = ch.norm(dim=1).cpu().numpy()
                    cs = ch.sum(0); tot["toward"][l, pi], tot["mates"][l, pi] = float(cs @ uv[l]), float((um[:, l] @ cs).mean())
                # phase 0 (b): all-head boost, parts kept apart
                b0v, b0m = pull_parts(H, wi, last, uv, um)
                Hb, _ = run(ids, bias(T, wsp, vsp, 4, range(12, 25)))
                b1v, b1m = pull_parts(Hb, wi, last, uv, um)
                boost0.append({"dv": (b1v - b0v).tolist(), "dm": (b1m - b0m).tolist(), "n_prev": PAIRS[pi][5]})
                pi += 1
        print(si, sent, flush=True)
    # copy-suppression signature, weights only
    first = sorted({tok(" " + w, add_special_tokens=False)["input_ids"][0] for w in W})
    E = m.model.embed_tokens.weight[first].float(); U = (m.lm_head.weight[first].float() * m.model.norm.weight.float()[None])
    cps = np.zeros((NL, NH), np.float32)
    with torch.no_grad():
        for l in range(NL):
            at = m.model.layers[l]; x = at.input_layernorm(E.to(torch.bfloat16)).float()
            vv = (x @ at.self_attn.v_proj.weight.float().T).view(-1, KV, HD).repeat_interleave(NH // KV, 1)   # n x NH x HD
            out = torch.einsum("nhd,ehd->hne", vv, WO[l])                                                  # NH x n x D
            Mx = torch.einsum("hne,ke->hnk", out, U)                                                       # NH x n x n
            dg = Mx.diagonal(dim1=1, dim2=2).mean(1); off = (Mx.sum((1, 2)) - Mx.diagonal(dim1=1, dim2=2).sum(1)) / (len(first) * (len(first) - 1))
            sd = Mx.flatten(1).std(1); cps[l] = ((dg - off) / sd).cpu().numpy()
    fold = np.array([PAIRS[p][0] % 2 for p in range(NP)])
    np.savez_compressed(OUT + ".npz", score=score, toward=toward, mates=mates, mag=mag, fold=fold, copy=cps,
                        tot_toward=tot["toward"], tot_mates=tot["mates"])
    json.dump({"boost0": boost0}, open(OUT + "_boost0.json", "w"))
    print("phase 0 (a) head-summed payload, mean over pairs, floors 0..35 step 3:")
    print(" toward v ", np.round(tot["toward"].mean(1)[::3], 3).tolist())
    print(" toward mates", np.round(tot["mates"].mean(1)[::3], 3).tolist())
    dv = np.array([b["dv"] for b in boost0]); dm = np.array([b["dm"] for b in boost0])
    print("phase 0 (b) all-head boost: change in cos to v / to mates, floors 12,18,24,30,35:")
    for l in (12, 18, 24, 30, 35): print(" ", l, round(float(dv[:, l].mean()), 4), round(float(dm[:, l].mean()), 4))

if MODE == "causal":   # heads chosen on one fold (top-K by t), tested on the other; both directions of the split
    d = np.load(OUT + ".npz"); Sc, fold = d["score"], d["fold"]; K = 24; rng = np.random.default_rng(0)
    def tstat(mask): X = Sc[:, :, mask]; return X.mean(2) / (X.std(2, ddof=1) / np.sqrt(X.shape[2]) + 1e-12)
    res = {}
    for train in (0, 1):
        t = tstat(fold == train).ravel(); order = np.argsort(t)
        sets = {"attract": order[::-1][:K], "repel": order[:K]}
        for nm in ("attract", "repel"):   # floor-matched random heads
            fl = sets[nm] // NH; sets[nm + "_rand"] = np.array([l * NH + rng.integers(NH) for l in fl])
        sets = {k: {int(l): [int(i % NH) for i in v if i // NH == l] for l in set((v // NH).tolist())} for k, v in sets.items()}
        for pi, (si, w, wsp, v, vsp, _) in enumerate(PAIRS):
            if fold[pi] == train: continue
            ids, words = parse(SENT[si][0]); inside = [x[0] for x in words]; T = len(ids); wi = widx[w]
            uv, um = dirs(wi, v, inside); H0, _ = run(ids); a0, m0 = pull_parts(H0, wi, last := wsp[-1], uv, um)
            for nm, hs in sets.items():
                for act, x in (("boost", 4), ("hide", 0)):
                    H1, _ = run(ids, bias(T, wsp, vsp, x, range(NL), hs)); a1, m1 = pull_parts(H1, wi, last, uv, um)
                    res.setdefault(f"{act}_{nm}", []).append(((a1 - m1) - (a0 - m0)).tolist())
        print("fold", train, "done", flush=True)
    out = {}
    for k, v in res.items():
        X = np.array(v); r = {}
        for l in (6, 12, 18, 24, 30, 35):
            x = X[:, l]; null = (rng.choice([-1, 1], (5000, len(x))) * x).mean(1)
            r[l] = (round(float(x.mean()), 4), round(float(((np.abs(null) >= abs(x.mean())).sum() + 1) / 5001), 4))
        out[k] = {"n": len(X), "floors": r}; print(k, out[k], flush=True)
    json.dump(out, open(OUT + "_causal.json", "w"))
