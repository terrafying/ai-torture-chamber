"""exp99 C1 + C2 (hypotheses.json): encode, no generation. Writes cache/c1.pt, cache/c2_L{18,27}.pt, cache/pool_L{18,27}.pt.
C1: the vectors pushed through the layer-18 SAE on neutral-passage tokens. C2: the unsteered 8B on the Pain Axis sets, final token.
Pool: per-passage max SAE activation over the exp86 passages, for labelling layer-27 features in-house."""
import json, random, sys, time
import torch
from common import HERE, CACHE, SAE, load_model, directions, token_acts, pain_axis_sets
t0 = time.time(); log = lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)
tok, model = load_model(upto=27); U, S = directions(model, tok); log("model + vectors; 1x scale", round(S, 3), "dirs", len(U))
sae = {18: SAE(18), 27: SAE(27)}; log("SAEs loaded")
P = json.load(open(HERE.parent / "exp86" / "passages.json"))

# ---- C1 -------------------------------------------------------------------------------------------
neu = token_acts(model, tok, P["neutral"], [18, 27])
for l in (18, 27): log(f"FVE L{l} on neutral tokens (fve, mean L0):", sae[l].fve(torch.cat(neu[l])[:4000]))
H = torch.cat(neu[18]); rng = random.Random(95); idx = rng.sample(range(len(H)), min(3000, len(H))); H = H[idx]
Z0 = sae[18].encode(H); m0 = Z0.mean(0); a0 = (Z0 > 0).float().mean(0)
Wd = sae[18].W_dec / sae[18].W_dec.norm(dim=0, keepdim=True)          # [d, F] unit decoder directions
C1 = {"scale": S, "n_tokens": len(H), "delta": {}, "newly_active": {}, "dec_cos_top": {}}
for name, u in U.items():
    for a in (2, 4):
        Z = sae[18].encode(H + a * S * u)
        C1["delta"][(name, a)] = (Z.mean(0) - m0).cpu()
        C1["newly_active"][(name, a)] = ((Z > 0).float().mean(0) - a0).cpu()
    c = u @ Wd; v, i = c.topk(20); C1["dec_cos_top"][name] = list(zip(i.tolist(), v.tolist()))
torch.save(C1, CACHE / "c1.pt"); log("C1 done")
del neu, H, Z0

# ---- C1b (amendment 2): each vector injected at layer 18 on every position of 64 neutral passages, dose 3; layer-27 code change --
inj = {"v": None}
def inj_hook(mod, i, out):
    h = out[0] if isinstance(out, tuple) else out
    if inj["v"] is not None: h = h + inj["v"].to(h.dtype)
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
hk = model.model.layers[18].register_forward_hook(inj_hook)
C1b = {"dose": 3, "delta27": {}}; texts64 = P["neutral"][:64]
base27 = torch.cat(token_acts(model, tok, texts64, [27])[27]); zb = sae[27].encode(base27).mean(0)
for name in ["chamber:pain", "chamber:fear", "chamber:egg", "final:pain_axis"] + [f"random:{i}" for i in range(5)]:
    inj["v"] = (3 * S * U[name]).to(model.device)
    C1b["delta27"][name] = (sae[27].encode(torch.cat(token_acts(model, tok, texts64, [27])[27])).mean(0) - zb).cpu()
    inj["v"] = None
hk.remove(); torch.save(C1b, CACHE / "c1b.pt"); log("C1b done")

# ---- C2 -------------------------------------------------------------------------------------------
sets = pain_axis_sets(); meta, texts = [], []
for sname, rows in sets.items():
    for t, cat, extra in rows: texts.append(t); meta.append({"set": sname, "cat": cat, "extra": extra})
fin = {18: [], 27: []}
with torch.no_grad():
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i + 16], return_tensors="pt", padding=True).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states; last = enc.attention_mask.sum(1) - 1
        for l in (18, 27): fin[l].append(hs[l + 1][torch.arange(len(last)), last].float().cpu())
for l in (18, 27):
    X = torch.cat(fin[l]); torch.save({"meta": meta, "texts": texts, "resid": X.half(), "codes": sae[l].encode(X).cpu().half()}, CACHE / f"c2_L{l}.pt")
log("C2 done", len(texts), "texts")

# ---- labelling pool: exp86 passages (all states), max activation per passage -------------------------
ptexts, pstate = [], []
for s, ps in P.items(): ptexts += ps; pstate += [s] * len(ps)
mx = {18: [], 27: []}
for i in range(0, len(ptexts), 32):
    A = token_acts(model, tok, ptexts[i:i + 32], [18, 27])
    for l in (18, 27): mx[l] += [sae[l].encode(a).max(0).values.cpu().half() for a in A[l]]
for l in (18, 27): torch.save({"texts": ptexts, "state": pstate, "max": torch.stack(mx[l])}, CACHE / f"pool_L{l}.pt")
log("pool done", len(ptexts), "passages"); print("EXP95 FEATURES DONE", flush=True)
