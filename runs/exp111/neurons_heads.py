"""exp111: a head atlas. What each of the model's attention heads does, measured on real text, so the tower can show
attention by kind. Raw attention overstates heads that park on the first token (a "sink": lots of weight, tiny values),
so every share here is value-weighted: weight(q,k) * |W_O,h v_k|, the size of what head h actually moves from k to q.
Per head, over cached FineWeb-Edu documents (corpus_ids.json) plus a repeated random-token test:
  sink   share moved from the first token          prev   share from the token just before
  self   share from the token itself                dist   mean distance (in tokens) of what it moves
  recent share from the 8 tokens before            focus  apart from the sink, the largest single source's share
  ind    induction: on a random sequence shown twice, share from the token after the earlier copy of the current one
  raw_sink  the same as sink but with raw attention  out   mean size of what the head writes (relative to the floor's)
Kind (first match): induction (ind >= .25), previous-word (prev >= .4), sink (sink >= .5), self (self >= .4),
recent words (recent >= .5), focused look-back (focus >= .3), broad context (otherwise). Writes heads.json.
    HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python neurons_heads.py [n_docs]"""
import json, sys
import numpy as np, torch
from tower_cfg import DATA as D, DEV, load
NDOC = int(sys.argv[1]) if len(sys.argv) > 1 else 24
tok, m = load(); m.set_attn_implementation("eager")
cfg = m.config; NL, NH, KV, HD = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
WO = [m.model.layers[l].self_attn.o_proj.weight.float().view(cfg.hidden_size, NH, HD) for l in range(NL)]   # H x heads x hd
cap = {}
def vhook(l):
    def f(mod, inp, out): cap[("v", l)] = out[0].float().view(-1, KV, HD)                                  # t x kv x hd
    return f
def ahook(l):
    def f(mod, args, kwargs, out): cap[("a", l)] = out[1][0].float()                                       # heads x q x k
    return f
for l in range(NL):
    m.model.layers[l].self_attn.v_proj.register_forward_hook(vhook(l))
    m.model.layers[l].self_attn.register_forward_hook(ahook(l), with_kwargs=True)
def flows(l):   # value-weighted flow heads x q x k, and raw attention
    A = cap[("a", l)]; v = cap[("v", l)]
    vn = torch.stack([(v[:, h // (NH // KV)] @ WO[l][:, h].T).norm(dim=-1) for h in range(NH)])          # heads x k
    return A * vn[:, None, :], A
S = {k: torch.zeros(NL, NH, device=DEV) for k in ("sink", "prev", "self", "dist", "raw_sink", "out", "recent", "focus")}; nq = 0
docs = json.load(open(D / "corpus_ids.json"))[:NDOC]
with torch.no_grad():
    for di, ids in enumerate(docs):
        ids = ids[:192]; t = len(ids); m(torch.tensor([ids], device=DEV)); q = torch.arange(1, t, device=DEV)
        for l in range(NL):
            F, A = flows(l); F, A = F[:, 1:], A[:, 1:]                                                      # skip query 0
            tot = F.sum(-1).clamp(min=1e-9)
            S["sink"][l] += (F[..., 0] / tot).sum(1); S["raw_sink"][l] += A[..., 0].sum(1)
            S["prev"][l] += (F.gather(2, (q - 1)[None, :, None].expand(NH, -1, 1))[..., 0] / tot).sum(1)
            S["self"][l] += (F.gather(2, q[None, :, None].expand(NH, -1, 1))[..., 0] / tot).sum(1)
            dist = (q[:, None] - torch.arange(t, device=DEV)[None, :]).clamp(min=0).float()
            S["dist"][l] += ((F * dist[None]).sum(-1) / tot).sum(1); S["out"][l] += tot.sum(1)
            S["recent"][l] += ((F * ((dist >= 1) & (dist <= 8))[None]).sum(-1) / tot).sum(1)
            P = F[..., 1:] / F[..., 1:].sum(-1, keepdim=True).clamp(min=1e-9)                              # without the sink
            S["focus"][l] += (P.max(-1).values).sum(1)
        nq += t - 1
        if di % 8 == 0: print("doc", di, flush=True)
    for k in S: S[k] /= nq
    g = torch.Generator().manual_seed(0); R = torch.randint(1000, 30000, (50,), generator=g).tolist(); ids = R + R
    m(torch.tensor([ids], device=DEV)); ind = torch.zeros(NL, NH, device=DEV)
    for l in range(NL):
        F, _ = flows(l); qs = torch.arange(51, 100, device=DEV); tot = F[:, qs].sum(-1).clamp(min=1e-9)
        ind[l] = (F[:, qs, :].gather(2, (qs - 49)[None, :, None].expand(NH, -1, 1))[..., 0] / tot).mean(1)
S["ind"] = ind; S["out"] = S["out"] / S["out"].mean(1, keepdim=True)
def kind(r):
    if r["ind"] >= .25: return "induction"
    if r["prev"] >= .4: return "previous-word"
    if r["sink"] >= .5: return "sink"
    if r["self"] >= .4: return "self"
    if r["recent"] >= .5: return "recent words"
    return "focused look-back" if r["focus"] >= .3 else "broad context"
heads = []
for l in range(NL):
    for h in range(NH):
        r = {k: round(float(S[k][l, h]), 4) for k in S}; r.update(l=l, h=h, kind=kind(r)); heads.append(r)
import collections
summary = {"kinds": dict(collections.Counter(x["kind"] for x in heads)), "docs": NDOC, "queries": nq,
           "mean_raw_sink": round(float(S["raw_sink"].mean()), 3), "mean_valueweighted_sink": round(float(S["sink"].mean()), 3)}
json.dump({"summary": summary, "heads": heads}, open(D / "heads.json", "w"), separators=(",", ":"))
print(json.dumps(summary))
for k in ("induction", "previous-word", "focused look-back", "recent words"):
    print(k, [(x["l"], x["h"]) for x in sorted(heads, key=lambda x: -x["out"]) if x["kind"] == k][:12])
