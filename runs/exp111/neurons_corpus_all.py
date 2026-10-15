"""exp111: corpus profile of EVERY MLP neuron of Qwen3-4B (vectorized, unlike neurons_corpus.py's featured-only heaps).
Runs ~400k tokens of FineWeb-Edu (the Hub's dataset-viewer API; token ids cached in viz/data/corpus_ids.json) and keeps,
per neuron: max, min, mean |a|, std, and the 5 strongest firings on each pole (doc, token). Activation a = the input of
down_proj (SiLU(gate) x up). The scale (max |a| over both poles) lets other scripts rank a change by the neuron's own
range, not by raw size, so a few massive neurons stop dominating every view.
Writes viz/data/corpus_all.npz (+ corpus_ids.json).
    HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python neurons_corpus_all.py [n_docs]"""
import json, sys, urllib.request
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent; D = HERE / "viz" / "data"; NDOC = int(sys.argv[1]) if len(sys.argv) > 1 else 1600; CTX, K = 256, 5
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
cache = D / "corpus_ids.json"
if cache.exists() and len(json.load(open(cache))) >= NDOC: docs = json.load(open(cache))[:NDOC]
else:
    docs = []
    for off in range(0, NDOC, 100):
        u = f"https://datasets-server.huggingface.co/rows?dataset=HuggingFaceFW/fineweb-edu&config=sample-10BT&split=train&offset={off}&length=100"
        for r in json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})))["rows"]:
            ids = tok(r["row"]["text"], add_special_tokens=False)["input_ids"][:CTX]
            if len(ids) >= 32: docs.append(ids)
    json.dump(docs, open(cache, "w"))
dev = "mps"; m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).to(dev).eval()
NL, I = m.config.num_hidden_layers, m.config.intermediate_size
cur = {}
for l in range(NL): m.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, inp, l=l: cur.__setitem__(l, inp[0][0].float()))
NEG = -1e9
topv = {p: torch.full((NL, K, I), NEG, device=dev) for p in ("pos", "neg")}
topi = {p: torch.zeros((NL, K, I), dtype=torch.int64, device=dev) for p in ("pos", "neg")}   # doc * CTX + token
s1 = torch.zeros(NL, I, device=dev); s2 = torch.zeros(NL, I, device=dev); sa = torch.zeros(NL, I, device=dev); ntok = 0
for di, ids in enumerate(docs):
    with torch.no_grad():
        m(torch.tensor([ids], device=dev))
        T = len(ids); pos_ix = torch.arange(T, device=dev)[:, None].expand(T, I) + di * CTX
        for l in range(NL):
            A = cur[l]; s1[l] += A.sum(0); s2[l] += (A * A).sum(0); sa[l] += A.abs().sum(0)
            for p, X in (("pos", A), ("neg", -A)):
                v = torch.cat([topv[p][l], X]); ix = torch.cat([topi[p][l], pos_ix])
                tv, tj = v.topk(K, dim=0); topv[p][l] = tv; topi[p][l] = ix.gather(0, tj)
    ntok += len(ids)
    if di % 100 == 0: print(di, ntok, flush=True)
mean = s1 / ntok; std = (s2 / ntok - mean * mean).clamp_min(0).sqrt()
np.savez_compressed(D / "corpus_all.npz", n_tokens=ntok, ctx=CTX,
    max=topv["pos"][:, 0].cpu().numpy(), min=(-topv["neg"][:, 0]).cpu().numpy(), mean_abs=(sa / ntok).cpu().numpy(), std=std.cpu().numpy(),
    pos_v=topv["pos"].cpu().numpy(), pos_i=topi["pos"].cpu().numpy(), neg_v=(-topv["neg"]).cpu().numpy(), neg_i=topi["neg"].cpu().numpy())
print("wrote corpus_all.npz", ntok, "tokens")
