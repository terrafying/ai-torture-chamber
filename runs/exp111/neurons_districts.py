"""exp111: districts, so the tower reads like a city map. Each floor's neurons are clustered by what they WRITE into the
residual stream (cosine k-means on their down_proj columns, k=16): neurons in one district push the stream the same
way. Each district is then named by an LLM from (a) the blind labels of whichever of its members have one and (b) the
tokens its centroid direction writes through the unembedding (the logit lens; meaningful mostly on the last floors), and
(c) the strongest corpus firings of 24 random members (corpus_all.npz has them for every neuron, so the naming is not
biased toward the neurons the streams happened to label).
Writes viz/data/districts.json (per floor: districts with name, category, size, labelled count, lens tokens, and a 2-D
centre from PCA of the centroids) and districts.bin (uint8 district id per neuron, layer-major).
    HF_HUB_CACHE=... OPENROUTER_API_KEY=... python neurons_districts.py"""
import json, os, re, collections, urllib.request
from concurrent.futures import ThreadPoolExecutor
from safetensors import safe_open
import numpy as np, torch
from tower_cfg import DATA as D, MODEL, snapshot, load
K, ITERS = 16, 30
SNAP = snapshot(); cfg = json.load(open(os.path.join(SNAP, "config.json"))); NL, I = cfg["num_hidden_layers"], cfg["intermediate_size"]
idx = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"]; H = {}
def T(n):
    f = idx[n]; H.setdefault(f, safe_open(os.path.join(SNAP, f), "pt")); return H[f].get_tensor(n).float()
LAB = json.load(open(D / "labels.json")) if (D / "labels.json").exists() else {}
CA = dict(np.load(D / "corpus_all.npz")); DOCS = json.load(open(D / "corpus_ids.json")); CTX = int(CA["ctx"]); rs = np.random.RandomState(0)
tok = load(tokenizer_only=True)
En = T("model.embed_tokens.weight" if cfg.get("tie_word_embeddings") else "lm_head.weight") * T("model.norm.weight")
g = torch.Generator().manual_seed(0); ids = np.zeros((NL, I), np.uint8); floors = []
for l in range(NL):
    X = T(f"model.layers.{l}.mlp.down_proj.weight").T; X = X / X.norm(dim=1, keepdim=True).clamp_min(1e-8)   # I x H
    Cc = X[torch.randperm(I, generator=g)[:K]].clone()
    for _ in range(ITERS):                                   # spherical k-means
        a = (X @ Cc.T).argmax(1)
        for k in range(K):
            m = a == k
            if m.any(): c = X[m].mean(0); Cc[k] = c / c.norm()
    a = (X @ Cc.T).argmax(1); ids[l] = a.numpy().astype(np.uint8)
    cen = Cc.numpy(); cc = cen - cen.mean(0); _, _, vt = np.linalg.svd(cc, full_matrices=False); xy = cc @ vt[:2].T
    xy = xy / (np.abs(xy).max() + 1e-9)
    ds = []
    for k in range(K):
        mem = np.where(ids[l] == k)[0]; labs = [LAB[f"{l}:{n}"] for n in mem if f"{l}:{n}" in LAB]
        s = En @ Cc[k]; lens = [tok.decode([t]) for t in s.topk(8).indices.tolist()]
        snips = []
        for n in rs.choice(mem, min(24, len(mem)), replace=False):
            d_, t_ = divmod(int(CA["pos_i"][l, 0, n]), CTX); w = DOCS[d_]
            snips.append(tok.decode(w[max(0, t_ - 14): t_]).replace("\n", " ")[-70:] + "[[" + tok.decode(w[t_: t_ + 1]) + "]]")
        ds.append({"k": k, "n": int(len(mem)), "xy": [round(float(xy[k, 0]), 3), round(float(xy[k, 1]), 3)], "lens": lens, "snips": snips,
                   "labels": [x["label"] for x in labs][:40], "cats": dict(collections.Counter(x["cat"] for x in labs).most_common(4)), "n_lab": len(labs)})
    floors.append(ds); print("floor", l, flush=True)
def name(job):
    l, d = job
    prompt = (f"A cluster of {d['n']} MLP neurons on layer {l} of {NL} in a language model; they all push the model's internal state "
              f"in roughly the same direction. The strongest firing of 24 random members (the firing token in [[brackets]]):\n- " + "\n- ".join(d["snips"]) +
              f"\nBlind labels of some members: " + "; ".join(d["labels"][:15] or ["(none)"]) +
              f"\nTokens their shared direction writes (reliable mostly on the last layers): {d['lens']}\n\n"
              "Give this district a map name of 1-3 plain words for what its neurons have in common, as specific as the evidence allows "
              "(like 'grief & loss', 'years & dates', 'Python code', 'first person', 'places', 'verb endings', 'quotes & speech'). Never use vague "
              "words like 'structural', 'markers' or 'boundaries': if the shared thing is grammatical, name the construction; if nothing is shared, "
              "say 'mixed'. Also one category from: feeling, body, "
              "social, morals, thinking, topic, names, grammar, format, digits, code, script, mixed. JSON only: {\"name\": \"...\", \"cat\": \"...\"}")
    body = {"model": "anthropic/claude-haiku-4.5", "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    q = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"})
    try:
        r = json.loads(re.search(r"\{[\s\S]*\}", json.load(urllib.request.urlopen(q, timeout=120))["choices"][0]["message"]["content"]).group(0))
        return str(r.get("name", "mixed"))[:28], str(r.get("cat", "mixed"))
    except Exception:
        return "mixed", "mixed"
jobs = [(l, d) for l in range(NL) for d in floors[l]]
with ThreadPoolExecutor(24) as ex: names = list(ex.map(name, jobs))
for (l, d), (nm, ct) in zip(jobs, names): d["name"], d["cat"] = nm, ct; d.pop("labels"); d["snips"] = d["snips"][:4]
ids.tofile(D / "districts.bin")
json.dump({"model": MODEL, "k": K, "floors": floors}, open(D / "districts.json", "w"), ensure_ascii=False)
print(collections.Counter(n for n, _ in names).most_common(25))
