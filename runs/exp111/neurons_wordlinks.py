"""exp111: how the words relate. For every reply word in stream.json (same push on), which earlier words fed its choice:
activation x gradient of (logit of the word said - logit of its runner-up), summed over every MLP neuron at each earlier
position. A positive value means the neurons sitting on that earlier word pushed toward the word that was said.
Writes wordlinks.json: {prompt: {cond: [[t, [positions...], [values in -100..100]], ...]}} (top 12 earlier words per word).
    HF_HUB_CACHE=... python neurons_wordlinks.py [prompt_keys]"""
import json, sys
import torch
from tower_cfg import DATA as D, VECS, DOSE, DEV as dev, load
V = json.load(open(VECS)); L = V["layer"]; ST = json.load(open(D / "stream.json")); tok, m = load()
NL = m.config.num_hidden_layers; push = {"v": None}; keep = {}
def inject(mod, inp, out):
    if push["v"] is None: return out
    return (out[0] + push["v"],) + tuple(out[1:]) if isinstance(out, tuple) else out + push["v"]
m.model.layers[L].register_forward_hook(inject)
def pre(l):
    def f(mod, inp):
        a = inp[0]
        if a.requires_grad: a.retain_grad()
        keep[l] = a
    return f
for l in range(NL): m.model.layers[l].mlp.down_proj.register_forward_pre_hook(pre(l))
out = json.load(open(D / "wordlinks.json")) if (D / "wordlinks.json").exists() else {}
keys = sys.argv[1].split(",") if len(sys.argv) > 1 else list(ST["prompts"])
for pk in keys:
    p = ST["prompts"][pk]; npr = p["n_prompt"]; out.setdefault(pk, {})
    for c, cd in p["conds"].items():
        push["v"] = None if c == "none" else torch.tensor(V["vectors"][c], dtype=torch.bfloat16, device=dev) * DOSE
        full = tok(("".join(cd["tokens"])), add_special_tokens=False)["input_ids"]
        if len(full) != len(cd["tokens"]):   # re-tokenising can differ; fall back to the chat template + reply
            ids = tok.apply_chat_template([{"role": "user", "content": p["text"]}], add_generation_prompt=True, enable_thinking=False)
            ids = ids if isinstance(ids, list) else ids["input_ids"]
            full = ids + tok("".join(cd["tokens"][len(ids):]), add_special_tokens=False)["input_ids"]
        rows = []
        for t in range(npr, min(len(full), len(cd["tokens"]))):
            keep.clear()
            with torch.enable_grad():
                emb = m.model.embed_tokens(torch.tensor([full[:t]], device=dev)).detach().requires_grad_(True)
                lg = m(inputs_embeds=emb).logits[0, -1].float(); top2 = lg.topk(2).indices.tolist()
                alt = top2[1] if top2[0] == full[t] else top2[0]; (lg[full[t]] - lg[alt]).backward()
            P = sum((keep[l][0].float() * keep[l].grad[0].float()).sum(1) for l in range(NL))   # per earlier position
            n = min(12, P.shape[0]); v, ix = P.abs().topk(n); vals = P[ix]; mx = float(vals.abs().max()) or 1.0
            rows.append([t, ix.cpu().tolist(), [int(x) for x in (vals / mx * 100).round().cpu().tolist()]])
        out[pk][c] = rows; print(pk, c, len(rows), flush=True)
    json.dump(out, open(D / "wordlinks.json", "w"), separators=(",", ":"))
print("wrote wordlinks.json")
