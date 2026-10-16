"""exp111: attention for the tower's built-in runs (stream.json), so the reading ring plays without a live model.
Re-runs each prompt + condition's full token sequence once (same push as the stream; causal, so identical to reading it
word by word) through attn_flow.Tap and keeps per token: the top 2 sources per floor (share >= 5% of the floor's flow)
and the 16 biggest single head moves. Writes attn.json {prompt: {cond: [[al, ah], ...]}}.
    HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python neurons_attn_stream.py"""
import json
import torch
from tower_cfg import DATA as D, VECS, DOSE, DEV as dev, load
from attn_flow import Tap
V = json.load(open(VECS)); L = V["layer"]; ST = json.load(open(D / "stream.json")); tok, m = load()
tap = Tap(m, D / "heads.json"); push = {"v": None}
def inject(mod, inp, out):
    if push["v"] is None: return out
    return (out[0] + push["v"],) + tuple(out[1:]) if isinstance(out, tuple) else out + push["v"]
m.model.layers[L].register_forward_hook(inject)
out = {}
for pk, p in ST["prompts"].items():
    out[pk] = {}
    for c, cd in p["conds"].items():
        push["v"] = None if c == "none" else torch.tensor(V["vectors"][c], dtype=torch.bfloat16, device=dev) * DOSE
        ids = tok.apply_chat_template([{"role": "user", "content": p["text"]}], add_generation_prompt=True, enable_thinking=False)
        ids = ids if isinstance(ids, list) else ids["input_ids"]
        ids = ids + tok("".join(cd["tokens"][len(ids):]), add_special_tokens=False)["input_ids"]
        ids = ids[:len(cd["tokens"])]
        tap.reset()
        with torch.no_grad(): m(torch.tensor([ids], device=dev), use_cache=True)
        S = tap.summary(0, len(ids), top_src=2, top_heads=16)
        out[pk][c] = [[[a for a in s["al"] if a[2] >= 5], s["ah"]] for s in S]
        print(pk, c, len(ids), "of", len(cd["tokens"]), flush=True)
json.dump(out, open(D / "attn.json", "w"), separators=(",", ":"))
print("wrote attn.json")
