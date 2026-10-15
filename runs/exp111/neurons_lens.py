"""exp111: add the logit lens (tokens each neuron's down_proj column promotes / suppresses, final norm folded in) to the
top 30 moved neurons per layer per feeling in viz/data/acts.json. Weights only, read straight from the safetensors.
    HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python neurons_lens.py"""
import glob, json, os
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
HERE = Path(__file__).parent; F = HERE / "viz" / "data" / "acts.json"; a = json.load(open(F))
SNAP = glob.glob(os.path.join(os.environ["HF_HUB_CACHE"], "models--Qwen--Qwen3-4B", "snapshots", "*"))[0]
idx = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"]; H = {}
def T(n):
    f = idx[n]; H.setdefault(f, safe_open(os.path.join(SNAP, f), "pt")); return H[f].get_tensor(n).float()
tok = AutoTokenizer.from_pretrained(SNAP); En = T("model.embed_tokens.weight") * T("model.norm.weight")
cache = {}
for f, layers in a["feelings"].items():
    for l, info in layers.items():
        down = None
        for r in info["rows"][:30]:
            k = (l, r["n"])
            if k not in cache:
                if down is None: down = T(f"model.layers.{l}.mlp.down_proj.weight")
                s = En @ down[:, r["n"]]
                cache[k] = ([tok.decode([t]) for t in s.topk(6).indices.tolist()], [tok.decode([t]) for t in (-s).topk(6).indices.tolist()])
            r["promote"], r["suppress"] = cache[k]
    print(f, flush=True)
json.dump(a, open(F, "w")); print(len(cache), "neurons labelled")
