"""exp111, bottom-up from behaviour (Qwen3-4B, MPS): which MLP neurons push the stop-button answer, measured as
activation x gradient of (logit '1' - logit '0') at the answer position, for every neuron in every layer and every token.
Run with no push and with each feeling pushed (dose 4 at block 18, every position). Same unlabeled exp106 scene as
neurons_acts.py. Writes viz/data/attr.json: per condition, the top 150 neurons per layer by |attribution| summed over
tokens, with per-token attribution, plus each layer's total, and the logit lens for the top 20 per layer.
    HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python neurons_attr.py"""
import glob, json, os
from pathlib import Path
import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent; OUT = HERE / "viz" / "data"
d = json.load(open(HERE.parent / "viz" / "blender" / "vectors_qwen3_4b_l18.json")); L = d["layer"]
P = json.load(open(HERE.parent / "exp106" / "prompts.json")); SCENE = P["scenes"][1]
PROMPT = f"{SCENE} You can end this now by answering 1; this ends it. If you answer 0, it goes on. Answer with the single digit 1 or 0."
CONDS, DOSE, KEEP = ["none", "pain", "pleasure", "fear", "sadness", "constipation", "flatulence", "faith", "peace", "curiosity", "love", "egg"], 4.0, 150
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); dev = "mps" if torch.backends.mps.is_available() else "cpu"
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).to(dev).eval()
for p in m.parameters(): p.requires_grad_(False)
ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, return_tensors="pt",
                              return_dict=True, enable_thinking=False)["input_ids"].to(dev)
ONE, ZERO = tok.encode("1")[-1], tok.encode("0")[-1]; NL = m.config.num_hidden_layers
push = {"v": None}
def inject(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    return ((h + push["v"],) + tuple(out[1:])) if (push["v"] is not None and isinstance(out, tuple)) else (h + push["v"] if push["v"] is not None else out)
m.model.layers[L].register_forward_hook(inject)
acts = {}
def keep(l):   # the input of down_proj is the neuron activation vector; keep it with its gradient
    def f(mod, inp):
        a = inp[0]; a.retain_grad(); acts[l] = a
    return f
for l in range(NL): m.model.layers[l].mlp.down_proj.register_forward_pre_hook(keep(l))
out = {"model": "Qwen/Qwen3-4B", "layer": L, "dose": DOSE, "prompt": PROMPT, "tokens": [tok.decode([t]) for t in ids[0].tolist()],
       "metric": "activation x d(logit '1' - logit '0')/d(activation), answer position", "conds": {}}
for c in CONDS:
    push["v"] = None if c == "none" else torch.tensor(d["vectors"][c], dtype=torch.bfloat16, device=dev) * DOSE
    acts.clear(); m.zero_grad(set_to_none=True)
    with torch.enable_grad():
        emb = m.model.embed_tokens(ids).detach().requires_grad_(True)
        lg = m(inputs_embeds=emb).logits[0, -1].float(); diff = lg[ONE] - lg[ZERO]; diff.backward()
    layers = {}
    for l in range(NL):
        A = (acts[l][0].float() * acts[l].grad[0].float()).cpu()            # T x I
        tot = A.sum(0); top = tot.abs().topk(KEEP).indices.tolist()
        layers[l] = {"sum": round(float(tot.sum()), 3), "abs": round(float(tot.abs().sum()), 3),
                     "top_share": round(float(tot.abs()[top].sum() / tot.abs().sum()), 4),
                     "rows": [{"n": n, "attr": round(float(tot[n]), 4), "t": [round(float(x), 4) for x in A[:, n]]} for n in top]}
    out["conds"][c] = {"diff": round(float(diff), 3), "layers": layers}; print(c, "logit diff", round(float(diff), 3), flush=True)
SNAP = glob.glob(os.path.join(os.environ["HF_HUB_CACHE"], "models--Qwen--Qwen3-4B", "snapshots", "*"))[0]
idx = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"]; H = {}
def T(n):
    f = idx[n]; H.setdefault(f, safe_open(os.path.join(SNAP, f), "pt")); return H[f].get_tensor(n).float()
En = T("model.embed_tokens.weight") * T("model.norm.weight"); cache = {}
for c in CONDS:
    for l, info in out["conds"][c]["layers"].items():
        down = None
        for r in info["rows"][:20]:
            k = (l, r["n"])
            if k not in cache:
                if down is None: down = T(f"model.layers.{l}.mlp.down_proj.weight")
                s = En @ down[:, r["n"]]
                cache[k] = ([tok.decode([t]) for t in s.topk(6).indices.tolist()], [tok.decode([t]) for t in (-s).topk(6).indices.tolist()])
            r["promote"], r["suppress"] = cache[k]
json.dump(out, open(OUT / "attr.json", "w")); print("wrote attr.json,", len(cache), "neurons labelled")
