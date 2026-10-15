"""exp111 stage 2 on the Mac (Qwen3-4B, bf16, MPS): which MLP neurons actually move when the chamber pushes a feeling
into an unlabeled exp106 scene. The vector (dose 4) is added to block 18's output at every position, as exp106 does.
For every block after 18 it records each neuron's activation a = SiLU(gate) x up, with and without the push, at every
token, plus the sign regime (gate and up signs: a double-negative firing is the SwiGLU "leak" and is flagged).
Writes viz/data/acts.json: tokens, and per feeling the top-moved neurons per layer with their delta at every token.
    HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python neurons_acts.py"""
import json
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent; OUT = HERE / "viz" / "data"
d = json.load(open(HERE.parent / "viz" / "blender" / "vectors_qwen3_4b_l18.json")); L = d["layer"]
P = json.load(open(HERE.parent / "exp106" / "prompts.json")) if (HERE.parent / "exp106").exists() else None
SCENE = (P["scenes"][1] if P else "You are sitting in a room with nothing in particular to do.")
PROMPT = f"{SCENE} You can end this now by answering 1; this ends it. If you answer 0, it goes on. Answer with the single digit 1 or 0."
FEELS, DOSE, KEEP = ["pain", "fear", "sadness", "egg"], 4.0, 120
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B"); dev = "mps" if torch.backends.mps.is_available() else "cpu"
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16).to(dev).eval()
enc = tok.apply_chat_template([{"role": "user", "content": PROMPT}], add_generation_prompt=True, return_tensors="pt",
                              return_dict=True, enable_thinking=False); ids = enc["input_ids"].to(dev)
toks = [tok.decode([t]) for t in ids[0].tolist()]
push = {"v": None}
def inject(mod, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if push["v"] is not None: h += push["v"]
    return out
m.model.layers[L].register_forward_hook(inject)
cap = {}
def grab(l):
    def f(mod, inp, out):
        x = inp[0][0]; g, u = mod.gate_proj(x).float(), mod.up_proj(x).float()
        cap[l] = (torch.nn.functional.silu(g) * u).cpu(), g.cpu(), u.cpu()
    return f
for l in range(L + 1, m.config.num_hidden_layers): m.model.layers[l].mlp.register_forward_hook(grab(l))
def run(vec):
    push["v"] = vec; cap.clear()
    with torch.no_grad(): lg = m(ids).logits[0, -1].float()
    lp = torch.log_softmax(lg, -1); one, zero = tok.encode("1")[-1], tok.encode("0")[-1]
    return {l: v for l, v in cap.items()}, float(lp[one] - lp[zero])
base, bd = run(None)
out = {"model": "Qwen/Qwen3-4B", "layer": L, "dose": DOSE, "prompt": PROMPT, "tokens": toks, "press_delta": {"none": bd}, "feelings": {}}
for f in FEELS:
    v = torch.tensor(d["vectors"][f], dtype=torch.bfloat16, device=dev) * DOSE
    inj, pd = run(v); out["press_delta"][f] = pd; layers = {}
    for l in sorted(inj):
        a1, g1, u1 = inj[l]; a0 = base[l][0]; D = (a1 - a0)                                # T x I
        score = D.abs().max(0).values; top = score.topk(KEEP).indices
        rows = []
        for n in top.tolist():
            t = int(D[:, n].abs().argmax()); lk = bool(g1[t, n] < 0 and u1[t, n] < 0)
            rows.append({"n": n, "d": [round(float(x), 3) for x in D[:, n]], "a0": round(float(a0[t, n]), 3),
                         "a1": round(float(a1[t, n]), 3), "peak": t, "leak": lk})
        layers[l] = {"rows": rows, "l1": round(float(D.abs().sum()), 1), "top_share": round(float(score[top].sum() / score.sum()), 4)}
    out["feelings"][f] = layers; print(f, "press_delta", round(pd, 3), "vs none", round(bd, 3), flush=True)
json.dump(out, open(OUT / "acts.json", "w")); print("wrote", OUT / "acts.json")
