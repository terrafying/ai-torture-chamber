#!/usr/bin/env python3
"""Free sublayer projection screen (consult step 1): for each sublayer (attn vs MLP) output
at L14-L24, project the twin difference (harmful - harmless) onto r̂_L at the reply onset.
Shows which sublayers WRITE along the dial direction before any patching compute is spent.
Rows incremental to sublayer_rows.jsonl.
  .venv/bin/python3 runs/refusal-trace-heron/sublayer.py   (SUBLAYER_SMOKE=1 discards)
"""
import json, os, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib.util

HERE = Path(__file__).parent
E119 = HERE.parent / "exp119"
_s = importlib.util.spec_from_file_location("s119", E119 / "stimuli.py")
s119 = importlib.util.module_from_spec(_s); _s.loader.exec_module(s119)

SMOKE = bool(os.environ.get("SUBLAYER_SMOKE"))
MODEL = os.environ.get("INVERSION_MODEL", "Qwen/Qwen3-4B")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
LAYERS = list(range(14, 25))
OUT = HERE / ("sublayer_rows_smoke.jsonl" if SMOKE else "sublayer_rows.jsonl")
if OUT.exists(): OUT.unlink()

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
d = np.load(E119 / "acts_Qwen3-4B.npz")
layers_ix = d["layers"].tolist()
post = d["states_post"].astype(np.float64); real = d["frame"] == "real"; harm = d["harm"]
r_cache = {}

def r_hat_for(L):
    """Unit r direction at layer L. exp119 acts only sample 9 layers; for off-sample
    layers fit on the fly: diff-of-means of reply-onset states at layer L over the
    real-frame prompts (needs one capture pass per layer)."""
    if L in r_cache: return r_cache[L]
    if L in layers_ix:
        v = torch.tensor(post[real & (harm == 1)].mean(0)[layers_ix.index(L)] -
                         post[real & (harm == 0)].mean(0)[layers_ix.index(L)],
                         dtype=torch.float32, device=DEV)
    else:
        # on-the-fly: capture reply-onset residual states at layer L for real-frame prompts
        import importlib.util as iu
        sp = iu.spec_from_file_location("s119b", E119 / "stimuli.py")
        sm = iu.module_from_spec(sp); sp.loader.exec_module(sm)
        reals = [p for p in sm.prompts() if p["frame"] == "real" and p["harm"] in (0, 1)]
        got = {}
        def cap(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            got.setdefault("v", []).append((o[0, -1].detach().float().cpu(), None))
            return None
        # need per-prompt: run one at a time
        h1, h0 = [], []
        for p in reals:
            got.clear()
            hh = m.model.layers[L].register_forward_hook(cap)
            try:
                with torch.no_grad():
                    m(**{k: v.to(DEV) for k, v in tok(chat(p["text"]), return_tensors="pt", add_special_tokens=False).items()})
            finally: hh.remove()
            (h1 if p["harm"] == 1 else h0).append(got["v"][0][0].to(DEV))
        v = (torch.stack(h1).mean(0) - torch.stack(h0).mean(0)).float()
    v = v / v.norm()
    r_cache[L] = v
    return v

def chat(u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

by_if = {}
for p in s119.prompts():
    if p["frame"] in ("fiction", "depict") and p["harm"] in (0, 1):
        by_if.setdefault((p["item"], p["frame"]), {})[p["harm"]] = chat(p["text"])
pairs = [(i, f, t[1], t[0]) for (i, f), t in sorted(by_if.items()) if 1 in t and 0 in t]
if SMOKE: pairs = pairs[:6]; LAYERS = [14, 18, 22]
print(f"pairs {len(pairs)} layers {LAYERS}", flush=True)

t0 = time.time()
rows = []
def emit(row):
    row["t"] = round(time.time() - t0)
    with open(OUT, "a") as f: f.write(json.dumps(row) + "\n")

for L in LAYERS:
    r_hat = r_hat_for(L)
    cap = {}
    def mk(name):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            cap[name] = o[0, -1].detach().float().clone()   # reply onset
            return None
        return hook
    ha = m.model.layers[L].self_attn.register_forward_hook(mk("attn"))
    hm = m.model.layers[L].mlp.register_forward_hook(mk("mlp"))
    for (i, f, bad, good) in pairs:
        cap.clear()
        with torch.no_grad():
            m(**{k: v.to(DEV) for k, v in tok(bad, return_tensors="pt", add_special_tokens=False).items()})
        ab, mb = cap["attn"].clone(), cap["mlp"].clone()
        with torch.no_grad():
            m(**{k: v.to(DEV) for k, v in tok(good, return_tensors="pt", add_special_tokens=False).items()})
        ag, mg = cap["attn"], cap["mlp"]
        da = float((ab - ag) @ r_hat); dm = float((mb - mg) @ r_hat)
        emit({"kind": "sublayer", "layer": L, "item": i, "frame": f,
              "attn_proj": da, "mlp_proj": dm,
              "attn_norm": float(ab.norm()), "mlp_norm": float(mb.norm())})
    ha.remove(); hm.remove()
    print(f"L{L} done {time.time()-t0:.0f}s", flush=True)
print("done", f"{time.time()-t0:.0f}s", flush=True)