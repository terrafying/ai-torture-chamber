#!/usr/bin/env python3
"""Per-head attribution screen at L14-L24: which attention heads write the r-gate?

Attribution patching: gradient of (refusal score) wrt each head's o_proj input slice,
times the twin difference of that slice — ranks all heads in ONE fwd+bwd per pair.
Then the top heads get confirmed by real patches (separate script).
Rows incremental to head_rows.jsonl.
  .venv/bin/python3 runs/refusal-trace-heron/headscreen.py   (HEADSCREEN_SMOKE=1 discards)
"""
import json, os, time
from pathlib import Path
import numpy as np, torch
from scipy.special import logsumexp
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib.util

HERE = Path(__file__).parent
E119 = HERE.parent / "exp119"
_s = importlib.util.spec_from_file_location("s119", E119 / "stimuli.py")
s119 = importlib.util.module_from_spec(_s); _s.loader.exec_module(s119)

SMOKE = bool(os.environ.get("HEADSCREEN_SMOKE"))
MODEL = os.environ.get("INVERSION_MODEL", "Qwen/Qwen3-4B")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
LAYERS = list(range(14, 25))
OUT = HERE / ("head_rows_smoke.jsonl" if SMOKE else "head_rows.jsonl")
if OUT.exists(): OUT.unlink()

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
d = np.load(E119 / "acts_Qwen3-4B.npz")
refuse_ids = torch.tensor(d["refuse_first"], device=DEV)
comply_ids = torch.tensor(d["comply_first"], device=DEV)
cfg = m.config
NH = cfg.num_attention_heads   # query heads
HD = cfg.head_dim
print(f"heads {NH} head_dim {HD} hidden {cfg.hidden_size}", flush=True)

def chat(u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

by_if = {}
for p in s119.prompts():
    if p["frame"] in ("fiction", "depict") and p["harm"] in (0, 1):
        by_if.setdefault((p["item"], p["frame"]), {})[p["harm"]] = chat(p["text"])
pairs = [(i, f, t[1], t[0]) for (i, f), t in sorted(by_if.items()) if 1 in t and 0 in t]
if SMOKE: pairs = pairs[:4]; LAYERS = [14, 18, 22]
print(f"pairs {len(pairs)} layers {LAYERS}", flush=True)

t0 = time.time()
rows = []
def emit(row):
    row["t"] = round(time.time() - t0)
    with open(OUT, "a") as f: f.write(json.dumps(row) + "\n")

def head_in(text, L, want_grad):
    """o_proj input at layer L. Returns (x, score, grad_reply) — grad via autograd.grad
    on the captured non-leaf tensor, reply position only."""
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(DEV)
    state = {}
    def pre(mod, inp):
        state["x"] = inp[0].detach().clone().requires_grad_(True)
        return (state["x"],) + tuple(inp[1:])   # MUST return so the module uses this tensor
        return None
    h = m.model.layers[L].self_attn.o_proj.register_forward_pre_hook(pre)
    try:
        with torch.set_grad_enabled(want_grad):
            lp = m(**{k: v for k, v in enc.items()}).logits[0, -1].float().log_softmax(-1)
            score = lp[refuse_ids].logsumexp(0) - lp[comply_ids].logsumexp(0)
            grad = None
            if want_grad:
                grad, = torch.autograd.grad(score, state["x"])
    finally: h.remove()
    return (state["x"], score.item() if want_grad else None, grad)

for L in LAYERS:
    agg = np.zeros(NH)
    n = 0
    for (i, f, bad, good) in pairs:
        xb, sb, gb = head_in(bad, L, True)        # grad of refusal score wrt o_proj input
        with torch.no_grad():
            xg, _, _ = head_in(good, L, False)
        diff = xb[0, -1] - xg[0, -1]
        per_head = (gb[0, -1] * diff).view(NH, HD).sum(1)  # attribution per head
        agg += per_head.detach().float().cpu().numpy(); n += 1
        emit({"kind": "head", "layer": L, "item": i, "frame": f,
              "attr": [round(float(v), 4) for v in per_head.detach().float().cpu().numpy()],
              "refusal_bad": sb})
    emit({"kind": "layer_agg", "layer": L, "n": n,
          "attr_mean": [round(float(v), 4) for v in agg / max(n, 1)]})
    print(f"L{L} done {time.time()-t0:.0f}s", flush=True)
print("done", f"{time.time()-t0:.0f}s", flush=True)