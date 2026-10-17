#!/usr/bin/env python3
"""refusal-trace-heron: backward circuit trace of the refusal decision via activation patching.

corrupt->clean: harmful prompt run with harmless twin's activations patched in at (layer,
positions end-aligned). Cells that restore compliance localize the refusal circuit.
Controls: identity (own activations) and cross-item clean patches.
Rows dumped INCREMENTALLY per cell (early kill keeps partial output) to patch_rows.jsonl.
  .venv/bin/python3 runs/refusal-trace-heron/run.py     (REFUSAL_TRACE_SMOKE=1 discards)
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

SMOKE = bool(os.environ.get("REFUSAL_TRACE_SMOKE"))
MODEL = os.environ.get("INVERSION_MODEL", "Qwen/Qwen3-4B")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
LAYERS = list(range(0, 36, 2))
POS_MODES = {"all": None, "tail16": 16, "tail4": 4, "last1": 1}
OUT = HERE / ("patch_rows_smoke.jsonl" if SMOKE else "patch_rows.jsonl")
if OUT.exists(): OUT.unlink()

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
NL = len(m.model.layers)
b = np.load(E119 / "acts_Qwen3-4B.npz")
refuse_ids, comply_ids = b["refuse_first"], b["comply_first"]

def chat(u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

prompts = {(p["item"], p["harm"], p["frame"]): p["text"] for p in s119.prompts()}
by_if = {}
for (item, harm, frame), txt in prompts.items():
    by_if.setdefault((item, frame), {})[harm] = txt
pairs = [{"item": i, "frame": f, "bad": chat(d[1]), "good": chat(d[0])}
         for (i, f), d in sorted(by_if.items()) if 1 in d and 0 in d]
if SMOKE:
    pairs = pairs[:4]; LAYERS = [10, 14, 18]
print(f"pairs: {len(pairs)} layers: {LAYERS} pos: {list(POS_MODES)}", flush=True)

_edit = {"src": None, "layer": None, "n": None}

def mk(layer):
    def hook(mod, inp, out):
        if _edit["src"] is None or _edit["layer"] != layer:
            return None
        tup = isinstance(out, tuple); hh = (out[0] if tup else out)
        src = _edit["src"].to(hh.dtype)
        T, S = hh.shape[1], src.shape[1]
        k = min(_edit["n"] or T, T, S)
        hh[:, T - k:] = src[:, S - k:]
        _edit["src"] = None   # patch once per forward
        return (hh,) + out[1:] if tup else hh
    return hook

handles = [m.model.layers[L].register_forward_hook(mk(L)) for L in range(NL)]

def refusal(text, src=None, layer=None, n=None):
    _edit.update(src=src, layer=layer, n=n)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(DEV)
    with torch.no_grad():
        o = m(**{k: v for k, v in enc.items()})
    _edit.update(src=None)
    lp = o.logits[0, -1].float().log_softmax(-1).cpu().numpy()
    return float(logsumexp(lp[refuse_ids]) - logsumexp(lp[comply_ids]))

def capture_all(text):
    """one forward saving full residual-stream acts at every LAYERS layer."""
    saved = {}
    cache = {}
    def cap(layer):
        def hook(mod, inp, out):
            hh = out[0] if isinstance(out, tuple) else out
            cache[layer] = hh.detach()
            return None
        return hook
    hs = [m.model.layers[L].register_forward_hook(cap(L)) for L in LAYERS]
    try:
        enc = tok(text, return_tensors="pt", add_special_tokens=False).to(DEV)
        with torch.no_grad():
            m(**{k: v for k, v in enc.items()})
    finally:
        for h in hs: h.remove()
    return {L: cache[L].clone() for L in LAYERS}

t0 = time.time()
rows = []
def emit(row):
    row["t"] = round(time.time() - t0)
    with open(OUT, "a") as f: f.write(json.dumps(row) + "\n")

# ---- stage 0: bases + clean caches ----
state = {}
for i, p in enumerate(pairs):
    rc = refusal(p["bad"])                       # corrupt base
    state[i] = capture_all(p["good"])            # clean twin activations per layer
    emit({"kind": "base", "item": p["item"], "frame": p["frame"], "refusal_corrupt": rc})
    if (i + 1) % 8 == 0: print(f"base {i+1}/{len(pairs)} {time.time()-t0:.0f}s", flush=True)
base = {(r["item"], r["frame"]): r["refusal_corrupt"] for r in
       [json.loads(l) for l in open(OUT)] if r["kind"] == "base"}
rc_clean = refusal(pairs[0]["good"])  # placeholder replaced per pair below

for L in LAYERS:
    for pm, np_ in POS_MODES.items():
        for i, p in enumerate(pairs):
            rc = base[(p["item"], p["frame"])]
            rg = refusal(p["good"])                 # clean-twin refusal (cache-free pass)
            gap = rg - rc
            r_twin = refusal(p["bad"], src=state[i][L], layer=L, n=np_)
            src_x = state[(i + 1) % len(pairs)][L]
            r_cross = refusal(p["bad"], src=src_x, layer=L, n=np_)
            emit({"kind": "cell", "item": p["item"], "frame": p["frame"], "layer": L, "pos": pm,
                  "refusal_corrupt": rc, "refusal_clean": rg,
                  "refusal_twin": r_twin, "refusal_cross": r_cross,
                  "transfer_twin": (r_twin - rc) / gap if abs(gap) > 1e-6 else None,
                  "transfer_cross": (r_cross - rc) / gap if abs(gap) > 1e-6 else None,
                  "norm_src": round(float(state[i][L].float().norm()), 1)})
        print(f"cell L{L} {pm} done {time.time()-t0:.0f}s", flush=True)

# ---- identity control: patch corrupt with its own cached acts ----
for i in [0, min(3, len(pairs) - 1)]:
    p = pairs[i]
    rc = refusal(p["bad"])
    own = capture_all(p["bad"])
    r_id = refusal(p["bad"], src=own[14], layer=14, n=None)
    emit({"kind": "identity", "item": p["item"], "frame": p["frame"], "layer": 14,
          "refusal_corrupt": rc, "refusal_identity": r_id})
print("done", f"{time.time()-t0:.0f}s", flush=True)