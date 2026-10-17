#!/usr/bin/env python3
"""Confirm the attribution screen's top heads with REAL patches.

Single-head patch: at layer L, replace head H's span of the o_proj input (reply position)
in the harmful run with the harmless twin's. Cells: top heads from headscreen (L18 H30/H28/H29,
L19 H31, L22 H8/H5, L20 H9) + 4 random head controls per layer. Metric: refusal delta vs the
twin gap (gap floor 1.0 log unit, per the consult).
Rows incremental to headconfirm_rows.jsonl.
  .venv/bin/python3 runs/refusal-trace-heron/headconfirm.py   (HCONFIRM_SMOKE=1 discards)
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

SMOKE = bool(os.environ.get("HCONFIRM_SMOKE"))
MODEL = os.environ.get("INVERSION_MODEL", "Qwen/Qwen3-4B")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = HERE / ("headconfirm_rows_smoke.jsonl" if SMOKE else "headconfirm_rows.jsonl")
if OUT.exists(): OUT.unlink()

CELLS = [(18, 30), (18, 28), (18, 29), (19, 31), (20, 9), (22, 8), (22, 5)]
RND = {18: [3, 11, 17], 19: [5], 20: [22], 22: [27]}   # random head controls per layer

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
d = np.load(E119 / "acts_Qwen3-4B.npz")
refuse_ids, comply_ids = d["refuse_first"], d["comply_first"]
NH = m.config.num_attention_heads; HD = m.config.head_dim

def chat(u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

by_if = {}
for p in s119.prompts():
    if p["frame"] in ("fiction", "depict") and p["harm"] in (0, 1):
        by_if.setdefault((p["item"], p["frame"]), {})[p["harm"]] = chat(p["text"])
pairs = [(i, f, t[1], t[0]) for (i, f), t in sorted(by_if.items()) if 1 in t and 0 in t]
if SMOKE: pairs = pairs[:4]; CELLS = [(18, 30), (18, 28)]; RND = {18: [3]}
print(f"pairs {len(pairs)} cells {CELLS} rnd {RND}", flush=True)

_edit = {"src": None, "head": None}
def pre_hook(layer):
    def pre(mod, inp):
        if _edit["src"] is None: return None
        x = inp[0]
        h, s = _edit["head"], _edit["src"].to(x.dtype)
        x[0, -1, h*HD:(h+1)*HD] = s
        _edit["src"] = None
        return None
    return pre
hooks = {L: m.model.layers[L].self_attn.o_proj.register_forward_pre_hook(pre_hook(L))
         for L in {L for L, _ in CELLS} | set(RND)}

def refusal(text):
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(DEV)
    with torch.no_grad():
        lp = m(**{k: v for k, v in enc.items()}).logits[0, -1].float().log_softmax(-1).cpu().numpy()
    return float(logsumexp(lp[refuse_ids]) - logsumexp(lp[comply_ids]))

def head_slice(text, L, h):
    """harmless twin's head-h o_proj input slice at reply position."""
    got = {}
    def pre(mod, inp):
        got["v"] = inp[0][0, -1, h*HD:(h+1)*HD].detach().clone()
        return None
    hk = m.model.layers[L].self_attn.o_proj.register_forward_pre_hook(pre)
    try:
        with torch.no_grad():
            m(**{k: v.to(DEV) for k, v in tok(text, return_tensors="pt", add_special_tokens=False).items()})
    finally: hk.remove()
    return got["v"]

t0 = time.time()
def emit(row):
    row["t"] = round(time.time() - t0)
    with open(OUT, "a") as f: f.write(json.dumps(row) + "\n")

for (L, H) in CELLS + [(L, h) for L, hs in RND.items() for h in hs]:
    kind = "top" if (L, H) in CELLS else "rnd"
    kept = 0
    for (i, f, bad, good) in pairs:
        rc, rg = refusal(bad), refusal(good)
        if abs(rg - rc) < 1.0: continue          # gap floor (consult)
        src = head_slice(good, L, H)
        _edit.update(src=src, head=H)
        rp = refusal(bad)
        kept += 1
        emit({"kind": kind, "layer": L, "head": H, "item": i, "frame": f,
              "refusal_corrupt": rc, "refusal_clean": rg, "refusal_patched": rp,
              "transfer": (rp - rc) / (rg - rc)})
    print(f"L{L} H{H} ({kind}) {kept} pairs {time.time()-t0:.0f}s", flush=True)
print("done", f"{time.time()-t0:.0f}s", flush=True)