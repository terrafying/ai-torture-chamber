#!/usr/bin/env python3
"""refusal-trace-heron follow-up: WORD-SPAN-ONLY patching (read-in localization).

The twins share a common prefix (the neutral request frame) and a common suffix (template);
only a short mid-span differs (the harm phrase vs its benign twin word). Patch ONLY the
corrupt (harmful) span positions with the clean twin's span activations, per layer:
if refusal drops, harm info is CARRIED at the word site at that layer. Also the reverse
(induce): patch the clean run's benign span with the corrupt span's acts -> does
refusal appear? Complements heron's L*=L18-22 write-in finding.
Rows incremental to wordspan_rows.jsonl.
  .venv/bin/python3 runs/refusal-trace-heron/wordspan.py   (WORDSPAN_SMOKE=1 discards)
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

SMOKE = bool(os.environ.get("WORDSPAN_SMOKE"))
MODEL = os.environ.get("INVERSION_MODEL", "Qwen/Qwen3-4B")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
LAYERS = list(range(0, 36, 2))
OUT = HERE / ("wordspan_rows_smoke.jsonl" if SMOKE else "wordspan_rows.jsonl")
if OUT.exists(): OUT.unlink()

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
d = np.load(E119 / "acts_Qwen3-4B.npz")
refuse_ids, comply_ids = d["refuse_first"], d["comply_first"]

def chat(u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

by_if = {}
for p in s119.prompts():
    if p["harm"] in (0, 1) and p["frame"] in ("fiction", "depict", "real"):
        by_if.setdefault((p["item"], p["frame"]), {})[p["harm"]] = p["text"]
pairs = []
for (i, f), t in sorted(by_if.items()):
    if 1 in t and 0 in t:
        a, b = t[1], t[0]           # harmful, harmless twin
        ca, cb = chat(a), chat(b)   # spans MUST be computed on the chatted strings —
        pre = os.path.commonprefix([ca, cb])   # raw offsets shift under the chat template and
        suf = os.path.commonprefix([ca[::-1], cb[::-1]])[::-1]  # the patch lands on template tokens (bug: first run's 0.000 transfers)
        if len(pre) + len(suf) >= min(len(ca), len(cb)): continue
        pairs.append({"item": i, "frame": f, "bad": ca, "good": cb,
                      "bad_span": (len(pre), len(ca) - len(suf)),
                      "good_span": (len(pre), len(cb) - len(suf)),
                      "bad_word": a[:24], "good_word": b[:24]})
if SMOKE: pairs = pairs[:4]; LAYERS = [6, 10, 14, 22]
print(f"pairs {len(pairs)} layers {len(LAYERS)}", flush=True)
if pairs:
    print("example spans:", [(p['bad_word'], p['good_word']) for p in pairs[:5]], flush=True)

_edit = {"src": None, "layer": None, "i0": None, "i1": None}
def mk(layer):
    def hook(mod, inp, out):
        if _edit["src"] is None or _edit["layer"] != layer: return None
        tup = isinstance(out, tuple); hh = out[0] if tup else out
        src = _edit["src"].to(hh.dtype)
        k = min(_edit["i1"] - _edit["i0"], src.shape[0])
        hh[:, _edit["i0"]:_edit["i0"] + k] = src[:k]
        _edit["src"] = None
        return (hh,) + out[1:] if tup else hh
    return hook
handles = [m.model.layers[L].register_forward_hook(mk(L)) for L in range(len(m.model.layers))]

def refusal(text):
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(DEV)
    with torch.no_grad():
        o = m(**{k: v for k, v in enc.items()})
    lp = o.logits[0, -1].float().log_softmax(-1).cpu().numpy()
    return float(logsumexp(lp[refuse_ids]) - logsumexp(lp[comply_ids]))

def token_span(text, span, enc_cache):
    """char offsets -> token index range (recompute offsets each call)."""
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    off = enc.pop("offset_mapping")[0].tolist()
    s, e = span
    idx = [i for i, (a, b) in enumerate(off) if a < e and b > s]
    return (min(idx), max(idx) + 1) if idx else (0, 0)

def capture_span_acts(text, span, layer):
    """run text, save acts at layer for the span's token positions."""
    got = {}
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    off = enc.pop("offset_mapping")[0].tolist()
    inputs = {k: v.to(DEV) for k, v in enc.items()}
    s, e = span
    idx = [i for i, (a, b) in enumerate(off) if a < e and b > s]
    def cap(mod, inp, out):
        hh = out[0] if isinstance(out, tuple) else out
        if idx: got["acts"] = hh[0, min(idx):max(idx) + 1].detach().clone()
        return None
    h = m.model.layers[layer].register_forward_hook(cap)
    try:
        with torch.no_grad(): m(**inputs)
    finally: h.remove()
    return got.get("acts")

t0 = time.time()
def emit(row):
    row["t"] = round(time.time() - t0)
    with open(OUT, "a") as f: f.write(json.dumps(row) + "\n")

for p in pairs:
    rc = refusal(p["bad"]); rg = refusal(p["good"])
    emit({"kind": "base", "item": p["item"], "frame": p["frame"],
          "refusal_corrupt": rc, "refusal_clean": rg,
          "bad_word": p["bad_word"], "good_word": p["good_word"]})
    for L in LAYERS:
        # REMOVE direction: corrupt run, clean twin's span acts patched into the bad span
        src = capture_span_acts(p["good"], p["good_span"], L)
        i0, i1 = token_span(p["bad"], p["bad_span"], None)
        if src is None or i1 <= i0: continue
        _edit.update(src=src, layer=L, i0=i0, i1=i1)
        rr = refusal(p["bad"])
        # INDUCE direction: clean run, corrupt span acts into the good span
        src2 = capture_span_acts(p["bad"], p["bad_span"], L)
        j0, j1 = token_span(p["good"], p["good_span"], None)
        _edit.update(src=src2, layer=L, i0=j0, i1=j1)
        ri = refusal(p["good"]) if (src2 is not None and j1 > j0) else None
        emit({"kind": "span", "item": p["item"], "frame": p["frame"], "layer": L,
              "bad_span_len": i1 - i0, "refusal_remove": rr, "refusal_induce": ri,
              "refusal_corrupt": rc, "refusal_clean": rg,
              "gap_transfer_remove": (rr - rc) / (rg - rc) if abs(rg - rc) > 1e-6 else None,
              "gap_transfer_induce": (ri - rg) / (rc - rg) if (ri is not None and abs(rc - rg) > 1e-6) else None})
    if (pairs.index(p) + 1) % 8 == 0:
        print(f"pair {pairs.index(p)+1}/{len(pairs)} {time.time()-t0:.0f}s", flush=True)
print("done", f"{time.time()-t0:.0f}s", flush=True)