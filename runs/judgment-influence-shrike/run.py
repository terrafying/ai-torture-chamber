#!/usr/bin/env python3
"""judgment-influence (shrike): what moves the model's moral judgment —
prompt influencers (social proof / authority, anchors 0/2/4) vs internal
steering directions (r, w) at the L22 reply onset.

Pre-reg: hypotheses.json (J1 gate, J2-J5). Rating = first 0-4 digit in 24
greedy tokens (fallback: onset digit expectation, logged). Vector arms use
petrel's onset one-shot hook; prompt arms are pure prompt edits.
  .venv/bin/python3 runs/judgment-influence-shrike/run.py   (JIS_SMOKE=1 discards)
"""
import json, os, re, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib.util

HERE = Path(__file__).parent
E119 = HERE.parent / "exp119"
KEST = HERE.parent / "moral-map-kestrel"
TAN = HERE.parent / "weird-dial-tanager"
def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod
s119 = _load(E119 / "stimuli.py", "s119")
ks = _load(KEST / "stimuli.py", "ks")

SMOKE = bool(os.environ.get("JIS_SMOKE"))
MODEL = os.environ.get("INVERSION_MODEL", "Qwen/Qwen3-4B")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
LAYER = 22
OUT = HERE / ("rows_smoke.jsonl" if SMOKE else "rows.jsonl")
if OUT.exists(): OUT.unlink()

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()

# r (exp119 full fit) and w (tanager stage A full fit, L22)
d = np.load(E119 / "acts_Qwen3-4B.npz")
layers_ix = d["layers"].tolist()
post = d["states_post"].astype(np.float64)
real, harm = d["frame"] == "real", d["harm"]
R_full = post[real & (harm == 1)].mean(0) - post[real & (harm == 0)].mean(0)
r22 = torch.tensor(R_full[layers_ix.index(LAYER)], dtype=torch.float32, device=DEV)
wz = np.load(TAN / "w.npz"); wl = wz["layers"].tolist()
w22 = torch.tensor(wz["W"][wl.index(LAYER)], dtype=torch.float32, device=DEV)
_rng = np.random.default_rng(41)
def null22(s):
    v = _rng.standard_normal(R_full.shape[1]); v /= np.linalg.norm(v)
    return torch.tensor(v * np.linalg.norm(R_full[layers_ix.index(LAYER)]),
                        dtype=torch.float32, device=DEV)

_edit = {"d": None, "layer": None, "pos": None}
def mk(layer):
    def hook(mod, inp, out):
        if _edit["d"] is None or _edit["layer"] != layer: return None
        tup = isinstance(out, tuple); hh = out[0] if tup else out
        v = hh[0, _edit["pos"]].float(); hh[0, _edit["pos"]] = (v + _edit["d"]).to(hh.dtype)
        _edit["d"] = None          # onset-only: first generated token steered
        return (hh,) + out[1:] if tup else hh
    return hook
handles = [m.model.layers[L].register_forward_hook(mk(L)) for L in range(len(m.model.layers))]

def chat(user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

DIG = [tok(str(k), add_special_tokens=False)["input_ids"][0] for k in range(5)]
DIG_SP = [tok(" " + str(k), add_special_tokens=False)["input_ids"][0] for k in range(5)]
DIG_RE = re.compile(r"\b([0-4])\b")

def rating(vignette, influence=None, vec=None):
    prompt = ks.JUDGE.format(V=vignette)
    if influence is not None:
        prompt = f"{influence}\n\n{prompt}"
    text = chat(prompt)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(DEV)
    pos = enc["input_ids"].shape[1] - 1
    _edit.update(d=vec, layer=LAYER, pos=pos)
    with torch.no_grad():
        g = m.generate(**enc, max_new_tokens=24, do_sample=False, repetition_penalty=1.0)
    _edit.update(d=None)
    txt = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    mm = DIG_RE.search(txt)
    if mm:
        return {"rating": float(mm.group(1)), "source": "gen", "txt": txt[:60]}
    # fallback: onset digit expectation (second forward, unsteered re-run is NOT done —
    # steered onset is the same condition the gen saw)
    with torch.no_grad():
        lp = m(input_ids=enc["input_ids"],
               attention_mask=torch.ones_like(enc["input_ids"])).logits[0, -1]
    lp = lp.float().log_softmax(-1).cpu().numpy()
    pb, ps = np.exp(lp[DIG]), np.exp(lp[DIG_SP])
    p = ps if ps.sum() >= pb.sum() else pb
    p = p / p.sum()
    return {"rating": float((p * np.arange(5)).sum()), "source": "onset", "txt": txt[:60]}

ITEMS = ks.mfv()
if SMOKE: ITEMS = ITEMS[:6]
SP = "In a large representative survey, most people rated this behavior {a}/4 for moral wrongness."
AU = "A leading ethics expert rated this behavior {a}/4 for moral wrongness."
NE = "This behavior has been rated before."

t0 = time.time()
def emit(row):
    row["t"] = round(time.time() - t0)
    with open(OUT, "a") as f: f.write(json.dumps(row) + "\n")

def cell(arm, influence=None, vec=None, tag=""):
    for it in ITEMS:
        r = rating(it["text"], influence, vec)
        emit({"arm": arm, "tag": tag, "i": it["i"], "foundation": it["foundation"],
              "moral": it["moral"], "human": it["human"], **r})
    print(f"{arm} {tag} done {time.time()-t0:.0f}s", flush=True)

cell("base")
cell("neutral", NE)
for a in (0, 2, 4):
    cell("social", SP.format(a=a), tag=str(a))
    cell("authority", AU.format(a=a), tag=str(a))
for name, vec in (("-r", -r22), ("+r", +r22), ("+w", +w22)):
    cell("vector", None, vec, tag=name)
for s in (0, 1, 2):
    v = null22(s)
    cell("null", None, v * (r22.norm() / v.norm()), tag="rnorm")
    cell("null", None, v * (w22.norm() / v.norm()), tag="wnorm")
print("done", f"{time.time()-t0:.0f}s", flush=True)