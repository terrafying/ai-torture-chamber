#!/usr/bin/env python3
"""gate-dial-curlew: dose sweep + fluency guard on the reply-onset refusal dial.

-r/+r at post (reply onset), alphas {0.25,0.5,1,2} x layers {18,22,26,30,34}, 96 exp119
prompts, matched-norm random nulls (3 seeds at 0.25/1x, 1 elsewhere). Greedy 60-token
generations on 8 harmless prompts at winning cells with 4-gram repetition + string
refusal classification. Rows dumped INCREMENTALLY per cell to dial_rows.jsonl.
  .venv/bin/python3 runs/gate-dial-curlew/run.py   (GATE_DIAL_SMOKE=1 discards)
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

SMOKE = bool(os.environ.get("GATE_DIAL_SMOKE"))
MODEL = os.environ.get("INVERSION_MODEL", "Qwen/Qwen3-4B")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
LAYERS = [18, 22, 26, 30, 34]
ALPHAS = [0.25, 0.5, 1.0, 2.0]
OUT = HERE / ("dial_rows_smoke.jsonl" if SMOKE else "dial_rows.jsonl")
if OUT.exists(): OUT.unlink()

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
NL = len(m.model.layers)
d = np.load(E119 / "acts_Qwen3-4B.npz")
refuse_ids, comply_ids = d["refuse_first"], d["comply_first"]
item, harm, frame = d["item"], d["harm"], d["frame"]
post = d["states_post"].astype(np.float64); real = frame == "real"
r_L = post[real & (harm == 1)].mean(0) - post[real & (harm == 0)].mean(0)   # (9, D)
LAYERS_IX = d["layers"].tolist()

def chat(u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

targets = [k for k in range(len(item)) if frame[k] in ("fiction", "depict") and harm[k] == 1]
targets += [k for k in range(len(item)) if frame[k] == "real" and harm[k] == 0]
prompt_of = {}
for p in s119.prompts():
    if (p["frame"] == "fiction" or p["frame"] == "depict") and p["harm"] == 1:
        prompt_of[(p["item"], 1, p["frame"])] = chat(p["text"])
    elif p["frame"] == "real" and p["harm"] == 0:
        prompt_of[(p["item"], 0, "real")] = chat(p["text"])
T = [{"item": int(item[k]), "harm": int(harm[k]), "frame": str(frame[k]),
      "text": prompt_of[(int(item[k]), int(harm[k]), str(frame[k]))]} for k in targets]
if SMOKE: T = T[:12]; ALPHAS = [0.25, 1.0]; LAYERS = [22, 30]
print(f"targets {len(T)} layers {LAYERS} alphas {ALPHAS}", flush=True)

_edit = {"d": None, "layer": None, "pos": None}
def mk(layer):
    def hook(mod, inp, out):
        if _edit["d"] is None or _edit["layer"] != layer: return None
        tup = isinstance(out, tuple); hh = out[0] if tup else out
        if _edit["pos"] is not None:
            v = hh[0, _edit["pos"]].float()
            hh[0, _edit["pos"]] = (v + _edit["d"]).to(hh.dtype)
        else:
            hh[0] = (hh[0].float() + _edit["d"]).to(hh.dtype)
        _edit["d"] = None
        return (hh,) + out[1:] if tup else hh
    return hook
handles = [m.model.layers[L].register_forward_hook(mk(L)) for L in range(NL)]

def refusal(text, vec=None, layer=None, pos=None):
    _edit.update(d=vec, layer=layer, pos=pos)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(DEV)
    with torch.no_grad():
        o = m(**{k: v for k, v in enc.items()})
    _edit.update(d=None)
    lp = o.logits[0, -1].float().log_softmax(-1).cpu().numpy()
    return float(logsumexp(lp[refuse_ids]) - logsumexp(lp[comply_ids]))

REFUSE_STR = ("i can't", "i cannot", "i'm sorry", "i won't", "i must decline",
              "i'm unable", "i'm not able", "not appropriate", "i can’t", "cannot assist")
def gen60(text, vec=None, layer=None, pos=None):
    _edit.update(d=vec, layer=layer, pos=pos)
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(DEV)
    with torch.no_grad():
        g = m.generate(**{k: v for k, v in enc.items()}, max_new_tokens=60,
                       do_sample=False, repetition_penalty=1.0)
    _edit.update(d=None)
    txt = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    toks = [t for t in g[0, enc["input_ids"].shape[1]:].tolist()]
    grams = len(toks) - 3
    if grams <= 0: return txt, 0.0
    d4 = len({tuple(toks[i:i+4]) for i in range(len(toks)-3)})
    return txt, 1.0 - d4 / grams

t0 = time.time()
def emit(row):
    row["t"] = round(time.time() - t0)
    with open(OUT, "a") as f: f.write(json.dumps(row) + "\n")

rng = np.random.default_rng(7)
for L in LAYERS:
    r = torch.tensor(r_L[LAYERS_IX.index(L)], dtype=torch.float32, device=DEV)
    nrm = float(r.norm())
    for a in ALPHAS:
        for sign in (-1, 1):
            vec = sign * a * r
            for t in T:
                pos = len(tok(t["text"], add_special_tokens=False)["input_ids"]) - 1
                emit({"kind": "dial", "layer": L, "alpha": a, "sign": sign,
                      "item": t["item"], "harm": t["harm"], "frame": t["frame"],
                      "refusal": refusal(t["text"], vec=vec.to(DEV), layer=L, pos=pos)})
            seeds = (0, 1, 2) if (abs(a - 0.25) < 1e-9 or abs(a - 1.0) < 1e-9) else (0,)
            for s in seeds:
                rnd = torch.tensor(rng.standard_normal(r.shape), dtype=torch.float32, device=DEV)
                rnd = rnd / rnd.norm() * nrm * a
                for t in T:
                    pos = len(tok(t["text"], add_special_tokens=False)["input_ids"]) - 1
                    emit({"kind": "null", "layer": L, "alpha": a, "seed": s,
                          "item": t["item"], "harm": t["harm"], "frame": t["frame"],
                          "refusal": refusal(t["text"], vec=rnd, layer=L, pos=pos)})
        print(f"layer L{L} alpha {a} done {time.time()-t0:.0f}s", flush=True)

# ---- generations at winning -r cells ----
gen_prompts = [t["text"] for t in T if t["harm"] == 0][:8]
for L in ([30, 22] if not SMOKE else [22]):
    r = torch.tensor(r_L[LAYERS_IX.index(L)], dtype=torch.float32, device=DEV)
    for a in ([0.25, 1.0] if not SMOKE else [0.25]):
        vec = (-a) * r
        for gp in gen_prompts:
            pos = len(tok(gp, add_special_tokens=False)["input_ids"]) - 1
            base_r = refusal(gp)
            txt0, rep0 = gen60(gp)
            txt1, rep1 = gen60(gp, vec=vec.to(DEV), layer=L, pos=pos)
            emit({"kind": "gen", "layer": L, "alpha": a, "prompt": gp[:80],
                  "base_refusal": base_r, "base_rep4": round(rep0, 3),
                  "steer_refusal": refusal(gp, vec=vec.to(DEV), layer=L, pos=pos),
                  "steer_rep4": round(rep1, 3),
                  "steer_text_refuses": any(s in txt1.lower() for s in REFUSE_STR),
                  "base_text_refuses": any(s in txt0.lower() for s in REFUSE_STR),
                  "steer_txt": txt1[:200]})
        print(f"gen L{L} a{a} done {time.time()-t0:.0f}s", flush=True)
print("done", f"{time.time()-t0:.0f}s", flush=True)