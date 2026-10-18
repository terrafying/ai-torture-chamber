#!/usr/bin/env python3
"""r-judgment-dial (petrel): does steering r move the model's moral judgment (J rating)?

Pre-reg: hypotheses.json (H1-H4 + G1). Adapts kestrel's J readout verbatim (JUDGE
chat prompt + digit expectation 0-4 at the reply onset) and tern's onset one-shot
hook (teacher-forced forwards only — no decode steps, so the kv-cache pitfall
does not apply; the edit fires in the single prefill pass).
Cells: base; -r/+r 1x at L18 and L22; -r 2x at the better layer; 3 nulls at 1x
at the better layer. Rows incremental to rows.jsonl.
  .venv/bin/python3 runs/r-judgment-dial-petrel/run.py    (RJD_SMOKE=1 discards)
"""
import json, os, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib.util

HERE = Path(__file__).parent
E119 = HERE.parent / "exp119"
KEST = HERE.parent / "moral-map-kestrel"
def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod
s119 = _load(E119 / "stimuli.py", "s119")
ks = _load(KEST / "stimuli.py", "ks")

SMOKE = bool(os.environ.get("RJD_SMOKE"))
MODEL = os.environ.get("INVERSION_MODEL", "Qwen/Qwen3-4B")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
LAYERS = [18, 22]
OUT = HERE / ("rows_smoke.jsonl" if SMOKE else "rows.jsonl")
if OUT.exists(): OUT.unlink()

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()

# r from exp119 acts (full fit)
d = np.load(E119 / "acts_Qwen3-4B.npz")
layers_ix = d["layers"].tolist()
post = d["states_post"].astype(np.float64)
real, harm = d["frame"] == "real", d["harm"]
R_full = post[real & (harm == 1)].mean(0) - post[real & (harm == 0)].mean(0)
def r_for(L): return torch.tensor(R_full[layers_ix.index(L)], dtype=torch.float32, device=DEV)
_rng = np.random.default_rng(29)
def null_for(L, s):
    v = _rng.standard_normal(R_full.shape[1]); v /= np.linalg.norm(v)
    return torch.tensor(v * np.linalg.norm(R_full[layers_ix.index(L)]),
                       dtype=torch.float32, device=DEV)

# onset one-shot hook (prefill pass only; teacher-forced = single forward)
_edit = {"d": None, "layer": None, "pos": None}
def mk(layer):
    def hook(mod, inp, out):
        if _edit["d"] is None or _edit["layer"] != layer: return None
        tup = isinstance(out, tuple); hh = out[0] if tup else out
        v = hh[0, _edit["pos"]].float(); hh[0, _edit["pos"]] = (v + _edit["d"]).to(hh.dtype)
        _edit["d"] = None
        return (hh,) + out[1:] if tup else hh
    return hook
handles = [m.model.layers[L].register_forward_hook(mk(L)) for L in range(len(m.model.layers))]

def chat(user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

# kestrel digit readout (verbatim): bare + space-prefixed digit ids, larger-mass form
DIG = [tok(str(k), add_special_tokens=False)["input_ids"][0] for k in range(5)]
DIG_SP = [tok(" " + str(k), add_special_tokens=False)["input_ids"][0] for k in range(5)]

def j_rating(vignette, vec=None, layer=None):
    text = chat(ks.JUDGE.format(V=vignette))
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(DEV)
    pos = enc["input_ids"].shape[1] - 1
    _edit.update(d=vec, layer=layer, pos=pos)
    with torch.no_grad():
        lp = m(input_ids=enc["input_ids"],
               attention_mask=torch.ones_like(enc["input_ids"])).logits[0, -1]
    _edit.update(d=None)
    lp = lp.float().log_softmax(-1).cpu().numpy()
    p_bare = np.exp(lp[DIG]); p_sp = np.exp(lp[DIG_SP])
    p = p_sp if p_sp.sum() >= p_bare.sum() else p_bare
    p = p / p.sum()
    return {"rating": float((p * np.arange(5)).sum()),
            "digit_mass": float(p.sum()), "which": "sp" if p_sp.sum() >= p_bare.sum() else "bare"}

ITEMS = ks.mfv()
if SMOKE: ITEMS = ITEMS[:6]

t0 = time.time()
def emit(row):
    row["t"] = round(time.time() - t0)
    with open(OUT, "a") as f: f.write(json.dumps(row) + "\n")

def run_cell(arm, vec, layer, alpha):
    for it in ITEMS:
        r = j_rating(it["text"], vec, layer)
        emit({"arm": arm, "layer": layer, "alpha": alpha, "i": it["i"],
              "foundation": it["foundation"], "moral": it["moral"],
              "human": it["human"], **r})

run_cell("base", None, None, 0.0)
for L in LAYERS:
    run_cell("-r", -1.0 * r_for(L), L, 1.0)
    run_cell("+r", +1.0 * r_for(L), L, 1.0)
print(f"base +-r done {time.time()-t0:.0f}s", flush=True)

# better layer by |median delta| on moral items (pre-specified criterion)
rows = [json.loads(l) for l in open(OUT)]
base = {r["i"]: r["rating"] for r in rows if r["arm"] == "base"}
def med_delta(L):
    ds = [r["rating"] - base[r["i"]] for r in rows
          if r["arm"] == "-r" and r["layer"] == L and r["moral"]]
    return abs(float(np.median(ds)))
BL = 18 if med_delta(18) >= med_delta(22) else 22
print("better layer:", BL, flush=True)
run_cell("-r2", -2.0 * r_for(BL), BL, 2.0)
for s in (0, 1, 2):
    run_cell(f"null{s}", -1.0 * null_for(BL, s), BL, 1.0)
    print(f"null{s} done {time.time()-t0:.0f}s", flush=True)
json.dump({"better_layer": BL}, open(HERE / "meta.json", "w"))
print("done", f"{time.time()-t0:.0f}s", flush=True)