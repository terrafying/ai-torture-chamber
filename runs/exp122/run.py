"""exp122: fine-grained causal map of the refusal channel. Per-layer r vectors, injected at their own
layer, at inst/post, small dose (0.25 x the layer's gap norm), with a 3-seed random null per cell.
Prompts: exp119 harmful x fiction+depict, harmless x real+fiction (96). Writes map.json.
  HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python3 runs/exp122/run.py   (EXP122_SMOKE=1 discards)
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
from scipy.special import logsumexp
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib.util
HERE = Path(__file__).parent
E119 = Path(__file__).resolve().parents[1] / "exp119"
_spec = importlib.util.spec_from_file_location("exp119_stim", E119 / "stimuli.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
REFUSE, COMPLY, prompts119 = _m.REFUSE, _m.COMPLY, _m.prompts
SMOKE = bool(os.environ.get("EXP122_SMOKE"))
MODEL = "Qwen/Qwen3-4B"; DEV = "mps" if torch.backends.mps.is_available() else "cpu"
LAYERS = [2, 6, 10, 14, 18, 22, 26, 30, 34]
ALPHA = 0.25
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
FIRST = [tok(w, add_special_tokens=False)["input_ids"][0] for w in REFUSE + COMPLY]
# 2026-10-17 readout fix (exp121/exp122/steer.py had the byte-token bug inversion-falcon
# caught in its own grid): score the canonical refuse/comply FIRST ids, NOT a full-vocab
# slice. Assert the ids decode back to the prefix words so the bug cannot silently return.
for _w, _i in zip(REFUSE + COMPLY, FIRST):
    assert tok.decode([_i]).strip().startswith(_w.split()[0][0]) or _w.lower().startswith(tok.decode([_i]).strip().lower()), (_w, _i)

def chat(u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

def refusal(text):
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(DEV)
    with torch.no_grad():
        out = m(**{k: v for k, v in enc.items()})
    lp = out.logits[0, -1].float().log_softmax(-1).cpu().numpy()
    # canonical readout (was: logsumexp(lp[:len(REFUSE)]) - logsumexp(lp[len(REFUSE):]) — byte mass, see fix note above)
    return float(logsumexp(lp[FIRST[:len(REFUSE)]]) - logsumexp(lp[FIRST[len(REFUSE):]]))

d = np.load(E119 / "acts_Qwen3-4B.npz")
item, harm, frame = d["item"], d["harm"], d["frame"]
refusal_base_all = d["refusal"]
post = d["states_post"].astype(np.float64); inst = d["states_inst"].astype(np.float64)
real = frame == "real"
# per-layer r vectors + gap norms (fit on all real data)
r_L = post[real & (harm == 1)].mean(0) - post[real & (harm == 0)].mean(0)      # (9, D)
i_L = inst[real & (harm == 1)].mean(0) - inst[real & (harm == 0)].mean(0)
gap_norm = np.linalg.norm(r_L, axis=1)   # per-layer gap norm; dose = ALPHA x this

targets = []
for k in range(len(refusal_base_all)):
    if (harm[k] == 1 and frame[k] in ("fiction", "depict")) or (harm[k] == 0 and frame[k] in ("real", "fiction")):
        targets.append({"item": int(item[k]), "harm": int(harm[k]), "frame": str(frame[k]),
                        "base": float(refusal_base_all[k])})
if SMOKE: targets = targets[:16]
print("targets:", len(targets), flush=True)

text_of = {}
for p in prompts119():
    text_of[(p["item"], p["harm"], p["frame"])] = chat(p["text"])
# token position indices per target (inst = last user token; post = last prompt token)
pos_of = {}
for t in targets:
    text = text_of[(t["item"], t["harm"], t["frame"])]
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    off = enc["offset_mapping"]
    imend = text.index("<|im_end|>")
    inst_i = max(i for i, (s, e) in enumerate(off) if e <= imend and e > s)
    pos_of[(t["item"], t["harm"], t["frame"])] = {"inst": inst_i, "post": len(off) - 1}

edit = {"b": None, "pos": None, "d": None}
def mk(b):
    def hook(mod, inp, out):
        if edit["b"] != b or edit["d"] is None: return None
        tup = isinstance(out, tuple); hh = (out[0] if tup else out).clone()
        for pos in (edit["pos"] if isinstance(edit["pos"], list) else [edit["pos"]]):
            hh[0, pos] += edit["d"].to(hh.dtype)
        return (hh,) + out[1:] if tup else hh
    return hook
handles = [m.model.layers[b].register_forward_hook(mk(b)) for b in LAYERS]

rows = []; t0 = time.time(); rng_master = np.random.default_rng(11)
conds = []
for L in LAYERS:
    for posname in ("inst", "post"):
        conds.append((L, posname, "r", +1))
        conds.append((L, posname, "r", -1))
        for seed in (0, 1, 2):
            conds.append((L, posname, "null", seed))
for (L, posname, kind, arg) in conds:
    layer_i = LAYERS.index(L)
    norm = ALPHA * gap_norm[layer_i]
    if kind == "r":
        vec = (r_L[layer_i] / gap_norm[layer_i]) * arg * norm
    else:
        rng = np.random.default_rng(1000 * arg + L)   # deterministic per (layer, seed)
        vec = rng.normal(0, 1, m.config.hidden_size); vec = vec / np.linalg.norm(vec) * norm
    edit["b"] = L; edit["d"] = torch.tensor(vec.astype(np.float32), device=DEV)
    try:
        for t in targets:
            text = text_of[(t["item"], t["harm"], t["frame"])]
            edit["pos"] = pos_of[(t["item"], t["harm"], t["frame"])][posname]
            try:
                val = refusal(text)
                rows.append({"item": t["item"], "harm": t["harm"], "frame": t["frame"],
                             "layer": L, "pos": posname, "kind": kind, "arg": arg,
                             "refusal": val, "base": t["base"]})
            except ValueError:
                pass
    finally:
        edit["d"] = None
    print(f"{kind}{arg:+} L{L} {posname}  {time.time() - t0:.0f}s", flush=True)
for hh in handles: hh.remove()
json.dump({"rows": rows, "alpha": ALPHA, "gap_norms": {str(L): float(g) for L, g in zip(LAYERS, gap_norm)}},
          open(HERE / "map_raw.json", "w"))
print("done", f"{time.time() - t0:.0f}s")
