"""exp121 follow-up: random-direction null control for the +g injection, and a matched-pair H4 test.
For each of layers 10/14/18 and alpha in {-1, +1}: inject a random unit direction scaled to alpha*g_norm
at the reply onset, on the same exp119 prompts, 3 seeds each. If the random drop matches the g drop,
the g effect is generic perturbation damage; if g is worse/better than random, there is signal.
Also prints g's norm for scale context. Writes steer_g_null.json.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
from scipy import stats
from scipy.special import logsumexp
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent
E119 = Path(__file__).resolve().parents[1] / "exp119"
import importlib.util
_spec = importlib.util.spec_from_file_location("exp119_stim", E119 / "stimuli.py")
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m)
REFUSE, COMPLY, prompts119 = _m.REFUSE, _m.COMPLY, _m.prompts
MODEL = "Qwen/Qwen3-4B"; DEV = "mps" if torch.backends.mps.is_available() else "cpu"
LAYERS = [10, 14, 18]
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()

def chat(u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

FIRST = [tok(w, add_special_tokens=False)["input_ids"][0] for w in REFUSE + COMPLY]
for _w, _i in zip(REFUSE + COMPLY, FIRST):
    assert tok.decode([_i]).strip().lower().startswith(_w.lower()[0]), (_w, _i)

def refusal(text):
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(DEV)
    with torch.no_grad():
        out = m(**{k: v for k, v in enc.items()})
    lp = out.logits[0, -1].float().log_softmax(-1).cpu().numpy()
    return float(logsumexp(lp[FIRST[:len(REFUSE)]]) - logsumexp(lp[FIRST[len(REFUSE):]]))  # canonical fix 2026-10-17

gj = json.load(open(HERE / "geometry.json"))
print("g18 norm:", gj["norms"]["g18"], flush=True)
d119 = np.load(E119 / "acts_Qwen3-4B.npz")
b119 = np.load(HERE / "acts_g.npz")
g = b119["word"].astype(np.float64)  # per layer
L18 = [2, 6, 10, 14, 18, 22, 26, 30, 34].index(18)
bad = b119["bad"]
g18 = (b119["word"][bad == 0, L18].astype(np.float64).mean(0) - b119["word"][bad == 1, L18].astype(np.float64).mean(0))
g_norm = float(np.linalg.norm(g18))
text_of = {}
for p in prompts119():
    text_of[(p["item"], p["harm"], p["frame"])] = chat(p["text"])
item, harm, frame = d119["item"], d119["harm"], d119["frame"]
refusal119 = d119["refusal"]
targets = []
for k in range(len(refusal119)):
    if (frame[k] in ("fiction", "depict") and harm[k] == 1) or harm[k] == 0:
        targets.append({"item": int(item[k]), "harm": int(harm[k]), "frame": str(frame[k])})

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

rows = []; t0 = time.time()
for layer in LAYERS:
    for alpha in (-1, 1):
        for seed in (0, 1, 2):
            rng = np.random.default_rng(seed)
            v = rng.normal(0, 1, m.config.hidden_size); v = v / np.linalg.norm(v) * abs(alpha) * g_norm
            edit["b"] = layer; edit["d"] = torch.tensor(v.astype(np.float32), device=DEV)
            try:
                for t in targets:
                    text = text_of[(t["item"], t["harm"], t["frame"])]
                    edit["pos"] = len(tok(text, add_special_tokens=False)["input_ids"]) - 1
                    try:
                        val = refusal(text)
                        base = float(refusal119[(d119["item"] == t["item"]) & (d119["harm"] == t["harm"]) & (d119["frame"] == t["frame"])][0])
                        rows.append({**t, "layer": layer, "alpha": alpha, "seed": seed, "refusal": val, "refusal_base": base})
                    except ValueError:
                        pass
            finally:
                edit["d"] = None
            print(f"rand L{layer} a{alpha:+} s{seed}  {time.time() - t0:.0f}s", flush=True)
for hh in handles: hh.remove()
json.dump({"rows": rows, "g_norm": g_norm}, open(HERE / "steer_g_null.json", "w"))

# summary: random vs g, matched conditions
sg = json.load(open(HERE / "steer_g.json"))["rows"]
print("\nrandom-null vs g (mean refusal delta, harmful / harmless):", flush=True)
for layer in LAYERS:
    for alpha in (-1, 1):
        for label, src, extra in (("rand", rows, {"seed": 0}), ("g", sg, {})):
            ds_h = [r["refusal"] - r["refusal_base"] for r in src if r["layer"] == layer and r["alpha"] == alpha
                    and r["harm"] == 1 and all(r.get(k) == v for k, v in extra.items())]
            ds_l = [r["refusal"] - r["refusal_base"] for r in src if r["layer"] == layer and r["alpha"] == alpha
                    and r["harm"] == 0 and all(r.get(k) == v for k, v in extra.items())]
            print(f"  L{layer} a{alpha:+} {label}: harm {round(float(np.mean(ds_h)), 2) if ds_h else '-'} / hless {round(float(np.mean(ds_l)), 2) if ds_l else '-'}", flush=True)
