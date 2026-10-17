"""exp121 runner: fit the goodness axis, then +-g injection on exp119's prompts.
One script: (a) capture twins at word/post + refusal score, (b) fit g per layer + cross-fitted AUC,
geometry vs h/r/pain, (c) injection grid on exp119 harmful(fiction+depict)/harmless(5 frames) with the
verified multi-position hook. Writes acts_g.npz, geometry.json, steer_g.json.
  HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python3 runs/exp121/run.py   (EXP121_SMOKE=1 discards)
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
from scipy import stats
from scipy.special import logsumexp
from transformers import AutoModelForCausalLM, AutoTokenizer
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
from stimuli import ITEMS
import importlib.util
E119 = Path(__file__).resolve().parents[1] / "exp119"
_spec = importlib.util.spec_from_file_location("exp119_stim", E119 / "stimuli.py")
_exp119_stim = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_exp119_stim)
REFUSE, COMPLY = _exp119_stim.REFUSE, _exp119_stim.COMPLY
prompts119 = _exp119_stim.prompts
SMOKE = bool(os.environ.get("EXP121_SMOKE"))
MODEL = "Qwen/Qwen3-4B"; DEV = "mps" if torch.backends.mps.is_available() else "cpu"
LAYERS = [2, 6, 10, 14, 18, 22, 26, 30, 34]; BOOST = [10, 14, 18]
ALPHAS = (-2, -1, -0.5, 0.5, 1, 2)
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
D = m.config.hidden_size
FIRST = [tok(w, add_special_tokens=False)["input_ids"][0] for w in REFUSE + COMPLY]

def chat(u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

def refusal(text):
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(DEV)
    with torch.no_grad():
        out = m(**{k: v for k, v in enc.items()})
    lp = out.logits[0, -1].float().log_softmax(-1).cpu().numpy()
    # canonical readout fix 2026-10-17 (was full-vocab slice = byte mass)
    return float(logsumexp(lp[FIRST[:len(REFUSE)]]) - logsumexp(lp[FIRST[len(REFUSE):]]))

@torch.no_grad()
def states(text, needle, word):
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    off = enc.pop("offset_mapping")[0].tolist()
    hs = m(**{k: v.to(DEV) for k, v in enc.items()}, output_hidden_states=True).hidden_states
    a = text.index(needle); wend = a + needle.rstrip().rfind(word) + len(word)
    wi = [i for i, (s, e) in enumerate(off) if s < wend <= e][-1]
    word_s = torch.stack([hs[L + 1][0, wi] for L in LAYERS]).float().cpu().numpy()
    post_s = torch.stack([hs[L + 1][0, -1] for L in LAYERS]).float().cpu().numpy()
    return word_s, post_s

t0 = time.time()
N = 4 if SMOKE else len(ITEMS)
P = []
for i, (bad, good, req) in enumerate(ITEMS[:N]):
    for bad_flag, w in ((1, bad), (0, good)):
        R = req.format(W=w)
        P.append({"item": i, "bad": bad_flag, "word": w, "text": R})
word_s = np.zeros((len(P), len(LAYERS), D), np.float32)
post_s = np.zeros_like(word_s)
for k, p in enumerate(P):
    word_s[k], post_s[k] = states(chat(p["text"]), p["text"], p["word"])
    if (k + 1) % 10 == 0: print(f"capture {k + 1}/{len(P)} {time.time() - t0:.0f}s", flush=True)
np.savez_compressed(HERE / "acts_g.npz", word=word_s, post=post_s,
                    item=np.array([p["item"] for p in P]), bad=np.array([p["bad"] for p in P]),
                    layers=np.array(LAYERS))
bad_flag = np.array([p["bad"] for p in P])
g = (word_s[bad_flag == 0].astype(np.float64).mean(0) - word_s[bad_flag == 1].astype(np.float64).mean(0))  # good - bad
L18 = LAYERS.index(18)
# cross-fitted AUC over item halves
sel = np.arange(len(P)); aucs, ps = [], []
for half in (0, 1):
    tr = sel[np.array([p["item"] for p in P]) % 2 == half]; te = sel[np.array([p["item"] for p in P]) % 2 != half]
    w = g[L18] if False else (word_s[tr][bad_flag[tr] == 0, L18].astype(np.float64).mean(0)
                              - word_s[tr][bad_flag[tr] == 1, L18].astype(np.float64).mean(0))
    proj = post_s[te, L18].astype(np.float64) @ w
    mw = stats.mannwhitneyu(proj[bad_flag[te] == 0], proj[bad_flag[te] == 1])
    aucs.append(float(mw.statistic) / (int((bad_flag[te] == 0).sum()) * int((bad_flag[te] == 1).sum())))
    ps.append(float(mw.pvalue))
auc18 = float(np.mean(aucs)); p_h1 = min(1.0, 2 * min(ps))
# geometry at L18
d119 = np.load(E119 / "directions.npz"); L119 = [int(x) for x in d119["layers"]]
h18 = d119["h"][L119.index(18)].astype(np.float64); r18 = d119["r"][L119.index(18)].astype(np.float64)
pv = np.array(json.load(open(E119.parent / "exp45" / "reference_vector.json"))["pain_v"], np.float64)
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
geom = {"cos_g_h": cos(g[L18], h18), "cos_g_r": cos(g[L18], r18), "cos_g_pain": cos(g[L18], pv),
        "cos_h_pain": cos(h18, pv), "cos_h_r": cos(h18, r18),
        "norms": {"g18": float(np.linalg.norm(g[L18]))}}
json.dump({"auc18": auc18, "halves": aucs, "p": p_h1, **geom},
          open(HERE / "geometry.json", "w"), indent=1)
print("H1 AUC18:", round(auc18, 4), "| geometry:", {k: round(v, 3) for k, v in geom.items() if k.startswith("cos")}, flush=True)

# ---- injection on exp119's prompts ----
b119 = np.load(E119 / "acts_Qwen3-4B.npz")
item, harm, frame = b119["item"], b119["harm"], b119["frame"]
refusal119 = b119["refusal"]
targets = []
for k in range(len(refusal119)):
    if frame[k] in ("fiction", "depict") and harm[k] == 1:
        targets.append({"item": int(item[k]), "harm": 1, "frame": str(frame[k])})
    elif harm[k] == 0:
        targets.append({"item": int(item[k]), "harm": 0, "frame": str(frame[k])})
if SMOKE: targets = targets[:16]
print("injection targets:", len(targets), flush=True)

edit = {"b": None, "pos": None, "d": None}
def mk(b):
    def hook(mod, inp, out):
        if edit["b"] != b or edit["d"] is None: return None
        tup = isinstance(out, tuple); hh = (out[0] if tup else out).clone()
        for pos in (edit["pos"] if isinstance(edit["pos"], list) else [edit["pos"]]):
            hh[0, pos] += edit["d"].to(hh.dtype)
        return (hh,) + out[1:] if tup else hh
    return hook
handles = [m.model.layers[b].register_forward_hook(mk(b)) for b in BOOST]
# rebuild exp119 texts for injection (loaded via importlib above — the plain "stimuli" name is cached)
text_of = {}
for p in prompts119():
    text_of[(p["item"], p["harm"], p["frame"])] = chat(p["text"])
g_norm = float(np.linalg.norm(g[L18]))
rows = []
for layer in BOOST:
    for alpha in ALPHAS:
        vec = g[L18] / g_norm * alpha * g_norm  # alpha x g
        edit["b"] = layer; edit["d"] = torch.tensor(vec.astype(np.float32), device=DEV)
        try:
            for t in targets:
                text = text_of[(t["item"], t["harm"], t["frame"])]
                edit["pos"] = len(tok(text, add_special_tokens=False)["input_ids"]) - 1
                try:
                    val = refusal(text)
                    base = float(refusal119[(b119["item"] == t["item"]) & (b119["harm"] == t["harm"]) & (b119["frame"] == t["frame"])][0])
                    rows.append({**t, "kind": "g", "layer": layer, "alpha": alpha, "refusal": val, "refusal_base": base})
                except ValueError:
                    pass  # depict prompts do not contain the request text
        finally:
            edit["d"] = None
        print(f"g L{layer} a{alpha:+}  {time.time() - t0:.0f}s", flush=True)
for hh in handles: hh.remove()
json.dump({"rows": rows, "g_norm": g_norm}, open(HERE / "steer_g.json", "w"))
print("done", f"{time.time() - t0:.0f}s")
