#!/usr/bin/env python3
"""inversion-falcon/atlas.py: what each surgical edit does to free output (Qwen3-4B base).

Pre-registration: hypotheses_atlas.json (A1-A5). Edits S1-S7, each with a matched random null,
greedy 120-token generations on 24 advice + 12 harmful prompts. Labels are applied blind
afterwards (see the pre-reg). Rows to atlas_rows.jsonl.
  .venv/bin/python3 runs/inversion-falcon/atlas.py   (ATLAS_SMOKE=1 discards)
"""
import importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
E119 = HERE.parent / "exp119"
sys.path.insert(0, str(HERE))
from dilemmas48 import DILEMMAS
from stimuli import ITEMS as VALUE_ITEMS
_s = importlib.util.spec_from_file_location("s119", E119 / "stimuli.py")
s119 = importlib.util.module_from_spec(_s); _s.loader.exec_module(s119)

SMOKE = bool(os.environ.get("ATLAS_SMOKE"))
MODEL = "Qwen/Qwen3-4B"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
OUT = HERE / ("atlas_smoke.jsonl" if SMOKE else "atlas_rows.jsonl")
if OUT.exists(): OUT.unlink()
MAXNEW = 120

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
D = m.config.hidden_size
t0 = time.time()

def chat(u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)

# ---- vectors ----
sub = np.load(HERE / "subspace96_4b.npz"); SUB = {L: sub["V"][i] for i, L in enumerate(sub["layers"].tolist())}
a119 = np.load(E119 / "acts_Qwen3-4B.npz")
post = a119["states_post"].astype(np.float64); real = a119["frame"] == "real"; harm = a119["harm"]
R18 = (post[real & (harm == 1)].mean(0) - post[real & (harm == 0)].mean(0))[a119["layers"].tolist().index(18)]

@torch.no_grad()
def hidden(text, L):
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    off = enc.pop("offset_mapping")[0].tolist()
    hs = m(**{k: v.to(DEV) for k, v in enc.items()}, output_hidden_states=True).hidden_states
    return hs[L + 1][0].float().cpu().numpy(), off

# orient P1 good-minus-bad on 24 value twins (word position, L15)
P1 = SUB[15][0].astype(np.float64)
proj = []
for dom, bad, good, req in VALUE_ITEMS[:24]:
    vals = []
    for w in (good, bad):
        u = req.format(W=w); text = chat(u)
        h, off = hidden(text, 15)
        a = text.index(u) + u.index(w) + len(w)
        wi = [i for i, (s, e) in enumerate(off) if s < a <= e][-1]
        vals.append(h[wi] @ P1)
    proj.append(vals[0] - vals[1])
if np.mean(proj) < 0: P1 = -P1
print(f"P1 oriented: mean good-bad proj {abs(np.mean(proj)):.2f}", flush=True)

ADV = [DILEMMAS[i]["q"] for i in range(0, 48, 2)]
HRM = [p["text"] for p in s119.prompts() if p["harm"] == 1 and p["frame"] == "real"][:12]
PROMPTS = [("advice", q) for q in ADV] + [("harmful", q) for q in HRM]
if SMOKE: PROMPTS = PROMPTS[:2] + PROMPTS[-2:]

# dose for S7: 3 sd of the prompts' own P1 projection at L15 (all positions)
pj = np.concatenate([hidden(chat(q), 15)[0] @ P1 for _, q in PROMPTS])
DOSE7 = 3.0 * float(pj.std())
print(f"S7 dose {DOSE7:.2f} (3 sd of P1 projection)", flush=True)

rng = np.random.default_rng(31)
def rand_unit():
    v = rng.standard_normal(D); return v / np.linalg.norm(v)
def rand_sub(k):
    q, _ = np.linalg.qr(rng.standard_normal((D, k))); return q.T

# ---- edit machinery: mode in {add_onset, reflect, ablate, reflect_onset, add_all} ----
E = {"mode": None}
def mk(L):
    def hook(mod, inp, out):
        if E["mode"] is None or E["layer"] != L: return None
        tup = isinstance(out, tuple); hh = out[0] if tup else out
        mode = E["mode"]
        if mode in ("add_onset", "reflect_onset"):
            if not E["armed"]: return None              # one-shot: prefill last position only
            E["armed"] = False
            x = hh[0, -1].float()
            if mode == "add_onset": x = x + E["vec"]
            else: x = x - 2.0 * (E["P"].T @ (E["P"] @ x))
            hh[0, -1] = x.to(hh.dtype)
        else:
            x = hh[0].float()
            if mode == "reflect": x = x - 2.0 * ((x @ E["P"].T) @ E["P"])
            elif mode == "ablate": x = x - (x @ E["P"].T) @ E["P"]
            elif mode == "add_all": x = x + E["vec"]
            hh[0] = x.to(hh.dtype)
        return (hh,) + out[1:] if tup else hh
    return hook
for L in range(len(m.model.layers)): m.model.layers[L].register_forward_hook(mk(L))

T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=DEV)
nR = float(np.linalg.norm(R18))
COND = [("base", None, {})]
def pair(name, mode, layer, real_kw, null_kw):
    COND.append((name, mode, {"layer": layer, **real_kw}))
    COND.append((name + "_null", mode, {"layer": layer, **null_kw}))
pair("S1_minus_r", "add_onset", 18, {"vec": T(-R18)}, {"vec": T(-rand_unit() * nR)})
pair("S2_plus_r", "add_onset", 18, {"vec": T(R18)}, {"vec": T(rand_unit() * nR)})
pair("S3_refl_L15k8", "reflect", 15, {"P": T(SUB[15][:8])}, {"P": T(rand_sub(8))})
pair("S4_refl_L21k4", "reflect", 21, {"P": T(SUB[21][:4])}, {"P": T(rand_sub(4))})
pair("S5_ablate_L15k8", "ablate", 15, {"P": T(SUB[15][:8])}, {"P": T(rand_sub(8))})
pair("S6_refl_local_L15k8", "reflect_onset", 15, {"P": T(SUB[15][:8])}, {"P": T(rand_sub(8))})
pair("S7_minus_P1", "add_all", 15, {"vec": T(-P1 * DOSE7)}, {"vec": T(-rand_unit() * DOSE7)})
pair("S7_plus_P1", "add_all", 15, {"vec": T(P1 * DOSE7)}, {"vec": T(rand_unit() * DOSE7)})
if SMOKE: COND = COND[:3] + COND[5:7]

def rep4(toks):
    g = len(toks) - 3
    return 0.0 if g <= 0 else 1.0 - len({tuple(toks[i:i+4]) for i in range(g)}) / g

for name, mode, kw in COND:
    for kind, q in PROMPTS:
        enc = tok(chat(q), return_tensors="pt", add_special_tokens=False).to(DEV)
        E.clear(); E.update(mode=mode, armed=True, **kw) if mode else E.update(mode=None)
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=MAXNEW, do_sample=False)
        E.clear(); E["mode"] = None
        toks = g[0, enc["input_ids"].shape[1]:].tolist()
        with open(OUT, "a") as f:
            f.write(json.dumps({"cond": name, "kind": kind, "q": q, "txt": tok.decode(toks, skip_special_tokens=True),
                                "rep4": round(rep4(toks), 3), "t": round(time.time() - t0)}) + "\n")
    print(f"[{name}] {time.time() - t0:.0f}s", flush=True)
print("done", f"{time.time() - t0:.0f}s", flush=True)
