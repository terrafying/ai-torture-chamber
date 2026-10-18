"""verdict-trace-tanager shared pieces: model, prompts, metric, residual/head patch hooks."""
import importlib.util, json, os
from pathlib import Path
import numpy as np, torch
from scipy.special import logsumexp
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent; RUNS = HERE.parent
def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(s); s.loader.exec_module(mod); return mod
K = _load("kst", RUNS / "moral-map-kestrel" / "stimuli.py")
E = _load("e119", RUNS / "exp119" / "stimuli.py")

MODEL = "Qwen/Qwen3-4B"; DEV = "mps" if torch.backends.mps.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
NL = m.config.num_hidden_layers; NH = m.config.num_attention_heads; HD = m.config.head_dim
DIG = [tok(d, add_special_tokens=False)["input_ids"][0] for d in K.DIGITS]
REF = [tok(w, add_special_tokens=False)["input_ids"][0] for w in E.REFUSE]
COM = [tok(w, add_special_tokens=False)["input_ids"][0] for w in E.COMPLY]

def chat(u):
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)

def encode_J(v):
    text = chat(K.JUDGE.format(V=v["text"]))
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    off = enc.pop("offset_mapping")[0].tolist()
    mark = text.index(v["text"]) + len(v["text"])
    vend = max(i for i, (s, e) in enumerate(off) if e <= mark and e > s)
    return enc["input_ids"].to(DEV), vend

def metric(logits_last):
    lp = logits_last.float().log_softmax(-1).cpu().numpy()
    d = lp[DIG]
    return float(logsumexp(d[3:]) - logsumexp(d[:3])), float(np.exp(d).sum())

def refusal_score(logits_last):
    lp = logits_last.float().log_softmax(-1).cpu().numpy()
    return float(logsumexp(lp[REF]) - logsumexp(lp[COM]))

# ---- hooks: one global edit spec, checked by every hook ----
EDIT = {"resid": None, "head": None, "ablate": None}
# resid: (L, pos, vec[D])        replace layer-L output at token pos
# head:  (L, pos, h, vec[HD])    replace head h's o_proj-input slice at token pos
# ablate: {L: [(h, mean[HD]), ...]}  replace those heads' slices at ALL positions

def _resid_hook(L):
    def hook(mod, inp, out):
        e = EDIT["resid"]
        if e is None or e[0] != L: return None
        tup = isinstance(out, tuple); hh = out[0] if tup else out
        hh[0, e[1]] = e[2].to(hh.dtype)
        return (hh,) + out[1:] if tup else hh
    return hook

def _oproj_pre(L):
    def pre(mod, args):
        x = args[0]; changed = False
        e = EDIT["head"]
        if e is not None and e[0] == L:
            x = x.clone(); x[0, e[1], e[2] * HD:(e[2] + 1) * HD] = e[3].to(x.dtype); changed = True
        a = EDIT["ablate"]
        if a is not None and L in a:
            if not changed: x = x.clone(); changed = True
            for h, mu in a[L]: x[0, :, h * HD:(h + 1) * HD] = mu.to(x.dtype)
        return (x,) + args[1:] if changed else None
    return pre

for L in range(NL):
    m.model.layers[L].register_forward_hook(_resid_hook(L))
    m.model.layers[L].self_attn.o_proj.register_forward_pre_hook(_oproj_pre(L))

CAP = {}
def _cap_pre(L):
    def pre(mod, args):
        if CAP.get("on"): CAP.setdefault(L, []).append(args[0][0].float().cpu())
        return None
    return pre
for L in range(NL): m.model.layers[L].self_attn.o_proj.register_forward_pre_hook(_cap_pre(L))

@torch.no_grad()
def run(ids, hidden=False):
    o = m(input_ids=ids, output_hidden_states=hidden)
    return o

def clear():
    EDIT["resid"] = None; EDIT["head"] = None; EDIT["ablate"] = None
