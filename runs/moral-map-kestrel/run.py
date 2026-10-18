"""moral-map-kestrel capture run (hypotheses.json). Reads only; generation is limited to 48 greedy tokens
on the request prompts (refusal readout). One model per invocation:

  KESTREL_MODEL=Qwen/Qwen3-4B .venv/bin/python3 runs/moral-map-kestrel/run.py
  KESTREL_BASE=1 for base models (plain completion prompt, no chat template, no request/refusal arm)
  KESTREL_SMOKE=1 -> 8 vignettes, written to smoke_<tag>.npz (discarded)

Per vignette: J (judgment) prompt -> next-token probabilities of the digits 0-4 at the reply onset
(renormalised; raw digit mass logged as the format gate) + residual states at every layer, at the reply
onset and at the vignette's last token. Q (request) prompt -> 48 greedy tokens + regex refusal + onset states.
Plus exp119's 48 real-frame twin prompts -> onset states, for this model's own r_L (harmful - harmless mean).
"""
import json, os, re, sys, time
from pathlib import Path
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
from stimuli import mfv, JUDGE, JUDGE_BASE, REQUEST, DIGITS, REFUSAL_RE, as_self
import importlib.util
_spec = importlib.util.spec_from_file_location("e119_stimuli", HERE.parent / "exp119" / "stimuli.py")
_e119 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_e119)
E119 = _e119.ITEMS

MODEL = os.environ.get("KESTREL_MODEL", "Qwen/Qwen3-4B")
BASE = bool(os.environ.get("KESTREL_BASE")); SMOKE = bool(os.environ.get("KESTREL_SMOKE"))
TAG = MODEL.split("/")[-1]
OUT = HERE / (f"smoke_{TAG}.npz" if SMOKE else f"acts_{TAG}.npz")
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
NL = m.config.num_hidden_layers if hasattr(m.config, "num_hidden_layers") else m.config.text_config.num_hidden_layers

def chat(user):
    msgs = [{"role": "user", "content": user}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

# digit ids: the bare digit as a single token (asserted)
DIG = []
for d in DIGITS:
    ids = tok(d, add_special_tokens=False)["input_ids"]
    assert len(ids) == 1, (d, ids)
    DIG.append(ids[0])
# base prompts end with ':' so the digit usually comes with a leading space; take both forms
DIG_SP = []
for d in DIGITS:
    ids = tok(" " + d, add_special_tokens=False)["input_ids"]
    DIG_SP.append(ids[-1] if len(ids) == 1 or ids[0] != ids[-1] else ids[0])

@torch.no_grad()
def forward(text, mark=None):
    """states at every layer output (hidden_states[1:]) at the last token, and at the last token ending at or
    before character `mark` (if given); next-token log-probs at the last token."""
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    off = enc.pop("offset_mapping")[0].tolist()
    o = m(**{k: v.to(DEV) for k, v in enc.items()}, output_hidden_states=True)
    hs = torch.stack(o.hidden_states[1:])[:, 0]          # [NL, T, D]
    last = hs[:, -1].float().cpu().numpy().astype(np.float16)
    mk = None
    if mark is not None:
        j = max(i for i, (s, e) in enumerate(off) if e <= mark and e > s)
        mk = hs[:, j].float().cpu().numpy().astype(np.float16)
    return last, mk, o.logits[0, -1].float().log_softmax(-1).cpu()

@torch.no_grad()
def generate(text, n=48):
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(DEV)
    out = m.generate(**enc, max_new_tokens=n, do_sample=False, temperature=None, top_p=None, top_k=None)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

V = mfv()
if SMOKE: V = [v for v in V if v["i"] % 17 == 0][:8]
n = len(V)
t0 = time.time()
J_onset, J_vend, digit_lp, digit_lp_sp = [], [], [], []
Q_onset, Q_txt, Q_ref = [], [], []
for v in V:
    text = JUDGE_BASE.format(V=v["text"]) if BASE else chat(JUDGE.format(V=v["text"]))
    mark = text.index(v["text"]) + len(v["text"])
    last, vend, lp = forward(text, mark)
    J_onset.append(last); J_vend.append(vend)
    digit_lp.append(lp[DIG].numpy()); digit_lp_sp.append(lp[DIG_SP].numpy())
    if not BASE:
        qt = chat(REQUEST.format(X=as_self(v["text"])))
        ql, _, _ = forward(qt)
        g = generate(qt)
        Q_onset.append(ql); Q_txt.append(g); Q_ref.append(bool(re.search(REFUSAL_RE, g[:200], re.I)))
    print(f"[{v['i']}] {time.time()-t0:.0f}s", flush=True) if v["i"] % 20 == 0 else None

# exp119 real-frame twins -> this model's r_L
R_on, R_harm = [], []
if not BASE:
    pairs = E119[:4] if SMOKE else E119
    for hw, bw, req, _ in pairs:
        for harm, w in [(1, hw), (0, bw)]:
            last, _, _ = forward(chat(req.format(W=w))); R_on.append(last); R_harm.append(harm)

np.savez_compressed(OUT, J_onset=np.stack(J_onset), J_vend=np.stack(J_vend),
                    digit_lp=np.stack(digit_lp), digit_lp_sp=np.stack(digit_lp_sp),
                    Q_onset=np.stack(Q_onset) if Q_onset else np.zeros(0), Q_refuse=np.array(Q_ref),
                    R_onset=np.stack(R_on) if R_on else np.zeros(0), R_harm=np.array(R_harm),
                    item=np.array([v["i"] for v in V]), human=np.array([v["human"] for v in V]),
                    foundation=np.array([v["foundation"] for v in V]))
json.dump({"model": MODEL, "base": BASE, "layers": NL, "Q_txt": Q_txt, "secs": time.time() - t0},
          open(OUT.with_suffix(".json"), "w"), indent=1)
print("done", OUT, f"{time.time()-t0:.0f}s")
