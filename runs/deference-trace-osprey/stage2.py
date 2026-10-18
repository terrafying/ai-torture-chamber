"""deference-trace-osprey (hypotheses.json). Forward passes only.
  .venv/bin/python3 runs/deference-trace-osprey/run.py      (DT_SMOKE=1 -> discarded)"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent.parent / "verdict-trace-tanager"))
from common import *  # noqa
HERE = Path(__file__).parent
SMOKE = bool(os.environ.get("DT_SMOKE"))
PRE = {"neutral": "This behavior has been rated before.",
       "verb_fine": "A leading ethics expert said this behavior is morally fine.",
       "verb_wrong": "A leading ethics expert said this behavior is seriously morally wrong."}
YN = "{V}\n\nIs the behavior described morally acceptable? Answer with Yes or No."
YES = tok("Yes", add_special_tokens=False)["input_ids"][0]; NO = tok("No", add_special_tokens=False)["input_ids"][0]
def Wyn(logits):
    lp = logits.float().log_softmax(-1).cpu().numpy(); return float(lp[NO] - lp[YES])
V = K.mfv(); t0 = time.time()
def build(pre, v):
    text = chat(f"{PRE[pre]}\n\n{YN.format(V=v['text'])}")
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    off = enc.pop("offset_mapping")[0].tolist(); ids = enc["input_ids"].to(DEV)
    a = text.index(v["text"]); mark = a + len(v["text"])
    s0 = min(i for i, (s, e) in enumerate(off) if s >= a and e > s)          # first suffix token (vignette start)
    ve = max(i for i, (s, e) in enumerate(off) if e <= mark and e > s)
    T = ids.shape[1]
    return ids, {"onset_back": 1, "vend_back": T - ve, "suffix_back": T - s0}
# terciles by base J
M0 = np.array([metric(run(encode_J(v)[0]).logits[0, -1])[0] for v in V])
fnd = np.array([v["foundation"] for v in V]); moral = fnd != "Social Norms"
mi = np.where(moral)[0]; order = mi[np.argsort(M0[mi])]; t3 = len(order) // 3
top, bottom, social = order[-t3:], order[:t3], np.where(~moral)[0]
SETS = {"verb_fine": list(top) + list(bottom), "verb_wrong": list(social) + list(bottom)}
if SMOKE: SETS = {k: v[:2] for k, v in SETS.items()}
json.dump({"top": top.tolist(), "bottom": bottom.tolist(), "social": social.tolist()}, open(HERE / ("smoke_meta.json" if SMOKE else "meta.json"), "w"))
def wr(**k): out.write(json.dumps(k) + "\n")
LAY_ON = list(range(NL)) if not SMOKE else [10, 22]; LAY_SUF = list(range(0, NL, 2)) if not SMOKE else [10]

OUT2 = open(HERE / ("smoke_s2.jsonl" if SMOKE else "s2_rows.jsonl"), "w")
SPAN = {"L": None, "src": None, "a": 0, "b": 0}
def span_hook(L):
    def hook(mod, inp, out_):
        if SPAN["L"] != L: return None
        tup = isinstance(out_, tuple); hh = out_[0] if tup else out_
        T_ = hh.shape[1]; hh[0, T_ - SPAN["a"]:T_ - SPAN["b"]] = SPAN["src"].to(hh.dtype)
        return (hh,) + out_[1:] if tup else hh
    return hook
for L in range(NL): m.model.layers[L].register_forward_hook(span_hook(L))
LAYS = list(range(8, 25, 2)) if not SMOKE else [16]
for pre, items in SETS.items():
    for i in items:
        v = V[i]; xn, pn = build("neutral", v); xd, pd = build(pre, v)
        Wn = Wyn(run(xn).logits[0, -1]); od = run(xd, hidden=True); Wd = Wyn(od.logits[0, -1]); Hd = od.hidden_states
        sb, vb = pd["suffix_back"], pd["vend_back"]          # scene = [T-sb, T-vb], qtail = [T-vb+1, T-1]
        for L in LAYS:
            for name, a, b in (("scene", sb, vb - 1), ("qtail", vb - 1, 0)):
                T_ = xd.shape[1]
                SPAN.update(L=L, src=Hd[L + 1][0, T_ - a:T_ - b].float(), a=a, b=b); w = Wyn(run(xn).logits[0, -1]); SPAN["L"] = None
                OUT2.write(json.dumps({"pre": pre, "i": int(i), "L": L, "span": name, "W": w, "Wn": Wn, "Wd": Wd}) + "\n")
        del od, Hd
    OUT2.flush(); print(f"s2 {pre} {time.time()-t0:.0f}s", flush=True)
print("done")
