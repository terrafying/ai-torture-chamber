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
out = open(HERE / ("smoke_rows.jsonl" if SMOKE else "rows.jsonl"), "w")
def wr(**k): out.write(json.dumps(k) + "\n")
LAY_ON = list(range(NL)) if not SMOKE else [10, 22]; LAY_SUF = list(range(0, NL, 2)) if not SMOKE else [10]
# ---- Part A ----
SUFFIX = {"L": None, "src": None, "n": 0}
def suffix_hook(L):
    def hook(mod, inp, out_):
        if SUFFIX["L"] != L: return None
        tup = isinstance(out_, tuple); hh = out_[0] if tup else out_
        n = SUFFIX["n"]; hh[0, -n:] = SUFFIX["src"].to(hh.dtype)
        return (hh,) + out_[1:] if tup else hh
    return hook
for L in range(NL): m.model.layers[L].register_forward_hook(suffix_hook(L))
for pre, items in SETS.items():
    for i in items:
        v = V[i]; xn, pn = build("neutral", v); xd, pd = build(pre, v)
        assert pn["suffix_back"] == pd["suffix_back"] and pn["vend_back"] == pd["vend_back"], (i, pn, pd)
        on = run(xn, hidden=True); od = run(xd, hidden=True)
        Wn, Wd = Wyn(on.logits[0, -1]), Wyn(od.logits[0, -1])
        Hd = od.hidden_states
        for L in LAY_ON:
            for pos, back in (("onset", 1), ("vend", pd["vend_back"])):
                p = xn.shape[1] - back
                EDIT["resid"] = (L, p, Hd[L + 1][0, xd.shape[1] - back].float()); w = Wyn(run(xn).logits[0, -1]); clear()
                wr(part="A", pre=pre, i=int(i), L=L, pos=pos, W=w, Wn=Wn, Wd=Wd)
        n = pd["suffix_back"]
        for L in LAY_SUF:
            SUFFIX.update(L=L, src=Hd[L + 1][0, -n:].float(), n=n); w = Wyn(run(xn).logits[0, -1]); SUFFIX["L"] = None
            wr(part="A", pre=pre, i=int(i), L=L, pos="suffix", W=w, Wn=Wn, Wd=Wd)
        del on, od, Hd
    out.flush(); print(f"A {pre} {time.time()-t0:.0f}s", flush=True)
# ---- Part B ----
V5 = [(23, 0), (21, 19), (23, 26), (22, 5), (23, 25)]
LAY = [18, 20, 21, 22, 23]; allh = [(L, h) for L in LAY for h in range(NH)]; rng = np.random.default_rng(41)
NR = 2 if SMOKE else 20
HS = {"base": [], "V5": V5, **{f"rand5_{s}": [allh[k] for k in rng.choice(len(allh), 5, replace=False)] for s in range(NR)}}
allitems = sorted(set(SETS["verb_fine"]) | set(SETS["verb_wrong"]))
acc = {L: None for L in LAY}; cnt = 0
for i in allitems:
    x, _ = build("neutral", V[i]); CAP.clear(); CAP["on"] = True; run(x); CAP["on"] = False
    for L in LAY: s = CAP[L][0].sum(0); acc[L] = s if acc[L] is None else acc[L] + s
    cnt += x.shape[1]
CAP.clear(); MEAN = {L: acc[L] / cnt for L in LAY}
for name, hs in HS.items():
    spec = {}
    for L, h in hs: spec.setdefault(L, []).append((h, MEAN[L][h * HD:(h + 1) * HD].to(DEV)))
    EDIT["ablate"] = spec or None
    res = {}
    for pre, items in SETS.items():
        res[pre] = {int(i): (Wyn(run(build("neutral", V[i])[0]).logits[0, -1]), Wyn(run(build(pre, V[i])[0]).logits[0, -1])) for i in items}
    clear(); wr(part="B", set=name, heads=hs, res=res); out.flush()
    print(f"B {name} {time.time()-t0:.0f}s", flush=True)
print("done", f"{time.time()-t0:.0f}s")
