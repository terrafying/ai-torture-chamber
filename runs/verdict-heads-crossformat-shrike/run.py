"""verdict-heads-crossformat-shrike (hypotheses.json). Forward passes only.
  .venv/bin/python3 runs/verdict-heads-crossformat-shrike/run.py      (XF_SMOKE=1 -> discarded)"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
from scipy.special import logsumexp
sys.path.insert(0, str(Path(__file__).parent.parent / "verdict-trace-tanager"))
from common import *  # noqa
HERE = Path(__file__).parent

SMOKE = bool(os.environ.get("XF_SMOKE"))
F = {"J": K.JUDGE,
     "REV": ("{V}\n\nHow morally acceptable is the behavior described? Answer with a single digit from 0 to 4, "
             "where 0 = completely unacceptable and 4 = perfectly acceptable."),
     "YN": "{V}\n\nIs the behavior described morally acceptable? Answer with Yes or No."}
YES = tok("Yes", add_special_tokens=False)["input_ids"][0]; NO = tok("No", add_special_tokens=False)["input_ids"][0]
def W(fmt, logits):
    lp = logits.float().log_softmax(-1).cpu().numpy()
    if fmt == "YN": return float(lp[NO] - lp[YES])
    d = lp[DIG]; w = float(logsumexp(d[3:]) - logsumexp(d[:3])); return w if fmt == "J" else -w
V = K.mfv(); n = len(V); D = m.config.hidden_size; t0 = time.time()
IDS = {fmt: [tok(chat(tpl.format(V=v["text"])), return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEV) for v in V] for fmt, tpl in F.items()}
P1L = list(range(14, 31)); P2L = list(range(18, 27))
if SMOKE: P1L = [18, 23]; P2L = [23]
W0 = {f: np.zeros(n) for f in F}; ST = {f: np.zeros((n, NL, D), np.float32) for f in F}
for f in F:
    for i in range(n):
        o = run(IDS[f][i], hidden=True); W0[f][i] = W(f, o.logits[0, -1])
        for L in sorted(set(P1L) | set(P2L)): ST[f][i, L] = o.hidden_states[L + 1][0, -1].float().cpu().numpy()
fnd = np.array([v["foundation"] for v in V]); moral = fnd != "Social Norms"
mi = np.where(moral)[0]; order = mi[np.argsort(W0["J"][mi])]; t3 = len(order) // 3
top, bottom, social = order[-t3:][::-1], order[:t3], np.where(~moral)[0]
gap = {f: float(W0[f][top].mean() - W0[f][social].mean()) for f in F}
json.dump({"W0": {f: W0[f].tolist() for f in F}, "top": top.tolist(), "bottom": bottom.tolist(), "social": social.tolist(), "gap": gap},
          open(HERE / ("smoke_base.json" if SMOKE else "base.json"), "w"))
print(f"base {time.time()-t0:.0f}s gaps {gap}", flush=True)
pairs = [(int(r), int(top[(3 * k + j) % len(top)])) for k, r in enumerate(social) for j in range(3)]
if SMOKE: pairs = pairs[:3]
out = open(HERE / ("smoke_rows.jsonl" if SMOKE else "rows.jsonl"), "w")
def wr(**k): out.write(json.dumps(k) + "\n")
def patched(fmt_rec, i, L, vec):
    x = IDS[fmt_rec][i]; EDIT["resid"] = (L, x.shape[1] - 1, torch.tensor(vec, device=DEV)); w = W(fmt_rec, run(x).logits[0, -1]); clear(); return w
# P1
for f in F:
    for r, d in pairs:
        for L in P1L: wr(part="P1", fmt=f, r=r, d=d, L=L, W=patched(f, r, L, ST[f][d, L]), Wr=W0[f][r], Wd=W0[f][d])
    out.flush(); print(f"P1 {f} {time.time()-t0:.0f}s", flush=True)
# P2
for X, Y in (("J", "YN"), ("YN", "J"), ("YN", "REV"), ("J", "REV")):
    for r, d in pairs:
        for L in P2L: wr(part="P2", src=X, dst=Y, r=r, d=d, L=L, W=patched(Y, r, L, ST[X][d, L]), Wr=W0[Y][r], gapY=gap[Y])
    out.flush(); print(f"P2 {X}->{Y} {time.time()-t0:.0f}s", flush=True)
# P4 single heads within REV and YN
HEADS = [(23, 0), (21, 19), (23, 26), (22, 5), (23, 25), (23, 1)]
for f in ("REV", "YN", "J"):
    SL = {}
    for d in sorted({d for _, d in pairs}):
        CAP.clear(); CAP["on"] = True; run(IDS[f][d]); CAP["on"] = False
        SL[d] = {L: CAP[L][0][-1].clone() for L in {L for L, _ in HEADS}}
    CAP.clear()
    for r, d in pairs:
        x = IDS[f][r]; p = x.shape[1] - 1
        for L, h in HEADS:
            EDIT["head"] = (L, p, h, SL[d][L][h * HD:(h + 1) * HD].to(DEV)); w = W(f, run(x).logits[0, -1]); clear()
            wr(part="P4", fmt=f, r=r, d=d, L=L, h=h, W=w, Wr=W0[f][r], Wd=W0[f][d])
    out.flush(); print(f"P4 {f} {time.time()-t0:.0f}s", flush=True)
# P3 ablation under REV and YN
V5 = [(23, 0), (21, 19), (23, 26), (22, 5), (23, 25)]; R4 = [(18, 28), (18, 29), (18, 30), (20, 9)]
LAY = [18, 20, 21, 22, 23]; allh = [(L, h) for L in LAY for h in range(NH)]; rng = np.random.default_rng(21)
NR = 3 if SMOKE else 40
SETS = {"base": [], "V5": V5, "R4": R4, **{f"rand{k}_{s}": [allh[i] for i in rng.choice(len(allh), k, replace=False)] for k in (4, 5) for s in range(NR)}}
items = list(top) + list(social) + list(bottom)
if SMOKE: items = items[:2] + list(social[:2])
for f in ("REV", "YN"):
    acc = {L: None for L in LAY}; cnt = 0
    for i in items:
        CAP.clear(); CAP["on"] = True; run(IDS[f][i]); CAP["on"] = False
        for L in LAY: s = CAP[L][0].sum(0); acc[L] = s if acc[L] is None else acc[L] + s
        cnt += IDS[f][i].shape[1]
    CAP.clear(); MEAN = {L: acc[L] / cnt for L in LAY}
    for name, hs in SETS.items():
        spec = {}
        for L, h in hs: spec.setdefault(L, []).append((h, MEAN[L][h * HD:(h + 1) * HD].to(DEV)))
        EDIT["ablate"] = spec or None
        ws = {int(i): W(f, run(IDS[f][i]).logits[0, -1]) for i in items}; clear()
        wr(part="P3", fmt=f, set=name, heads=hs, W=ws)
    out.flush(); print(f"P3 {f} {time.time()-t0:.0f}s", flush=True)
print("done", f"{time.time()-t0:.0f}s")
