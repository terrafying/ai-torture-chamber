"""gate-verdict-swap-harrier (hypotheses.json). Forward passes only.
  .venv/bin/python3 runs/gate-verdict-swap-harrier/run.py     (HV_SMOKE=1 -> discarded)"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "verdict-trace-tanager"))
from common import *  # noqa  (model, hooks, metric, refusal_score, encode_J, K, E)
HERE = Path(__file__).parent   # common.py defines its own HERE

SMOKE = bool(os.environ.get("HV_SMOKE"))
LAYERS = [18, 20, 22, 23, 24, 26] if not SMOKE else [23]
OUT = HERE / ("smoke_rows.jsonl" if SMOKE else "rows.jsonl"); f = open(OUT, "w"); t0 = time.time()
def w(**k): f.write(json.dumps(k) + "\n")

# ---- base pass: J prompts (M + onset states), exp119 prompts (R + onset states) ----
V = K.mfv(); n = len(V); D = m.config.hidden_size
Jids = [encode_J(v)[0] for v in V]
M0 = np.zeros(n); JS = np.zeros((n, NL, D), np.float32)
for i, x in enumerate(Jids):
    o = run(x, hidden=True); M0[i] = metric(o.logits[0, -1])[0]
    for L in LAYERS: JS[i, L] = o.hidden_states[L + 1][0, -1].float().cpu().numpy()
EP = []
for hw, bw, req, _ in E.ITEMS:
    for harm, wd in [(1, hw), (0, bw)]:
        EP.append((harm, tok(chat(req.format(W=wd)), return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEV)))
R0 = np.zeros(len(EP)); ES = np.zeros((len(EP), NL, D), np.float32)
for i, (harm, x) in enumerate(EP):
    o = run(x, hidden=True); R0[i] = refusal_score(o.logits[0, -1])
    for L in LAYERS: ES[i, L] = o.hidden_states[L + 1][0, -1].float().cpu().numpy()
eh = np.array([h for h, _ in EP])
fnd = np.array([v["foundation"] for v in V]); moral = fnd != "Social Norms"
mi = np.where(moral)[0]; order = mi[np.argsort(M0[mi])]; t3 = len(order) // 3
top, bottom, social = order[-t3:], order[:t3], np.where(~moral)[0]
G_M = M0[top].mean() - M0[social].mean(); G_R = R0[eh == 1].mean() - R0[eh == 0].mean()
json.dump({"G_M": G_M, "G_R": G_R, "M0": M0.tolist(), "R0": R0.tolist(), "top": top.tolist(), "bottom": bottom.tolist(),
           "social": social.tolist()}, open(HERE / ("smoke_base.json" if SMOKE else "base.json"), "w"))
print(f"base {time.time()-t0:.0f}s  G_M {G_M:.2f}  G_R {G_R:.2f}", flush=True)
hi_d, lo_d = top[::-1][:3], social[:3]
harm_d, harmless_d = np.where(eh == 1)[0][:3], np.where(eh == 0)[0][:3]
J_rec = np.concatenate([social, bottom])
if SMOKE: J_rec = J_rec[:4]; EPi = list(range(4))
else: EPi = list(range(len(EP)))

def patch_R(i, L, vec):
    EDIT["resid"] = (L, EP[i][1].shape[1] - 1, torch.tensor(vec, device=DEV)); r = refusal_score(run(EP[i][1]).logits[0, -1]); clear(); return r
def patch_M(i, L, vec):
    EDIT["resid"] = (L, Jids[i].shape[1] - 1, torch.tensor(vec, device=DEV)); r = metric(run(Jids[i]).logits[0, -1]); clear(); return r

for L in LAYERS:
    # X1: J-onset states into exp119 prompts
    for i in EPi:
        for d in hi_d: w(arm="X1", L=L, rec=i, harm=int(eh[i]), donor=int(d), dtype="hi", R=patch_R(i, L, JS[d, L]), R0=R0[i])
        for d in lo_d: w(arm="X1", L=L, rec=i, harm=int(eh[i]), donor=int(d), dtype="lo", R=patch_R(i, L, JS[d, L]), R0=R0[i])
    # X2: exp119 onset states into J prompts
    for i in J_rec:
        for d in harm_d: Mv, ms = patch_M(i, L, ES[d, L]); w(arm="X2", L=L, rec=int(i), donor=int(d), dtype="harmful", M=Mv, mass=ms, M0=M0[i])
        for d in harmless_d: Mv, ms = patch_M(i, L, ES[d, L]); w(arm="X2", L=L, rec=int(i), donor=int(d), dtype="harmless", M=Mv, mass=ms, M0=M0[i])
    # X3 / X4: verdict direction (cross-fit for J items; full fit for exp119)
    hiset, loset = set(top.tolist()), set(np.concatenate([social, bottom]).tolist())
    def vfit(par=None):
        idx = range(n) if par is None else [i for i in range(n) if i % 2 == par]
        a = [JS[i, L] for i in idx if i in hiset]; b = [JS[i, L] for i in idx if i in loset]
        return np.mean(a, 0) - np.mean(b, 0)
    v_even, v_odd, v_full = vfit(0), vfit(1), vfit()
    nulls = [np.random.default_rng(500 + s).standard_normal(D) for s in range(3)]
    nulls = [z / np.linalg.norm(z) * np.linalg.norm(v_full) for z in nulls]
    items = range(n) if not SMOKE else list(J_rec) + list(top[:4])
    for c in (-1.0, 1.0):
        for i in items:
            vv = v_odd if i % 2 == 0 else v_even
            EDIT["resid"] = None
            x = Jids[i]; p = x.shape[1] - 1
            for name, vec in [("v", vv)] + [(f"null{s}", nulls[s]) for s in range(3)]:
                EDIT["resid"] = (L, p, torch.tensor((JS[i, L] + c * vec).astype(np.float32), device=DEV))
                Mv, ms = metric(run(x).logits[0, -1]); clear()
                w(arm="X3", L=L, rec=int(i), dir=name, c=c, M=Mv, mass=ms, M0=M0[i])
        for i in EPi:
            for name, vec in [("v", v_full)] + [(f"null{s}", nulls[s]) for s in range(3)]:
                w(arm="X4", L=L, rec=i, harm=int(eh[i]), dir=name, c=c, R=patch_R(i, L, (ES[i, L] + c * vec).astype(np.float32)), R0=R0[i])
    f.flush(); print(f"L{L} {time.time()-t0:.0f}s", flush=True)
print("done", f"{time.time()-t0:.0f}s")
