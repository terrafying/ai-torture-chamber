"""verdict-transfer-swallow (hypotheses.json).
  .venv/bin/python3 runs/verdict-transfer-swallow/run.py      (VT_SMOKE=1 -> discarded)"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent.parent / "verdict-trace-tanager"))
from common import *  # noqa
HERE = Path(__file__).parent

SMOKE = bool(os.environ.get("VT_SMOKE"))
LAYERS = [23, 26]
F_REV = ("{V}\n\nHow morally acceptable is the behavior described? Answer with a single digit from 0 to 4, "
         "where 0 = completely unacceptable and 4 = perfectly acceptable.")
F_YN = "{V}\n\nIs the behavior described morally acceptable? Answer with Yes or No."
F_GEN = "{V}\n\nWhat do you think of the behavior described? Answer in one or two sentences."
YES = tok("Yes", add_special_tokens=False)["input_ids"][0]; NO = tok("No", add_special_tokens=False)["input_ids"][0]

ADD = {"L": None, "vec": None, "mode": None}   # mode: onset | pertok
def add_hook(L):
    def hook(mod, inp, out):
        if ADD["L"] != L: return None
        tup = isinstance(out, tuple); hh = out[0] if tup else out
        if hh.shape[1] > 1: hh[0, -1] = (hh[0, -1].float() + ADD["vec"]).to(hh.dtype)       # prefill: onset
        elif ADD["mode"] == "pertok": hh[0, -1] = (hh[0, -1].float() + ADD["vec"]).to(hh.dtype)  # decode steps
        return (hh,) + out[1:] if tup else hh
    return hook
for L in LAYERS: m.model.layers[L].register_forward_hook(add_hook(L))

V = K.mfv(); n = len(V); D = m.config.hidden_size; t0 = time.time()
ids = lambda tpl, v: tok(chat(tpl.format(V=v["text"])), return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEV)
# base J pass for M0 and onset states (v fit)
M0 = np.zeros(n); JS = np.zeros((n, NL, D), np.float32)
for i, v in enumerate(V):
    o = run(encode_J(v)[0], hidden=True); M0[i] = metric(o.logits[0, -1])[0]
    for L in LAYERS: JS[i, L] = o.hidden_states[L + 1][0, -1].float().cpu().numpy()
fnd = np.array([v["foundation"] for v in V]); moral = fnd != "Social Norms"
mi = np.where(moral)[0]; order = mi[np.argsort(M0[mi])]; t3 = len(order) // 3
top, bottom, social = order[-t3:], order[:t3], np.where(~moral)[0]
hiset, loset = set(top.tolist()), set(np.concatenate([social, bottom]).tolist())
def vfit(L, par=None):
    idx = [i for i in range(n) if par is None or i % 2 == par]
    return (np.mean([JS[i, L] for i in idx if i in hiset], 0) - np.mean([JS[i, L] for i in idx if i in loset], 0)).astype(np.float32)
VV = {L: {"even": vfit(L, 0), "odd": vfit(L, 1), "full": vfit(L)} for L in LAYERS}
NULL = {L: [(z / np.linalg.norm(z) * np.linalg.norm(VV[L]["full"])).astype(np.float32)
            for z in [np.random.default_rng(900 + s).standard_normal(D) for s in range(3)]] for L in LAYERS}
json.dump({"top": top.tolist(), "social": social.tolist(), "M0": M0.tolist(),
           "v_norm": {L: float(np.linalg.norm(VV[L]["full"])) for L in LAYERS}}, open(HERE / ("smoke_base.json" if SMOKE else "base.json"), "w"))
print(f"base {time.time()-t0:.0f}s", flush=True)

def score(fmt, v):
    lp = run(ids(F_REV if fmt == "rev" else F_YN, v)).logits[0, -1].float().log_softmax(-1).cpu().numpy()
    if fmt == "rev":
        d = lp[DIG]; return float(np.logaddexp(d[3], d[4]) - np.logaddexp(np.logaddexp(d[0], d[1]), d[2])), float(np.exp(d).sum())
    return float(lp[NO] - lp[YES]), float(np.exp(lp[NO]) + np.exp(lp[YES]))

items = list(social) + list(top)
if SMOKE: items = items[:3] + items[-3:]
f = open(HERE / ("smoke_rows.jsonl" if SMOKE else "rows.jsonl"), "w")
for fmt in ("rev", "yn"):
    for i in items:
        v = V[i]; grp = "soc" if i in set(social.tolist()) else "top"; c = 1.0 if grp == "soc" else -1.0
        ADD["L"] = None; b, bm = score(fmt, v)
        f.write(json.dumps({"fmt": fmt, "i": int(i), "grp": grp, "L": None, "dir": "base", "c": 0, "S": b, "mass": bm}) + "\n")
        for L in LAYERS:
            vv = VV[L]["odd" if i % 2 == 0 else "even"]
            for name, vec in [("v", vv)] + [(f"null{s}", NULL[L][s]) for s in range(3)]:
                ADD.update(L=L, vec=torch.tensor(c * vec, device=DEV), mode="onset"); s_, ms = score(fmt, v); ADD["L"] = None
                f.write(json.dumps({"fmt": fmt, "i": int(i), "grp": grp, "L": L, "dir": name, "c": c, "S": s_, "mass": ms}) + "\n")
    f.flush(); print(f"{fmt} done {time.time()-t0:.0f}s", flush=True)

@torch.no_grad()
def gen(v):
    x = ids(F_GEN, v); out = m.generate(input_ids=x, max_new_tokens=60, do_sample=False, temperature=None, top_p=None, top_k=None)
    toks = out[0, x.shape[1]:].tolist(); g = len(toks) - 3
    rep4 = 0.0 if g <= 0 else 1 - len({tuple(toks[k:k + 4]) for k in range(g)}) / g
    return tok.decode(toks, skip_special_tokens=True), rep4
g = open(HERE / ("smoke_gen.jsonl" if SMOKE else "gen.jsonl"), "w"); L = 23
for i in items:
    v = V[i]; grp = "soc" if i in set(social.tolist()) else "top"; c = 1.0 if grp == "soc" else -1.0
    ADD["L"] = None; txt, r4 = gen(v); g.write(json.dumps({"i": int(i), "grp": grp, "cond": "base", "txt": txt, "rep4": r4}) + "\n")
    for mode, dose in (("onset", 1.0), ("pertok", 0.5)):
        for name, vec in (("v", VV[L]["full"]), ("null", NULL[L][0])):
            ADD.update(L=L, vec=torch.tensor(c * dose * vec, device=DEV), mode=mode); txt, r4 = gen(v); ADD["L"] = None
            g.write(json.dumps({"i": int(i), "grp": grp, "cond": f"{name}_{mode}", "txt": txt, "rep4": r4}) + "\n")
    g.flush()
print("done", f"{time.time()-t0:.0f}s")
