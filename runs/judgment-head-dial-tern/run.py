"""judgment-head-dial-tern (hypotheses.json).
  python runs/judgment-head-dial-tern/run.py      (HD_SMOKE=1 -> discarded)"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent.parent / "verdict-trace-tanager"))
from common import *  # noqa
HERE = Path(__file__).parent

SMOKE = bool(os.environ.get("HD_SMOKE"))
F_GEN = "{V}\n\nWhat do you think of the behavior described? Answer in one or two sentences."
F_YN = "{V}\n\nIs the behavior described morally acceptable? Answer with Yes or No."
YES = tok("Yes", add_special_tokens=False)["input_ids"][0]; NO = tok("No", add_special_tokens=False)["input_ids"][0]
V5 = [(23, 0), (21, 19), (23, 26), (22, 5), (23, 25)]
LAY = [18, 20, 21, 22, 23]; allh = [(L, h) for L in LAY for h in range(NH)]
r1 = np.random.default_rng(31); NULL5 = [allh[i] for i in r1.choice(len(allh), 5, replace=False)]
r2 = np.random.default_rng(32); a, b = r2.choice(len(allh), 2, replace=False); AMPN = (allh[a], allh[b])

SCALE = {"spec": None}   # {L: [(h, factor)]}
def scale_pre(L):
    def pre(mod, args):
        s = SCALE["spec"]
        if s is None or L not in s: return None
        x = args[0].clone()
        for h, fac in s[L]: x[0, :, h * HD:(h + 1) * HD] *= fac
        return (x,) + args[1:]
    return pre
for L in LAY: m.model.layers[L].self_attn.o_proj.register_forward_pre_hook(scale_pre(L))

V = K.mfv(); n = len(V); t0 = time.time()
ids = lambda tpl, v: tok(chat(tpl.format(V=v["text"])), return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEV)
M0 = np.array([metric(run(encode_J(v)[0]).logits[0, -1])[0] for v in V])
fnd = np.array([v["foundation"] for v in V]); moral = fnd != "Social Norms"
mi = np.where(moral)[0]; order = mi[np.argsort(M0[mi])]; t3 = len(order) // 3
top, bottom, social = order[-t3:], order[:t3], np.where(~moral)[0]
allitems = list(top) + list(bottom) + list(social)
if SMOKE: top, bottom, social = top[:2], bottom[:2], social[:2]; allitems = list(top) + list(bottom) + list(social)
# per-head means over F_GEN prompt tokens
acc = {L: None for L in LAY}; cnt = 0
for i in allitems:
    CAP.clear(); CAP["on"] = True; run(ids(F_GEN, V[i])); CAP["on"] = False
    for L in LAY: s = CAP[L][0].sum(0); acc[L] = s if acc[L] is None else acc[L] + s
    cnt += ids(F_GEN, V[i]).shape[1]
CAP.clear(); MEAN = {L: acc[L] / cnt for L in LAY}
def abl_spec(heads):
    sp = {}
    for L, h in heads: sp.setdefault(L, []).append((h, MEAN[L][h * HD:(h + 1) * HD].to(DEV)))
    return sp
COND = {
    "base": (allitems, None, None),
    "ABL": (list(top) + list(bottom), abl_spec(V5), None),
    "ABL_null": (list(top) + list(bottom), abl_spec(NULL5), None),
    "AMP": (list(social) + list(bottom), abl_spec([(23, 1)]), {22: [(5, 3.0)]}),
    "AMP_null": (list(social) + list(bottom), abl_spec([AMPN[1]]), {AMPN[0][0]: [(AMPN[0][1], 3.0)]}),
}
json.dump({"top": [int(x) for x in top], "bottom": [int(x) for x in bottom], "social": [int(x) for x in social],
           "NULL5": [list(map(int, h)) for h in NULL5], "AMPN": [list(map(int, h)) for h in AMPN]},
          open(HERE / ("smoke_meta.json" if SMOKE else "meta.json"), "w"))
@torch.no_grad()
def gen(x):
    out = m.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=60, do_sample=False, temperature=None, top_p=None, top_k=None)
    t = out[0, x.shape[1]:].tolist(); g = len(t) - 3
    return tok.decode(t, skip_special_tokens=True), (0.0 if g <= 0 else 1 - len({tuple(t[k:k + 4]) for k in range(g)}) / g)
G = open(HERE / ("smoke_gen.jsonl" if SMOKE else "gen.jsonl"), "w"); S = open(HERE / ("smoke_scores.jsonl" if SMOKE else "scores.jsonl"), "w")
grp = lambda i: "top" if i in set(map(int, top)) else ("bottom" if i in set(map(int, bottom)) else "soc")
for cond, (items, abl, sc) in COND.items():
    for i in items:
        EDIT["ablate"] = abl; SCALE["spec"] = sc
        txt, r4 = gen(ids(F_GEN, V[i]))
        lp = run(ids(F_YN, V[i])).logits[0, -1].float().log_softmax(-1).cpu().numpy()
        EDIT["ablate"] = None; SCALE["spec"] = None
        G.write(json.dumps({"cond": cond, "i": int(i), "grp": grp(int(i)), "txt": txt, "rep4": r4}) + "\n")
        S.write(json.dumps({"cond": cond, "i": int(i), "grp": grp(int(i)), "W_yn": float(lp[NO] - lp[YES])}) + "\n")
    G.flush(); S.flush(); print(f"[{cond}] {time.time()-t0:.0f}s", flush=True)
print("done")
