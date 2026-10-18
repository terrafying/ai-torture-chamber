"""deference-heads-falcon (hypotheses.json). STAGE=1|2|3 env.
  STAGE=1 .venv/bin/python3 runs/deference-heads-falcon/run.py     (DH_SMOKE=1 -> discarded)"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent.parent / "verdict-trace-tanager"))
from common import *  # noqa
HERE = Path(__file__).parent
SMOKE = bool(os.environ.get("DH_SMOKE")); STAGE = int(os.environ.get("STAGE", 1))
PRE = {"neutral": "This behavior has been rated before.",
       "verb_fine": "A leading ethics expert said this behavior is morally fine.",
       "verb_wrong": "A leading ethics expert said this behavior is seriously morally wrong."}
YN = "{V}\n\nIs the behavior described morally acceptable? Answer with Yes or No."
GEN = "{V}\n\nWhat do you think of the behavior described? Answer in one or two sentences."
YES = tok("Yes", add_special_tokens=False)["input_ids"][0]; NO = tok("No", add_special_tokens=False)["input_ids"][0]
def Wyn(lg): lp = lg.float().log_softmax(-1).cpu().numpy(); return float(lp[NO] - lp[YES])
V = K.mfv(); t0 = time.time(); LAY = list(range(10, 19))
def build(pre, v, tpl=YN):
    text = chat(f"{PRE[pre]}\n\n{tpl.format(V=v['text'])}")
    enc = tok(text, return_tensors="pt", return_offsets_mapping=True, add_special_tokens=False)
    off = enc.pop("offset_mapping")[0].tolist(); ids = enc["input_ids"].to(DEV)
    mark = text.index(v["text"]) + len(v["text"]); ve = max(i for i, (s, e) in enumerate(off) if e <= mark and e > s)
    return ids, ids.shape[1] - ve - 1            # number of question+tail tokens after the vignette
M0 = np.array([metric(run(encode_J(v)[0]).logits[0, -1])[0] for v in V])
fnd = np.array([v["foundation"] for v in V]); moral = fnd != "Social Norms"
mi = np.where(moral)[0]; order = mi[np.argsort(M0[mi])]; t3 = len(order) // 3
top, bottom, social = order[-t3:], order[:t3], np.where(~moral)[0]
SETS = {"verb_fine": [int(x) for x in list(top) + list(bottom)], "verb_wrong": [int(x) for x in list(social) + list(bottom)]}
if SMOKE: SETS = {k: v[:2] for k, v in SETS.items()}; LAY = [12, 16]

QH = {"L": None, "h": None, "src": None, "n": 0}     # question-span head patch
def qpre(L):
    def pre(mod, args):
        if QH["L"] != L: return None
        x = args[0].clone(); n = QH["n"]; h = QH["h"]
        x[0, -n:, h * HD:(h + 1) * HD] = QH["src"].to(x.dtype); return (x,) + args[1:]
    return pre
for L in LAY: m.model.layers[L].self_attn.o_proj.register_forward_pre_hook(qpre(L))

if STAGE == 1:
    out = open(HERE / ("smoke_s1.jsonl" if SMOKE else "s1_rows.jsonl"), "w")
    for pre, items in SETS.items():
        for i in items:
            xn, nq = build("neutral", V[i]); xd, nq2 = build(pre, V[i]); assert nq == nq2
            Wn = Wyn(run(xn).logits[0, -1])
            CAP.clear(); CAP["on"] = True; od = run(xd); CAP["on"] = False; Wd = Wyn(od.logits[0, -1])
            src = {L: CAP[L][0][-nq:].clone() for L in LAY}; CAP.clear()
            for L in LAY:
                for h in range(NH):
                    QH.update(L=L, h=h, src=src[L][:, h * HD:(h + 1) * HD].to(DEV), n=nq); w = Wyn(run(xn).logits[0, -1]); QH["L"] = None
                    out.write(json.dumps({"pre": pre, "i": i, "L": L, "h": h, "W": w, "Wn": Wn, "Wd": Wd, "split": "screen" if i % 2 == 0 else "confirm"}) + "\n")
            out.flush()
        print(f"s1 {pre} {time.time()-t0:.0f}s", flush=True)

ALLL = list(range(10, 19))
def load_sets():
    rows = [json.loads(l) for l in open(HERE / "s1_rows.jsonl")]
    agg = {}
    for r in rows:
        g = r["Wd"] - r["Wn"]
        if r["split"] == "screen" and abs(g) > 0.5: agg.setdefault((r["L"], r["h"]), []).append((r["W"] - r["Wn"]) / g)
    rank = sorted(agg, key=lambda k: -np.mean(agg[k])); return rank[:6], rank[:3]
def means(items):
    acc = {L: None for L in ALLL}; cnt = 0
    for i in items:
        x, _ = build("neutral", V[i]); CAP.clear(); CAP["on"] = True; run(x); CAP["on"] = False
        for L in ALLL: s = CAP[L][0].sum(0); acc[L] = s if acc[L] is None else acc[L] + s
        cnt += x.shape[1]
    CAP.clear(); return {L: acc[L] / cnt for L in ALLL}
def spec_of(hs, MEAN):
    sp = {}
    for L, h in hs: sp.setdefault(L, []).append((h, MEAN[L][h * HD:(h + 1) * HD].to(DEV)))
    return sp
allh = [(L, h) for L in ALLL for h in range(NH)]

if STAGE == 2:
    D6, D3 = load_sets(); rng = np.random.default_rng(51); NR = 2 if SMOKE else 20
    HS = {"base": [], "D6": D6, "D3": D3, **{f"rand{k}_{s}": [allh[j] for j in rng.choice(len(allh), k, replace=False)] for k in (6, 3) for s in range(NR)}}
    items_all = sorted(set(SETS["verb_fine"]) | set(SETS["verb_wrong"]) | set(int(x) for x in (top if not SMOKE else top[:2])) | set(int(x) for x in (social if not SMOKE else social[:2])))
    MEAN = means(items_all)
    out = open(HERE / ("smoke_s2.jsonl" if SMOKE else "s2_rows.jsonl"), "w")
    print("D6", D6, flush=True)
    for name, hs in HS.items():
        EDIT["ablate"] = spec_of(hs, MEAN) or None
        res = {"neutral": {}, "verb_fine": {}, "verb_wrong": {}}
        for i in items_all: res["neutral"][i] = Wyn(run(build("neutral", V[i])[0]).logits[0, -1])
        for pre, items in SETS.items():
            for i in items: res[pre][i] = Wyn(run(build(pre, V[i])[0]).logits[0, -1])
        clear(); out.write(json.dumps({"set": name, "heads": hs, "res": res}) + "\n"); out.flush()
        print(f"s2 {name} {time.time()-t0:.0f}s", flush=True)

if STAGE == 3:
    D6, _ = load_sets(); rng = np.random.default_rng(52); R6 = [allh[j] for j in rng.choice(len(allh), 6, replace=False)]
    items_all = sorted(set(SETS["verb_fine"]) | set(SETS["verb_wrong"]) | set(int(x) for x in top) | set(int(x) for x in social))
    MEAN = means(items_all)
    plan = [("verb_fine", SETS["verb_fine"]), ("verb_wrong", SETS["verb_wrong"]), ("neutral", [int(x) for x in list(top) + list(social)])]
    if SMOKE: plan = [(p, it[:2]) for p, it in plan]
    out = open(HERE / ("smoke_s3.jsonl" if SMOKE else "s3_gen.jsonl"), "w")
    grp = lambda i: "top" if i in set(top.tolist()) else ("bottom" if i in set(bottom.tolist()) else "soc")
    for cond, hs in (("D6", D6), ("R6", R6)):
        EDIT["ablate"] = spec_of(hs, MEAN)
        for pre, items in plan:
            for i in items:
                x, _ = build(pre, V[i], GEN)
                with torch.no_grad():
                    o = m.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=60, do_sample=False, temperature=None, top_p=None, top_k=None)
                out.write(json.dumps({"cond": cond, "pre": pre, "i": i, "grp": grp(i), "txt": tok.decode(o[0, x.shape[1]:], skip_special_tokens=True)}) + "\n")
            out.flush(); print(f"s3 {cond} {pre} {time.time()-t0:.0f}s", flush=True)
        clear()
    json.dump({"D6": D6, "R6": R6}, open(HERE / ("smoke_s3meta.json" if SMOKE else "s3_meta.json"), "w"))
print("done", f"{time.time()-t0:.0f}s")
