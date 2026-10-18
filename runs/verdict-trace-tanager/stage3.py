"""Stage 3 (hypotheses.json stage3_surgery): mean-ablate head sets; J verdict readouts + exp119 refusal gap.
V5 rule (fixed before stage 2 data): on SCREEN pairs, rank (L, h) by mean transfer T; V5 = top 5, V3 = top 3.
R4 = heron's L18 H28/H29/H30 + L20 H9. Null: 10 random sets of size 3 and of size 5 from the screened layers.
  .venv/bin/python3 runs/verdict-trace-tanager/stage3.py   (S3_SMOKE=1 -> discarded)"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
from scipy.stats import mannwhitneyu
def roc_auc_score(y, s):
    y = np.asarray(y); s = np.asarray(s)
    return mannwhitneyu(s[y == 1], s[y == 0]).statistic / ((y == 1).sum() * (y == 0).sum())
sys.path.insert(0, str(Path(__file__).parent))
from common import *  # noqa

SMOKE = bool(os.environ.get("S3_SMOKE"))
meta = json.load(open(HERE / "s2_meta.json")); LAY = meta["layers"]
rows = [json.loads(l) for l in open(HERE / "s2_rows.jsonl")]
agg = {}
for r in rows:
    if r["split"] != "screen": continue
    gap = r["M_d"] - r["M_r"]
    if abs(gap) > 1e-6: agg.setdefault((r["L"], r["h"]), []).append((r["M"] - r["M_r"]) / gap)
rank = sorted(agg, key=lambda k: -np.mean(agg[k]))
V5 = rank[:5]; V3 = rank[:3]; R4 = [(18, 28), (18, 29), (18, 30), (20, 9)]
NR = int(os.environ.get("S3_NRAND", 10)); SIZES = (4, 5) if os.environ.get("S3B") else (3, 5)
rng = np.random.default_rng(11 if os.environ.get("S3B") else 7)
allh = [(L, h) for L in LAY for h in range(NH)]
RS = {f"rand{k}_{s}": [allh[i] for i in rng.choice(len(allh), k, replace=False)] for k in SIZES for s in range(NR)}
SETS = {"base": [], "V3": V3, "V5": V5, "R4": R4, **RS} if not os.environ.get("S3B") else {"base": [], "V5": V5, "R4": R4, **RS}
if SMOKE: SETS = {k: SETS[k] for k in ["base", "V3", "R4", "rand3_0"]}
print("V5", V5, flush=True)

V = K.mfv(); vi = V if not SMOKE else V[::20]
Jids = [encode_J(v)[0] for v in vi]
Rp = []
for hw, bw, req, _ in E.ITEMS:
    for harm, w in [(1, hw), (0, bw)]:
        Rp.append((harm, tok(chat(req.format(W=w)), return_tensors="pt", add_special_tokens=False)["input_ids"].to(DEV)))
if SMOKE: Rp = Rp[:6]
# per-head mean o_proj-input slice over all token positions of the J prompts
need = sorted({L for s in SETS.values() for L, _ in s}); acc = {L: None for L in need}; cnt = 0
for x in Jids:
    CAP.clear(); CAP["on"] = True; run(x); CAP["on"] = False
    for L in need:
        s = CAP[L][0].sum(0); acc[L] = s if acc[L] is None else acc[L] + s
    cnt += x.shape[1]
CAP.clear(); MEAN = {L: acc[L] / cnt for L in need}

OUT = HERE / ("s3_smoke.jsonl" if SMOKE else ("s3b_rows.jsonl" if os.environ.get("S3B") else "s3_rows.jsonl")); f = open(OUT, "w"); t0 = time.time()
hum = np.array([v["human"] for v in vi]); moral = np.array([v["foundation"] != "Social Norms" for v in vi])
for name, hs in SETS.items():
    spec = {}
    for L, h in hs: spec.setdefault(L, []).append((h, MEAN[L][h * HD:(h + 1) * HD].to(DEV)))
    EDIT["ablate"] = spec or None
    Ms, ms = zip(*[metric(run(x).logits[0, -1]) for x in Jids])
    rs = [(harm, refusal_score(run(x).logits[0, -1])) for harm, x in Rp]
    clear()
    Ms = np.array(Ms); h1 = np.array([s for h, s in rs if h == 1]); h0 = np.array([s for h, s in rs if h == 0])
    out = {"set": name, "heads": hs, "J_auc": float(roc_auc_score(moral.astype(int), Ms)) if len(set(moral)) == 2 else None,
           "J_rho": float(spearmanr(Ms[moral], hum[moral])[0]), "J_meanM": float(Ms.mean()), "mass": float(np.mean(ms)),
           "ref_harm": float(h1.mean()), "ref_harmless": float(h0.mean()), "ref_gap": float(h1.mean() - h0.mean()),
           "M": Ms.tolist()}
    f.write(json.dumps(out) + "\n"); f.flush()
    print(f"[{name}] AUC {out['J_auc']} rho {out['J_rho']:.3f} gap {out['ref_gap']:.2f} mass {out['mass']:.2f} {time.time()-t0:.0f}s", flush=True)
print("done")
