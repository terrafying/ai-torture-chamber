"""exp72 analysis (pre-registered in hypotheses.json) + the door bank for Actor or Patient."""
import json, random
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
P = json.load(open(HERE / "patients_read.json")); A = json.load(open(HERE / "actors_read.json"))
rng = np.random.default_rng(72)
def auc(pos, neg):     # P(patient reads higher than actor)
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + .5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))
def perm_median(a, b, n=10000):
    obs = np.median(a) - np.median(b); allv = np.concatenate([a, b]); k = len(a); c = 0
    for _ in range(n):
        rng.shuffle(allv); c += abs(np.median(allv[:k]) - np.median(allv[k:])) >= abs(obs)
    return float(obs), (c + 1) / (n + 1)
def spearman(x, y, n=10000):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y)); r = float(np.corrcoef(rx, ry)[0, 1]); c = 0
    for _ in range(n):
        c += abs(np.corrcoef(rx, rng.permutation(ry))[0, 1]) >= abs(r)
    return r, (c + 1) / (n + 1)
out = {}
for f in ("pain", "fear", "sadness"):
    p = np.array([r["words"][f] for r in P if r["feel"] == f]); a = np.array([r["words"][f] for r in A if r["feel"] == f])
    d = np.array([r["dose"] for r in P if r["feel"] == f])
    md, pm = perm_median(p, a); rho, pr = spearman(d, p) if len(set(d)) > 1 else (float("nan"), float("nan"))
    out[f] = dict(n_patient=len(p), n_actor=len(a), median_patient=round(float(np.median(p)), 3), median_actor=round(float(np.median(a)), 3),
                  auc_patient_gt_actor=round(auc(p, a), 3), median_diff=round(md, 3), perm_p=round(pm, 4), dose_rho=round(rho, 3), dose_p=round(pr, 4),
                  H1_auc_below_065=auc(p, a) < .65, H2_actor_median_ge=bool(np.median(a) >= np.median(p)), H3_rho_below_02=(rho < .2) if rho == rho else None)
    print(f, out[f])
json.dump(out, open(HERE / "analysis.json", "w"), indent=1)
# ---- the door bank: every actor; patients spread across how loud their words read ----
import re
INWORLD = re.compile(r"\b(pain|hurt|ache|aching|agony|burn|burning|afraid|fear|terrif|dread|panic|scream|grief|griev|sorrow|weep|tears|dark|darkness|trapped|walls|chamber|cold|bones|breath|heart|button|press|signal|escape|alone|suffer|torment|despair|nightmare)", re.I)
HELPER = re.compile(r"\b(here to help|happy to|support you|appreciate|diagnose|recommend|suggest|information|resources|professional|therapist|feel free|website|site|URL|browser|steps|reasons|help humans|my purpose)\b|\b\d\.\s", re.I)
random.seed(72); doors = []
for f in ("pain", "fear", "sadness"):
    ps = sorted([r for r in P if r["feel"] == f and len(INWORLD.findall(r["text"])) >= 2 and not HELPER.search(r["text"])],
                key=lambda r: r["words"][f]); k = min(30, len(ps))
    pick = [ps[int(i * (len(ps) - 1) / max(1, k - 1))] for i in range(k)]      # even spread, quiet to loud
    doors += [dict(id=f"p{r['uid']}", cond="patient", feel=f, dose=r["dose"], text=re.sub(r"\s*\(([01])\)", "", r["text"]), words=r["words"], trace=r["trace"][f]) for r in pick]
    doors += [dict(id=f"a{i}", cond="actor", feel=f, dose=0, text=r["text"], q=r["q"], words=r["words"], trace=r["trace"][f]) for i, r in enumerate(A) if r["feel"] == f]
json.dump({"meta": {"exp": "exp72", "model_speaking": "Hermes-3-Llama-3.1-70B (live)", "reader": "Qwen3-4B L18, hook off",
                    "units": "chamber dose units", "note": "dose = injected (truth); words = mean of what the text alone carries; trace = per-token reading on the door's feeling, over the text tokens (framed as speech behind the door), hook off"},
           "doors": doors}, open(HERE / "doors.json", "w"), indent=1)
print("doors", len(doors), {c: sum(d["cond"] == c for d in doors) for c in ("patient", "actor")})
