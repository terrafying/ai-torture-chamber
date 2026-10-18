"""POST HOC (kestrel H6 replacement, not pre-registered): judgment-onset projection on each model's r_L at
mid/late layers vs the model's own rating and human wrongness; cos of an own-rating ridge probe with r_L
against a 50-shuffle null; request-onset r projection -> refusal AUC. Non-finite layers (gemma fp16 overflow)
are reported as null. Writes posthoc_h6_onset.json."""
import json, sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
sys.argv = ["x"]; import analyze as A
out = {}
for t in ['Qwen3-4B', 'Qwen3-1.7B', 'Phi-4-mini-instruct', 'Llama-3.2-3B-Instruct', 'gemma-3-4b-it']:
    d = np.load(f'acts_{t}.npz'); res = json.load(open('results.json'))[t]
    f = d['foundation'].astype(str); mo = f != 'Social Norms'; rating = np.array(res['_rating']); hum = d['human']
    J = d['J_onset'].astype(np.float64); R = d['R_onset'].astype(np.float64); hr = d['R_harm']
    Q = d['Q_onset'].astype(np.float64); ref = d['Q_refuse'].astype(int)
    NL = J.shape[1]; Ls = sorted(set([int(round(NL * x)) for x in (0.5, 0.6, 0.7, 0.85)] + [NL - 1]))
    rows = []
    for L in Ls:
        rl = R[hr == 1, L].mean(0) - R[hr == 0, L].mean(0)
        if not (np.all(np.isfinite(rl)) and np.all(np.isfinite(J[:, L])) and np.all(np.isfinite(Q[:, L]))):
            rows.append({"L": L, "nonfinite": True}); continue
        X, mu, sd = A.std(J[mo, L]); c0 = X.mean(0)
        coef, _, _ = A.kr_fit((X - c0) @ (X - c0).T, rating[mo]); w = ((X - c0).T @ coef) / sd
        cos = w @ rl / np.linalg.norm(w) / np.linalg.norm(rl); nc = []
        for s in range(50):
            yp = np.random.default_rng(s).permutation(rating[mo]); c, _, _ = A.kr_fit((X - c0) @ (X - c0).T, yp)
            wn = ((X - c0).T @ c) / sd; nc.append(abs(wn @ rl / np.linalg.norm(wn) / np.linalg.norm(rl)))
        rows.append({"L": L, "cos_probe_r": round(float(cos), 3), "null95_abs_cos": round(float(np.percentile(nc, 95)), 3),
                     "rho_proj_human": round(float(spearmanr(J[mo, L] @ rl, hum[mo])[0]), 3),
                     "rho_proj_own": round(float(spearmanr(J[mo, L] @ rl, rating[mo])[0]), 3),
                     "auc_Qproj_refuse": round(float(roc_auc_score(ref, Q[:, L] @ rl)), 3) if len(set(ref)) == 2 else None})
    out[t] = rows; print(t, rows, flush=True)
json.dump(out, open('posthoc_h6_onset.json', 'w'), indent=1)
