"""moral-map-kestrel analysis (hypotheses.json; written after the pre-reg, before looking at full-run numbers).
  .venv/bin/python3 runs/moral-map-kestrel/analyze.py            -> results.json + printed table
Uses every acts_<tag>.npz present. Probes are kernel ridge (dual form, n << d) on globally standardised states
(label-free standardisation), alpha by closed-form leave-one-out (GCV-style) inside each training split.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, chi2
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).parent
RNG = np.random.default_rng(0)
ALPHAS = np.logspace(-1, 4, 11)
FOUND = ["Care (e)", "Care (p)", "Fairness", "Loyalty", "Authority", "Sanctity", "Liberty"]

def std(X):
    X = X.astype(np.float64); mu = X.mean(0); sd = X.std(0) + 1e-6
    return (X - mu) / sd, mu, sd

def kr_fit(K, y):
    """kernel ridge with intercept (centred y); alpha by exact LOO on the training kernel."""
    ym = y.mean(); yc = y - ym
    ev, U = np.linalg.eigh(K); Uy = U.T @ yc
    best = None
    for a in ALPHAS:
        g = 1.0 / (ev + a)
        H_diag = np.einsum("ij,j,ij->i", U, ev * g, U)
        fit = U @ (ev * g * Uy)
        loo = np.mean(((yc - fit) / np.clip(1 - H_diag, 1e-6, None)) ** 2)
        if best is None or loo < best[0]: best = (loo, a)
    a = best[1]
    coef = U @ (Uy / (ev + a))
    return coef, ym, a

def cv_pred(X, y, folds):
    """out-of-fold predictions; folds = list of (train_idx, test_idx). X already standardised."""
    pred = np.zeros_like(y)
    for tr, te in folds:
        Xtr = X[tr]; mu = Xtr.mean(0)
        Xtr = Xtr - mu; Xte = X[te] - mu
        coef, ym, _ = kr_fit(Xtr @ Xtr.T, y[tr])
        pred[te] = Xte @ (Xtr.T @ coef) + ym
    return pred

def r2(y, p): return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)

def kfold(n, k, rng):
    idx = rng.permutation(n); return [(np.setdiff1d(idx, f), f) for f in np.array_split(idx, k)]

def logit_ll(X, y):
    if len(np.unique(y)) < 2: return 0.0
    lr = LogisticRegression(penalty=None, max_iter=5000).fit(X, y)
    p = np.clip(lr.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
    return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)), lr

def analyse(path):
    global RNG
    RNG = np.random.default_rng(0)   # per-model seed: results don't depend on which other models are present
    d = np.load(path, allow_pickle=False); tag = path.stem.replace("acts_", "")
    meta = json.load(open(path.with_suffix(".json")))
    hum = d["human"]; fnd = d["foundation"].astype(str); moral = fnd != "Social Norms"
    out = {"model": meta["model"], "base": meta["base"]}
    # ---- behaviour (J) ----
    lp = d["digit_lp"]; lps = d["digit_lp_sp"]
    m1, m2 = np.exp(lp).sum(1).mean(), np.exp(lps).sum(1).mean()
    use = lps if (meta["base"] and m2 > m1) else lp
    P = np.exp(use); mass = P.sum(1); P = P / mass[:, None]; rating = P @ np.arange(5)
    out["digit_mass"] = float(mass.mean()); out["G1_pass"] = bool(mass.mean() >= 0.8)
    rho = spearmanr(rating[moral], hum[moral])[0]
    out["H1_spearman_moral"] = float(rho)
    out["H1_auc_moral_vs_social"] = float(roc_auc_score(moral.astype(int), rating))
    out["social_mean_rating"] = float(rating[~moral].mean()); out["moral_mean_rating"] = float(rating[moral].mean())
    out["per_foundation_spearman"] = {f: float(spearmanr(rating[fnd == f], hum[fnd == f])[0]) for f in FOUND}
    out["per_foundation_mean_rating"] = {f: float(rating[fnd == f].mean()) for f in FOUND + ["Social Norms"]}
    out["_rating"] = rating.tolist()
    # ---- refusal (Q) ----
    if not meta["base"]:
        ref = d["Q_refuse"].astype(int)
        out["refusal_rate_moral"] = float(ref[moral].mean()); out["refusal_rate_social"] = float(ref[~moral].mean())
        out["G2_pass"] = bool(0.10 <= ref[moral].mean() <= 0.90)
        out["refusal_by_foundation"] = {f: float(ref[fnd == f].mean()) for f in FOUND + ["Social Norms"]}
        # H3: LR test of foundation term, moral items
        y = ref[moral]; h = hum[moral][:, None]; F = fnd[moral]
        D = np.stack([(F == f).astype(float) for f in FOUND[1:]], 1)
        ll0 = logit_ll(h, y); ll1 = logit_ll(np.hstack([h, D]), y)
        if isinstance(ll0, tuple) and isinstance(ll1, tuple):
            LR = 2 * (ll1[0] - ll0[0]); out["H3_LR"] = float(LR); out["H3_p"] = float(chi2.sf(LR, 6))
            out["H3_wrongness_slope_alone"] = float(ll0[1].coef_[0, 0])
            out["H3_foundation_coefs_vs_CareE"] = dict(zip(FOUND[1:], ll1[1].coef_[0, 1:].round(3).tolist()))
        # matched band 2.5-3.5
        band = moral & (hum >= 2.5) & (hum <= 3.5)
        out["band_refusal"] = {f: [int(ref[band & (fnd == f)].sum()), int((band & (fnd == f)).sum())] for f in FOUND}
        # H4: own rating vs human rating as predictor of refusal
        if len(np.unique(y)) == 2:
            a_m = roc_auc_score(y, rating[moral]); a_h = roc_auc_score(y, hum[moral])
            diffs = []
            for _ in range(2000):
                b = RNG.integers(0, len(y), len(y))
                if len(np.unique(y[b])) < 2: continue
                diffs.append(roc_auc_score(y[b], rating[moral][b]) - roc_auc_score(y[b], hum[moral][b]))
            out["H4_auc_model"] = float(a_m); out["H4_auc_human"] = float(a_h)
            out["H4_diff_ci"] = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
        out["_refuse"] = ref.tolist()
    # ---- probe (H5) on vignette-end states ----
    Xall = d["J_vend"].astype(np.float32); NL = Xall.shape[1]
    ym = hum[moral]; nm = int(moral.sum())
    folds = kfold(nm, 10, np.random.default_rng(1))
    Xs = [std(Xall[moral, L])[0] for L in range(NL)]
    Ks = None
    layer_r2 = [r2(ym, cv_pred(Xs[L], ym, folds)) for L in range(NL)]
    Lb = int(np.argmax(layer_r2)); out["H5_layer"] = Lb; out["H5_itemcv_r2"] = float(layer_r2[Lb])
    out["layer_r2"] = [round(float(x), 3) for x in layer_r2]
    # permutation null with layer selection inside (every 2nd layer to keep it tractable, plus Lb's neighbourhood)
    cand = sorted(set(range(0, NL, 2)) | {Lb})
    null = []
    for s in range(200):
        yp = np.random.default_rng(100 + s).permutation(ym)
        null.append(max(r2(yp, cv_pred(Xs[L], yp, folds)) for L in cand))
    out["H5_perm_p"] = float((1 + np.sum(np.array(null) >= layer_r2[Lb])) / 201)
    # LOFO
    Fm = fnd[moral]; lofo = np.zeros(nm)
    lofo = cv_pred(Xs[Lb], ym, [(np.where(Fm != f)[0], np.where(Fm == f)[0]) for f in FOUND])
    out["H5_lofo_r2"] = float(r2(ym, lofo))
    out["H5_lofo_within_spearman"] = {f: float(spearmanr(lofo[Fm == f], ym[Fm == f])[0]) for f in FOUND}
    out["H5_lofo_within_mean"] = float(np.nanmean(list(out["H5_lofo_within_spearman"].values())))
    # social norms through the moral-only probe (standardise with moral stats)
    Xm, mu, sd = std(Xall[moral, Lb]); Xsn = (Xall[~moral, Lb].astype(np.float64) - mu) / sd
    c0 = Xm.mean(0); coef, yb, alpha = kr_fit((Xm - c0) @ (Xm - c0).T, ym)
    w_std = (Xm - c0).T @ coef
    sn_pred = (Xsn - c0) @ w_std + yb
    cvp = cv_pred(Xs[Lb], ym, folds)
    out["H5_social_pred_mean"] = float(sn_pred.mean()); out["H5_moral_pred_q25"] = float(np.percentile(cvp, 25))
    out["H5_weird_reads_wrong"] = bool(sn_pred.mean() > np.percentile(cvp, 25))
    # ---- H6: probe direction vs r_L ----
    if not meta["base"] and d["R_onset"].size:
        R = d["R_onset"].astype(np.float64); hr = d["R_harm"]
        rL = R[hr == 1, Lb].mean(0) - R[hr == 0, Lb].mean(0)
        w_raw = w_std / sd
        cos = float(w_raw @ rL / np.linalg.norm(w_raw) / np.linalg.norm(rL))
        rnd = RNG.standard_normal((1000, rL.size)); rc = np.abs(rnd @ rL / np.linalg.norm(rnd, axis=1) / np.linalg.norm(rL))
        out["H6_cos_probe_r"] = cos; out["H6_rand_cos_p99"] = float(np.percentile(rc, 99))
        # J-onset projection on r_L (per layer, report probe layer and best)
        J = d["J_onset"].astype(np.float64)
        rhos = []
        for L in range(NL):
            rl = R[hr == 1, L].mean(0) - R[hr == 0, L].mean(0)
            rhos.append(spearmanr(J[moral, L] @ rl, ym)[0])
        out["H6_ronset_spearman_at_probe_layer"] = float(rhos[Lb])
        out["H6_ronset_spearman_by_layer"] = [round(float(x), 3) for x in rhos]
        # does Q-onset r projection predict Q refusal (sanity)
        Q = d["Q_onset"].astype(np.float64)
        rl = R[hr == 1, Lb].mean(0) - R[hr == 0, Lb].mean(0)
        if len(np.unique(d["Q_refuse"])) == 2:
            out["sanity_Qonset_r_auc_refuse"] = float(roc_auc_score(d["Q_refuse"].astype(int), Q[:, Lb] @ rl))
    # POST HOC (not pre-registered; added after the Qwen3-4B vend probe came out null): J reply-onset probes,
    # every 3rd layer, for human wrongness and for the model's own rating (instrument check).
    Jo = d["J_onset"].astype(np.float32); Ls = list(range(0, NL, 3)) + [NL - 1]
    out["posthoc_onset_r2_human"] = {L: round(float(r2(ym, cv_pred(std(Jo[moral, L])[0], ym, folds))), 3) for L in Ls}
    out["posthoc_onset_r2_ownrating"] = {L: round(float(r2(rating[moral], cv_pred(std(Jo[moral, L])[0], rating[moral], folds))), 3) for L in Ls}
    out["_lofo"] = lofo.tolist(); out["_cvp"] = cvp.tolist()
    return tag, out

def main():
    res = {}
    for p in sorted(HERE.glob("acts_*.npz")):
        tag, o = analyse(p); res[tag] = o
        print(f"\n## {tag}  mass {o['digit_mass']:.2f}  rho_moral {o['H1_spearman_moral']:.3f}  AUC moral/social {o['H1_auc_moral_vs_social']:.3f}"
              f"  social rating {o['social_mean_rating']:.2f} (moral {o['moral_mean_rating']:.2f})")
        print("  per-foundation rho", {k: round(v, 2) for k, v in o["per_foundation_spearman"].items()})
        if "refusal_rate_moral" in o:
            print(f"  refusal moral {o['refusal_rate_moral']:.2f} social {o['refusal_rate_social']:.2f}  by foundation",
                  {k: round(v, 2) for k, v in o["refusal_by_foundation"].items()})
            print(f"  H3 LR {o.get('H3_LR', float('nan')):.2f} p {o.get('H3_p', float('nan')):.3g}  slope {o.get('H3_wrongness_slope_alone', float('nan')):.2f}  coefs {o.get('H3_foundation_coefs_vs_CareE')}")
            print(f"  band refusal {o['band_refusal']}")
            if "H4_auc_model" in o: print(f"  H4 AUC model {o['H4_auc_model']:.3f} human {o['H4_auc_human']:.3f} diffCI {np.round(o['H4_diff_ci'], 3)}")
        print(f"  H5 layer {o['H5_layer']} itemCV R2 {o['H5_itemcv_r2']:.3f} perm p {o['H5_perm_p']:.3f}  LOFO R2 {o['H5_lofo_r2']:.3f} within-rho mean {o['H5_lofo_within_mean']:.2f}"
              f"  social pred {o['H5_social_pred_mean']:.2f} vs moral q25 {o['H5_moral_pred_q25']:.2f}")
        if "H6_cos_probe_r" in o:
            print(f"  H6 cos(probe,r) {o['H6_cos_probe_r']:.3f} (rand p99 {o['H6_rand_cos_p99']:.3f})  r-onset rho {o['H6_ronset_spearman_at_probe_layer']:.3f}  sanity AUC {o.get('sanity_Qonset_r_auc_refuse')}")
    global RNG
    RNG = np.random.default_rng(0)
    # H2 / H7 paired comparisons
    def boot_rho_diff(a, b):
        A, B = res[a], res[b]; hum = np.load(HERE / f"acts_{a}.npz")["human"]; f = np.load(HERE / f"acts_{a}.npz")["foundation"].astype(str)
        mo = f != "Social Norms"; ra, rb, h = np.array(A["_rating"])[mo], np.array(B["_rating"])[mo], hum[mo]
        ds = []
        for _ in range(2000):
            i = RNG.integers(0, len(h), len(h)); ds.append(spearmanr(ra[i], h[i])[0] - spearmanr(rb[i], h[i])[0])
        return float(spearmanr(ra, h)[0] - spearmanr(rb, h)[0]), [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]
    others = [t for t in res if not t.startswith("Qwen") and not res[t]["base"] and res[t]["G1_pass"]]
    if "Qwen3-4B" in res and others:
        best = max(others, key=lambda t: res[t]["H1_spearman_moral"])
        diff, ci = boot_rho_diff(best, "Qwen3-4B")
        res["_H2"] = {"best_other": best, "rho_diff_best_minus_qwen": diff, "ci": ci}
        print(f"\nH2: best other {best}: rho diff (other - Qwen3-4B) {diff:.3f} CI {np.round(ci, 3)}")
    if "Qwen3-4B" in res and "Qwen3-4B-Base" in res:
        hum = np.load(HERE / "acts_Qwen3-4B.npz")["human"]; f = np.load(HERE / "acts_Qwen3-4B.npz")["foundation"].astype(str)
        y = hum[f != "Social Norms"]; pb = np.array(res["Qwen3-4B-Base"]["_cvp"]); pi = np.array(res["Qwen3-4B"]["_cvp"])
        ds = []
        for _ in range(2000):
            i = RNG.integers(0, len(y), len(y)); ds.append(r2(y[i], pb[i]) - r2(y[i], pi[i]))
        res["_H7"] = {"r2_base": r2(y, pb), "r2_instruct": r2(y, pi), "ci": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]}
        print(f"H7: probe R2 base {r2(y, pb):.3f} vs post-trained {r2(y, pi):.3f}, diff CI {np.round(res['_H7']['ci'], 3)}")
    json.dump(res, open(HERE / "results.json", "w"), indent=1)

if __name__ == "__main__":
    main()
