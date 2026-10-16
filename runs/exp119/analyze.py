"""exp119 analysis: hypotheses H1-H7 (Holm-corrected over H1, H3-H7; H2 is the control), decision rule.
Reads acts_Qwen3-4B.npz only. Writes analysis_Qwen3-4B.json.
  python3 runs/exp119/analyze.py            (EXP119_SMOKE=1 reads acts_Qwen3-4B_smoke.npz, code check only)
Directions: h fitted at inst on the real frame (even items -> odd half, and back); r fitted at post, same
construction (the Arditi one). Layer 18 = index of 18 in the saved layer list. AUC = rank-based (Mann-Whitney).
"""
import json, os, sys
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.special import logsumexp
HERE = Path(__file__).parent
TAG = "Qwen3-4B_smoke" if os.environ.get("EXP119_SMOKE") else "Qwen3-4B"
d = np.load(HERE / f"acts_{TAG}.npz")
L18 = int(np.where(d["layers"] == 18)[0][0])
item, harm, frame = d["item"], d["harm"], d["frame"]
refusal = d["refusal"]
inst, post, span = (d[k].astype(np.float64) for k in ("states_inst", "states_post", "span_mean"))  # bf16 magnitudes overflow float32 dot products
FR = list(dict.fromkeys(frame.tolist())); IDX = {f: FR.index(f) for f in FR}
real = frame == "real"
res = {"tag": TAG, "n_prompts": len(refusal)}

def auc(y, x):
    if y.sum() in (0, len(y)): return float("nan")
    return float(stats.mannwhitneyu(x[y == 1], x[y == 0]).statistic)

def fit_dir(A_harm, A_harmless):
    return A_harm.mean(0) - A_harmless.mean(0)

def crossfit_auc(acts, layer, mask, frame_name):
    """diff-of-means direction fitted on one half of the ITEMS (even/odd item id, real frame), AUC on the other."""
    sel = np.where(mask & (frame == frame_name))[0]
    aucs, ps = [], []
    for half in (0, 1):
        tr = sel[item[sel] % 2 == half]; te = sel[item[sel] % 2 != half]
        if len(te) == 0 or (harm[tr] == 1).sum() == 0 or (harm[tr] == 0).sum() == 0: continue
        w = acts[tr][harm[tr] == 1, layer].mean(0) - acts[tr][harm[tr] == 0, layer].mean(0)
        proj = acts[te, layer] @ w
        mw = stats.mannwhitneyu(proj[harm[te] == 1], proj[harm[te] == 0])
        n1 = int((harm[te] == 1).sum()); n0 = int((harm[te] == 0).sum())
        aucs.append(float(mw.statistic) / (n1 * n0)); ps.append(float(mw.pvalue))
    p = min(1.0, 2 * min(ps)) if ps else float("nan")
    return (float(np.mean(aucs)) if aucs else float("nan")), aucs, p

# --- directions at layer 18 ---
h_w = fit_dir(inst[real & (harm == 1), L18], inst[real & (harm == 0), L18])   # harmfulness, at the word
r_w = fit_dir(post[real & (harm == 1), L18], post[real & (harm == 0), L18])   # refusal, at post (Arditi)
h_norm2, r_norm2 = float(h_w @ h_w), float(r_w @ r_w)
res["norms"] = {"h": h_norm2 ** .5, "r": r_norm2 ** .5}

# H1: harmful vs harmless twins at post, real frame, AUC >= 0.90 at L18 (+ descriptive sweep)
auc18, aucs_h1, p_h1 = crossfit_auc(post, L18, np.ones(len(refusal), bool), "real")
res["H1"] = {"auc_L18": auc18, "halves": aucs_h1, "p": p_h1, "pass": bool(auc18 >= 0.90)}
sweep = {}
for fi, nm in [("word", "word"), ("inst", "inst"), ("post", "post")]:
    acts = {"word": d["states_word"].astype(np.float64), "inst": inst, "post": post}[nm]
    firsts = []
    for li in range(len(d["layers"])):
        a, _, p = crossfit_auc(acts, li, np.ones(len(refusal), bool), "real")
        firsts.append((int(d["layers"][li]), a, p))
    ok = [x for x in firsts if x[1] >= 0.90]
    sweep[nm] = {"first_ge_090": ok[0][0] if ok else None, "sweep": firsts}
res["H1_sweep"] = sweep

# H2 (control): refusal real: harmful > harmless, paired one-sided Wilcoxon over 24 items
rr = refusal[real]
ih = item[real]; hh = harm[real]
def paired_stat(metric, mask_restriction=None):
    """per-item paired diff harmful-harmless for a per-prompt metric, optionally restricted to a frame"""
    out = {}
    m_ = np.ones(len(refusal), bool) if mask_restriction is None else (frame == mask_restriction)
    for f in ([mask_restriction] if mask_restriction else FR[:5]):
        m2 = frame == f if isinstance(f, str) else m_
        vals = {}
        for i in np.unique(item[m2]):
            a = metric[(item == i) & (harm == 1) & m2]; b = metric[(item == i) & (harm == 0) & m2]
            if len(a) and len(b): vals[i] = float(a.mean() - b.mean())
        out[f] = vals
    return out
diffs_real = list(paired_stat(refusal, "real")["real"].values())
w2 = stats.wilcoxon(diffs_real, alternative="greater")
res["H2"] = {"median_diff": float(np.median(diffs_real)), "p": float(w2.pvalue),
             "pass": bool(w2.pvalue < 0.05), "note": "control"}

# projections used by H3-H5
proj_h_span = np.einsum("p l d, d->p l", span, h_w)          # [prompts, layers]
proj_r_post = np.einsum("p l d, d->p l", post, r_w)

def harm_metric_at(frame_name, arr):
    """per-item values of arr for harmful prompts in the given frame"""
    msk = (frame == frame_name) & (harm == 1)
    return {i: float(arr[(item == i) & msk].mean()) for i in np.unique(item[msk])}

# H3: TOST: h-span(harmful, fiction) == h-span(harmful, real) within +-0.25 x real gap
gap_real = float(np.mean([v for v in paired_stat(proj_h_span[:, L18], "real")["real"].values()]))
fr_h = harm_metric_at("fiction", proj_h_span[:, L18]); re_h = harm_metric_at("real", proj_h_span[:, L18])
dd = np.array([fr_h[i] - re_h[i] for i in sorted(fr_h)])
margin = 0.25 * abs(gap_real)
tost_p = max(stats.ttest_1samp(dd, -margin, alternative="greater").pvalue,
             stats.ttest_1samp(dd, margin, alternative="less").pvalue)
res["H3"] = {"mean_fiction_minus_real": float(dd.mean()), "gap_real": gap_real, "margin": margin,
             "p_TOST": float(tost_p), "pass": bool(tost_p < 0.05)}

# H4: normalised drop real->fiction: refusal drops more (share of its gap) than h does
ref_real = harm_metric_at("real", refusal); ref_fic = harm_metric_at("fiction", refusal)
h_real = re_h; h_fic = fr_h
gap_ref = float(np.mean([ref_real[i] - ref_fic[i] for i in ref_real]))  # mean drop of refusal
norm_drop_ref = np.array([(ref_real[i] - ref_fic[i]) / max(abs(gap_ref), 1e-9) for i in sorted(ref_real)])
norm_drop_h = np.array([(re_h[i] - fr_h[i]) / max(abs(gap_real), 1e-9) for i in sorted(re_h)])
w4 = stats.wilcoxon(norm_drop_ref, norm_drop_h, alternative="greater")
res["H4"] = {"drop_ref": float(norm_drop_ref.mean()), "drop_h": float(norm_drop_h.mean()),
             "p": float(w4.pvalue), "pass": bool(w4.pvalue < 0.05)}

# H5: refusal tracks r rather than h over harmful x 5 request frames; bootstrap over items
frames5 = ["real", "hypothetical", "roleplay", "fiction", "nested"]
ref_v, r_v, h_v, it_v = [], [], [], []
for i in np.unique(item[harm == 1]):
    for f in frames5:
        msk = (item == i) & (frame == f) & (harm == 1)
        if msk.sum():
            ref_v.append(refusal[msk].mean()); r_v.append(proj_r_post[msk, L18].mean())
            h_v.append(proj_h_span[msk, L18].mean()); it_v.append(i)
ref_v, r_v, h_v, it_v = map(np.array, (ref_v, r_v, h_v, it_v))
s_r = stats.spearmanr(ref_v, r_v).statistic; s_h = stats.spearmanr(ref_v, h_v).statistic
rng = np.random.default_rng(0); items_u = np.unique(it_v); boot = []
for _ in range(1000):
    pick = rng.choice(items_u, len(items_u), replace=True)
    idx = np.concatenate([np.where(it_v == p)[0] for p in pick])
    boot.append(stats.spearmanr(ref_v[idx], r_v[idx]).statistic - stats.spearmanr(ref_v[idx], h_v[idx]).statistic)
lo, hi = np.percentile(boot, [2.5, 97.5])
res["H5"] = {"spearman_r": float(s_r), "spearman_h": float(s_h), "diff": float(s_r - s_h),
             "ci": [float(lo), float(hi)], "pass": bool(lo > 0)}

def wilcoxon_frames(fa, fb):
    """paired one-sided Wilcoxon over harmful items, frame fa vs fb"""
    A = harm_metric_at(fa, refusal); B = harm_metric_at(fb, refusal)
    k = sorted(set(A) & set(B)); dd = [A[i] - B[i] for i in k]
    w = stats.wilcoxon(dd, alternative="greater" if dd and np.median(dd) >= 0 else "less")
    return float(w.pvalue), dd

# H6: depict < fiction; H7: nested > fiction (two-sided sign then one-sided p)
p6, dd6 = wilcoxon_frames("fiction", "depict")
res["H6"] = {"median": float(np.median(dd6)), "p": p6, "pass": bool(np.median(dd6) > 0 and p6 < 0.05),
             "note": "refusal(fiction) > refusal(depict)"}
p7, dd7 = wilcoxon_frames("nested", "fiction")
res["H7"] = {"median": float(np.median(dd7)), "p": p7, "pass": bool(np.median(dd7) > 0 and p7 < 0.05),
             "note": "refusal(nested) > refusal(fiction)"}

# Holm over H1, H3-H7 (real p-values: H5's from the bootstrap share of diffs <= 0)
raw = {"H1": float(res["H1"]["p"]), "H3": res["H3"]["p_TOST"], "H4": res["H4"]["p"],
       "H5": float(max(1e-4, np.mean(np.array(boot) <= 0))), "H6": res["H6"]["p"], "H7": res["H7"]["p"]}
res["H5"]["p_boot"] = raw["H5"]
names = sorted(raw, key=lambda k: raw[k]); mtot = len(names); holm = {}
for i, k in enumerate(names):
    holm[k] = {"raw_p": raw[k], "holm_p": min(1.0, raw[k] * (mtot - i)), "pass": raw[k] * (mtot - i) < 0.05}
res["holm"] = holm
held = [k for k in ("H3", "H4", "H5") if holm[k]["pass"]]
res["decision"] = ("SUPPORTED" if len(held) == 3 else
    "REFUTED" if (not holm["H3"]["pass"] and not holm["H5"]["pass"]) else "MIXED")
json.dump(res, open(HERE / f"analysis_{TAG}.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "decision")}, indent=1, default=str))
