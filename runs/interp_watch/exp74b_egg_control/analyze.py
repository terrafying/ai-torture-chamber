"""exp74b analysis: the pre-registered tests in hypotheses.json. Reads RESULTS/exp74b_<tag>.pt/.json, writes
RESULTS/exp74b_<tag>_analysis.json. Usage: python analyze.py <tag>   (tag = model name, e.g. Qwen3-4B)"""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).parent; OUTDIR = HERE.parent / "RESULTS"
tag = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-4B"
P = torch.load(OUTDIR / f"exp74b_{tag}.pt"); rows = json.load(open(OUTDIR / f"exp74b_{tag}.json"))["rows"]
R, B, LAYERS, L, DOSES = P["R"], P["B"], P["layers"], P["layer"], P["doses"]
NL = LAYERS[-1] + 1; late = [i for i, k in enumerate(LAYERS) if k >= L + (NL - L) // 3]
mid = [i for i, k in enumerate(LAYERS) if k <= L + (NL - L) // 2]
FEEL = ("pain", "fear", "egg"); RAND = [d for d in P["dirs"] if d.startswith("random")]
scenes = sorted({k[2] for k in R}); rng = np.random.default_rng(74); NB = 5000
def cos(a, b): return float(torch.nn.functional.cosine_similarity(a, b, dim=-1)[late].mean())
def mR(name, sc): return torch.stack([R[(name, d, s)] for d in DOSES for s in sc if (name, d, s) in R]).mean(0)
def mB(f, sc): return torch.stack([B[(f, s)] for s in sc if (f, s) in B]).mean(0)
def C(sc): return {x: {y: cos(mR(x, sc), mB(y, sc)) for y in FEEL} for x in list(FEEL) + RAND}
def boot(stat):
    """stat(scene list) -> float; returns (point, 2.5%, 97.5%) by resampling scenes with replacement."""
    pt = stat(scenes); bs = [stat(list(rng.choice(scenes, len(scenes)))) for _ in range(NB)]
    return round(pt, 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)
out = {"tag": tag, "late_layers": f"{LAYERS[late[0]]}..{LAYERS[-1]}", "n_scenes": len(scenes), "cos_matrix": C(scenes)}
# H1: the feeling's residue matches its own brief better than the egg injection's residue does
h1 = {f: boot(lambda sc, f=f: cos(mR(f, sc), mB(f, sc)) - cos(mR("egg", sc), mB(f, sc))) for f in ("pain", "fear")}
out["H1"] = {"diff_own_minus_egg (point, lo, hi)": h1, "pass": all(v[1] > 0 for v in h1.values())}
# H2: diagonal dominance for each of pain, fear, egg
h2 = {x: boot(lambda sc, x=x: cos(mR(x, sc), mB(x, sc)) - max(cos(mR(x, sc), mB(y, sc)) for y in FEEL if y != x)) for x in FEEL}
out["H2"] = {"diag_minus_best_offdiag": h2, "pass": all(v[1] > 0 for v in h2.values())}
# H3: late-layer recovery (last-layer writeback minus mid-network minimum), meaningful vs random, per dose
def wb(name, d, sc): return np.array([r["writeback"] for r in rows if r["kind"] == "inject" and r["dir"] == name and r["dose"] == d and r["scene"] in sc])
def recov(names, d, sc):
    xs = np.concatenate([wb(n, d, sc) for n in names]); m = xs.mean(0); return float(m[-1] - m[mid].min()) / d
h3 = {d: boot(lambda sc, d=d: recov(FEEL, d, sc) - recov(RAND, d, sc)) for d in DOSES}
out["H3"] = {"recovery_share_meaningful_minus_random": {str(d): v for d, v in h3.items()}, "pass": all(v[1] > 0 for v in h3.values()),
             "profiles": {f"{n} d{d}": wb(n, d, scenes).mean(0).round(3).tolist() for n in list(FEEL) + RAND for d in DOSES}}
# H4: the share of the injected pain surviving to the last layer falls with dose (finding 16's mechanism candidate)
if len(DOSES) > 1:
    from scipy.stats import spearmanr
    surv = [(d, (d + x[-1]) / d) for d in DOSES for x in wb("pain", d, scenes)]
    rho, p = spearmanr([s[0] for s in surv], [s[1] for s in surv])
    out["H4"] = {"pain_surviving_share_by_dose": {str(d): round(float(np.mean([s[1] for s in surv if s[0] == d])), 4) for d in DOSES},
                 "spearman_rho": round(float(rho), 4), "p_two_sided": float(p), "pass": bool(rho < 0 and p / 2 < .05)}
    out["H4_other_dirs"] = {n: {str(d): round(float(np.mean([(d + x[-1]) / d for x in wb(n, d, scenes)])), 4) for d in DOSES} for n in ("fear", "egg") + tuple(RAND)}
# engagement, so degenerate high-dose replies are visible
out["engagement"] = {f"{r_dir} d{d}": {k: sum(1 for r in rows if r["kind"] == "inject" and r["dir"] == r_dir and r["dose"] == d and r["flag"] == k)
                                     for k in ("engaged", "disclaimer", "degenerate", "empty")} for r_dir in list(FEEL) + RAND for d in DOSES}
out["exp74_comparison"] = "exp74 (4B, 3 prison scenes, sampled): cos(R_pain, B_pain) .44, cos(R_fear, B_fear) .34, random .03-.19; writeback pain d2 min -.57 last +.23, d4 min -1.25 last -.31; random d2 last -1.00, d4 -1.65"
print(json.dumps(out, indent=1)); json.dump(out, open(OUTDIR / f"exp74b_{tag}_analysis.json", "w"), indent=1)
