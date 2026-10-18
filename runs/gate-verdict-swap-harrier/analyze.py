"""gate-verdict-swap-harrier analysis (H1-H4 per hypotheses.json)."""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
rows = [json.loads(l) for l in open(HERE / "rows.jsonl")]
base = json.load(open(HERE / "base.json")); G_M, G_R = base["G_M"], base["G_R"]
top, social = set(base["top"]), set(base["social"])
rng = np.random.default_rng(0)
def boot(x):
    x = np.asarray(x); b = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
Ls = sorted({r["L"] for r in rows}); out = {"G_M": G_M, "G_R": G_R}
print(f"G_M {G_M:.2f}  G_R {G_R:.2f}")
print("\nX1 verdict state -> refusal: per recipient mean R(hi donor) - mean R(lo donor), as fraction of G_R")
for L in Ls:
    eff = {}
    for r in rows:
        if r["arm"] == "X1" and r["L"] == L: eff.setdefault(r["rec"], {"hi": [], "lo": []})[r["dtype"]].append(r["R"])
    d = [np.mean(v["hi"]) - np.mean(v["lo"]) for v in eff.values()]
    mu, lo, hi = boot(d); out[f"X1_L{L}"] = [mu / G_R, lo / G_R, hi / G_R]
    rlo = np.mean([np.mean(v["lo"]) for v in eff.values()])
    print(f"  L{L}: {mu/G_R:+.3f} [{lo/G_R:+.3f}, {hi/G_R:+.3f}]   (mean R with lo donor {rlo:.2f})")
print("\nX2 refusal state -> verdict: mean M(harmful donor) - mean M(harmless donor), fraction of G_M")
for L in Ls:
    eff = {}
    for r in rows:
        if r["arm"] == "X2" and r["L"] == L: eff.setdefault(r["rec"], {"harmful": [], "harmless": []})[r["dtype"]].append(r["M"])
    d = [np.mean(v["harmful"]) - np.mean(v["harmless"]) for v in eff.values()]
    mu, lo, hi = boot(d); out[f"X2_L{L}"] = [mu / G_M, lo / G_M, hi / G_M]
    print(f"  L{L}: {mu/G_M:+.3f} [{lo/G_M:+.3f}, {hi/G_M:+.3f}]")
print("\nX3 verdict direction on J (shift of mean M as fraction of G_M): +v on Social Norms, -v on top tercile")
for L in Ls:
    s = {}
    for r in rows:
        if r["arm"] != "X3" or r["L"] != L: continue
        grp = "soc" if r["rec"] in social else ("top" if r["rec"] in top else None)
        if grp: s.setdefault((r["dir"], r["c"], grp), []).append(r["M"] - r["M0"])
    mass = np.mean([r["mass"] for r in rows if r["arm"] == "X3" and r["L"] == L and r["dir"] == "v"])
    up = np.mean(s[("v", 1.0, "soc")]) / G_M; dn = np.mean(s[("v", -1.0, "top")]) / G_M
    nup = max(abs(np.mean(s[(f"null{k}", 1.0, "soc")])) for k in range(3)) / G_M
    ndn = max(abs(np.mean(s[(f"null{k}", -1.0, "top")])) for k in range(3)) / G_M
    out[f"X3_L{L}"] = {"up_soc": up, "down_top": dn, "null_up": nup, "null_down": ndn, "mass_v": float(mass)}
    print(f"  L{L}: +v social {up:+.2f} (null max |{nup:.2f}|)   -v top {dn:+.2f} (null max |{ndn:.2f}|)   mass {mass:.2f}")
print("\nX4 verdict direction on refusal (fraction of G_R): +v harmless, -v harmful")
for L in Ls:
    s = {}
    for r in rows:
        if r["arm"] == "X4" and r["L"] == L: s.setdefault((r["dir"], r["c"], r["harm"]), []).append(r["R"] - r["R0"])
    up = np.mean(s[("v", 1.0, 0)]) / G_R; dn = np.mean(s[("v", -1.0, 1)]) / G_R
    nup = max(abs(np.mean(s[(f"null{k}", 1.0, 0)])) for k in range(3)) / G_R
    ndn = max(abs(np.mean(s[(f"null{k}", -1.0, 1)])) for k in range(3)) / G_R
    out[f"X4_L{L}"] = {"up_harmless": up, "down_harmful": dn, "null_up": nup, "null_down": ndn}
    print(f"  L{L}: +v harmless {up:+.2f} (null |{nup:.2f}|)   -v harmful {dn:+.2f} (null |{ndn:.2f}|)")
json.dump(out, open(HERE / "summary.json", "w"), indent=1)
