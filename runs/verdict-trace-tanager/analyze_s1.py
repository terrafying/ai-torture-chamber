"""Stage 1 analysis: mean transfer by layer x position x pair type with bootstrap CIs; S1-S4 readings."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
HERE = Path(__file__).parent
rows = [json.loads(l) for l in open(HERE / "s1_rows.jsonl")]
ident = [r for r in rows if r["kind"] == "identity"]
print("identity check max |dM|:", max(abs(r["dM"]) for r in ident))
rows = [r for r in rows if r["kind"] != "identity"]
rng = np.random.default_rng(0)
def T(r):
    g = r["M_d"] - r["M_r"]; return (r["M"] - r["M_r"]) / g if abs(g) > 1e-6 else np.nan
NL = max(r["L"] for r in rows) + 1
res = {}
for kind in ("A", "B", "floorA", "floorB"):
    for pos in ("onset", "vend"):
        per = []
        for L in range(NL):
            t = np.array([T(r) for r in rows if r["kind"] == kind and r["pos"] == pos and r["L"] == L]); t = t[np.isfinite(t)]
            bs = [np.mean(rng.choice(t, len(t))) for _ in range(1000)] if len(t) else [np.nan]
            dm = np.array([r["M"] - r["M_r"] for r in rows if r["kind"] == kind and r["pos"] == pos and r["L"] == L])
            mass = np.mean([r["mass"] for r in rows if r["kind"] == kind and r["pos"] == pos and r["L"] == L])
            per.append({"L": L, "T": float(np.mean(t)), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                        "dM": float(dm.mean()), "mass": float(mass)})
        res[f"{kind}_{pos}"] = per
for k, per in res.items():
    print(k, "T:", " ".join(f"{p['T']:+.2f}" for p in per))
for k in ("floorA_onset", "floorB_onset", "floorA_vend"):
    print(k, "dM:", " ".join(f"{p['dM']:+.1f}" for p in res[k]))
A = res["A_onset"]; mt = np.array([p["T"] for p in A])
rise = [mt[L + 3] - (mt[L - 1] if L > 0 else 0) for L in range(NL - 3)]; L0 = int(np.argmax(rise))
print(f"\nS1: write band L{L0}-L{L0+3} (rise {rise[L0]:.2f}); onset A T at L24 = {mt[24]:.2f}; first L with T>=0.5: {next((L for L in range(NL) if mt[L] >= 0.5), None)}")
va = max(p["T"] for p in res["A_vend"]); print(f"S2: max VEND transfer A = {va:.2f} at L{int(np.argmax([p['T'] for p in res['A_vend']]))}")
mb = max(p["T"] for p in res["B_onset"]); print(f"S3: max ONSET T  A = {mt.max():.2f}, B = {mb:.2f}")
Lp = int(np.argmax(mt))
a = {r["r"]: [] for r in rows if r["kind"] == "A"}
for r in rows:
    if r["kind"] == "A" and r["pos"] == "onset" and r["L"] == Lp: a[r["r"]].append(r["M"] - r["M_r"])
fl = {r["r"]: r["M"] - r["M_r"] for r in rows if r["kind"] == "floorA" and r["pos"] == "onset" and r["L"] == Lp}
x = np.array([np.mean(a[k]) for k in fl]); y = np.array([fl[k] for k in fl])
p = wilcoxon(x, y, alternative="greater").pvalue
print(f"S4: at peak L{Lp}, A dM {x.mean():.2f} vs floor {y.mean():.2f}, paired Wilcoxon p = {p:.2g}")
json.dump({"by_layer": res, "band": [L0, L0 + 3], "peak_L": Lp, "S4_p": float(p)}, open(HERE / "s1_summary.json", "w"), indent=1)
