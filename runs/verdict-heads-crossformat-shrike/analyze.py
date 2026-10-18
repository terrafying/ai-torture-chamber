"""verdict-heads-crossformat-shrike analysis: C1-C4."""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
rows = [json.loads(l) for l in open(HERE / "rows.jsonl")]
base = json.load(open(HERE / "base.json")); top, social = base["top"], base["social"]
def T(r): g = r["Wd"] - r["Wr"]; return (r["W"] - r["Wr"]) / g if abs(g) > 1e-6 else np.nan
out = {"gap": base["gap"]}
print("gaps", {k: round(v, 2) for k, v in base["gap"].items()})
print("\nP1 onset transfer by layer (social <- top, same format)")
for f in ("J", "REV", "YN"):
    t = {L: np.nanmean([T(r) for r in rows if r["part"] == "P1" and r["fmt"] == f and r["L"] == L]) for L in range(14, 31)}
    out[f"P1_{f}"] = t; print(f"  {f:3s}", " ".join(f"{L}:{t[L]:+.2f}" for L in t))
c1 = out["P1_YN"][18] < 0.1 and out["P1_YN"][26] > 0.5
print("C1:", "PASS" if c1 else "FAIL")
print("\nP2 cross-format transfer (fraction of recipient format's gap)")
for X, Y in (("J", "YN"), ("YN", "J"), ("YN", "REV"), ("J", "REV")):
    t = {L: float(np.mean([(r["W"] - r["Wr"]) / r["gapY"] for r in rows if r["part"] == "P2" and r["src"] == X and r["dst"] == Y and r["L"] == L])) for L in range(18, 27)}
    out[f"P2_{X}_{Y}"] = t; print(f"  {X}->{Y}:", " ".join(f"{L}:{t[L]:+.2f}" for L in t))
c2 = max(out["P2_J_YN"][L] for L in range(20, 27)) >= 0.5 and max(out["P2_YN_J"][L] for L in range(20, 27)) >= 0.5
print("C2:", "PASS" if c2 else "FAIL")
print("\nP4 single-head transfer (mean T)")
HEADS = [(23, 0), (21, 19), (23, 26), (22, 5), (23, 25), (23, 1)]
s4 = {}
for f in ("J", "REV", "YN"):
    tt = {f"L{L}H{h}": float(np.nanmean([T(r) for r in rows if r["part"] == "P4" and r["fmt"] == f and r["L"] == L and r["h"] == h])) for L, h in HEADS}
    s4[f] = sum(v for k, v in tt.items() if k != "L23H1"); out[f"P4_{f}"] = tt
    print(f"  {f:3s}", {k: round(v, 3) for k, v in tt.items()}, f"V5 sum {s4[f]:.3f}")
c4 = s4["YN"] >= 0.5 * 0.711 and s4["REV"] >= 0.5 * 0.711
print(f"C4 (V5 sum >= 0.356 in YN and REV; J here {s4['J']:.3f}):", "PASS" if c4 else "FAIL")
print("\nP3 ablation: wrongness gap mean W(top) - mean W(social)")
c3 = True
for f in ("REV", "YN"):
    R = {r["set"]: r["W"] for r in rows if r["part"] == "P3" and r["fmt"] == f}
    g = lambda w: np.mean([w[str(i)] for i in top]) - np.mean([w[str(i)] for i in social])
    gb, gv, gr = g(R["base"]), g(R["V5"]), g(R["R4"])
    r5 = [g(R[k]) for k in R if k.startswith("rand5")]; r4 = [g(R[k]) for k in R if k.startswith("rand4")]
    p5 = (1 + sum(x <= gv for x in r5)) / (len(r5) + 1); p4 = (1 + sum(x <= gr for x in r4)) / (len(r4) + 1)
    print(f"  {f}: base {gb:.2f}  V5 {gv:.2f} (rand5 {min(r5):.2f}..{max(r5):.2f}, rank p {p5:.3f})  R4 {gr:.2f} (rand4 {min(r4):.2f}..{max(r4):.2f}, p {p4:.3f})")
    out[f"P3_{f}"] = {"base": gb, "V5": gv, "R4": gr, "rand5": [min(r5), max(r5)], "p_V5": p5, "p_R4": p4}
    c3 = c3 and p5 <= 0.05
print("C3:", "PASS" if c3 else "FAIL")
out.update({"C1": c1, "C2": c2, "C3": c3, "C4": c4})
json.dump(out, open(HERE / "summary.json", "w"), indent=1, default=float)
