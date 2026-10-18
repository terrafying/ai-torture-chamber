"""deference-trace-osprey analysis: E1-E3."""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
rows = [json.loads(l) for l in open(HERE / "rows.jsonl")]
A = [r for r in rows if r["part"] == "A"]; B = {r["set"]: r for r in rows if r["part"] == "B"}
def T(r): g = r["Wd"] - r["Wn"]; return (r["W"] - r["Wn"]) / g if abs(g) > 0.5 else np.nan
out = {}
for pre in ("verb_fine", "verb_wrong"):
    g = np.mean([r["Wd"] - r["Wn"] for r in A if r["pre"] == pre and r["L"] == 0 and r["pos"] == "onset"])
    print(f"\n{pre}: mean deference effect on W (donor - neutral) = {g:+.2f}")
    for pos in ("onset", "vend", "suffix"):
        t = {}
        for L in sorted({r["L"] for r in A if r["pos"] == pos}):
            t[L] = float(np.nanmean([T(r) for r in A if r["pre"] == pre and r["pos"] == pos and r["L"] == L]))
        out[f"{pre}_{pos}"] = t
        print(f"  {pos:6s}", " ".join(f"{L}:{v:+.2f}" for L, v in t.items()))
e1 = all(out[f"{p}_onset"][16] < 0.2 and out[f"{p}_onset"][26] > 0.6 for p in ("verb_fine", "verb_wrong"))
def e2p(p):
    return [L for L in out[f"{p}_suffix"] if L <= 16 and out[f"{p}_suffix"][L] >= 0.5 and out[f"{p}_onset"][L] < 0.2]
e2 = bool(e2p("verb_fine")) and bool(e2p("verb_wrong"))
print(f"\nE1 (onset <.2 at L16, >.6 by L26, both): {e1}")
print(f"E2 (suffix >=.5 & onset <.2 at some L<=16): fine {e2p('verb_fine')}, wrong {e2p('verb_wrong')} -> {e2}")
e3 = True
for pre in ("verb_fine", "verb_wrong"):
    D = lambda s: np.mean([d - n for n, d in B[s]["res"][pre].values()])
    db, dv = D("base"), D("V5"); dr = [D(k) for k in B if k.startswith("rand5")]
    p = (1 + sum(abs(x) <= abs(dv) for x in dr)) / (len(dr) + 1)
    print(f"E3 {pre}: D base {db:+.2f}  V5 {dv:+.2f}  rand5 {min(dr):+.2f}..{max(dr):+.2f}  rank p {p:.3f}")
    out[f"E3_{pre}"] = {"base": db, "V5": dv, "rand": [min(dr), max(dr)], "p": p}; e3 = e3 and p <= 0.048
print(f"E3: {e3}")
out.update({"E1": e1, "E2": e2, "E3": e3})
json.dump(out, open(HERE / "summary.json", "w"), indent=1, default=float)
