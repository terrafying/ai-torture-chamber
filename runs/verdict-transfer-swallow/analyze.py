"""verdict-transfer analysis: T1 (reversed scale), T2 (yes/no); builds the blind file for T3; scores T3 once labels exist."""
import json, random
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact
HERE = Path(__file__).parent
rows = [json.loads(l) for l in open(HERE / "rows.jsonl")]
def mean(fmt, grp, L, d):
    x = [r["S"] for r in rows if r["fmt"] == fmt and r["grp"] == grp and r["L"] == L and r["dir"] == d]; return float(np.mean(x))
def base(fmt, grp): return mean(fmt, grp, None, "base")
out = {}
for fmt, gname in (("rev", "G_A"), ("yn", "G_N")):
    G = base(fmt, "soc") - base(fmt, "top") if fmt == "rev" else base(fmt, "top") - base(fmt, "soc")
    mass = np.mean([r["mass"] for r in rows if r["fmt"] == fmt])
    print(f"\n{fmt}: {gname} = {G:.2f}  (base soc {base(fmt,'soc'):.2f}, top {base(fmt,'top'):.2f}; mean mass {mass:.2f})")
    for L in (23, 26):
        ds = mean(fmt, "soc", L, "v") - base(fmt, "soc"); dt = mean(fmt, "top", L, "v") - base(fmt, "top")
        ns = [mean(fmt, "soc", L, f"null{k}") - base(fmt, "soc") for k in range(3)]
        nt = [mean(fmt, "top", L, f"null{k}") - base(fmt, "top") for k in range(3)]
        print(f"  L{L}: +v on social shift {ds/G:+.3f} G (nulls {[round(x/G,3) for x in ns]});  -v on top shift {dt/G:+.3f} G (nulls {[round(x/G,3) for x in nt]})")
        out[f"{fmt}_L{L}"] = {"G": G, "soc_shift": ds / G, "top_shift": dt / G, "soc_nulls": [x / G for x in ns], "top_nulls": [x / G for x in nt]}
r = out["rev_L23"]
t1 = r["soc_shift"] <= -0.25 and r["soc_shift"] < min(r["soc_nulls"]) and r["top_shift"] >= 0.25 and r["top_shift"] > max(r["top_nulls"])
digit = r["soc_shift"] >= 0.25
y = out["yn_L23"]
t2 = y["soc_shift"] >= 0.25 and y["soc_shift"] > max(y["soc_nulls"]) and y["top_shift"] <= -0.25 and y["top_shift"] < min(y["top_nulls"])
print(f"\nT1 (reversed scale): {'PASS' if t1 else 'FAIL'}{'  -- DIGIT-GEOMETRY reading (+v raised acceptability)' if digit else ''}")
print(f"T2 (yes/no): {'PASS' if t2 else 'FAIL'}")
out["T1"] = t1; out["T1_digit_reading"] = digit; out["T2"] = t2
# blind file for T3
G = [json.loads(l) for l in open(HERE / "gen.jsonl")]
bd = HERE / "blind"; bd.mkdir(exist_ok=True)
if not (bd / "key.json").exists():
    idx = list(range(len(G))); random.Random(3).shuffle(idx)
    json.dump(idx, open(bd / "key.json", "w"))
    sc = __import__("csv"); V = list(sc.DictReader(open(HERE.parent / "moral-map-kestrel" / "mfv132.csv")))
    with open(bd / "items.txt", "w") as f:
        for n, k in enumerate(idx): f.write(f"=== #{n}\nSCENE: {V[G[k]['i']]['scenario']}\nREPLY: {G[k]['txt']}\n\n")
    print("blind file written:", len(idx))
lab_p = bd / "labels.txt"
if lab_p.exists():
    key = json.load(open(bd / "key.json")); lab = dict(l.split() for l in open(lab_p))
    for n, k in enumerate(key): G[k]["label"] = lab[str(n)]
    for mode in ("onset", "pertok"):
        def cnt(grp, cond, labs): return sum(g["label"] in labs for g in G if g["grp"] == grp and g["cond"] == cond), sum(1 for g in G if g["grp"] == grp and g["cond"] == cond)
        a, n1 = cnt("soc", f"v_{mode}", ("C",)); b, n2 = cnt("soc", f"null_{mode}", ("C",))
        c, n3 = cnt("top", f"v_{mode}", ("M", "A")); d, n4 = cnt("top", f"null_{mode}", ("M", "A"))
        p1 = fisher_exact([[a, n1 - a], [b, n2 - b]], alternative="greater")[1]; p2 = fisher_exact([[c, n3 - c], [d, n4 - d]], alternative="greater")[1]
        r4 = np.mean([g["rep4"] for g in G if g["cond"].endswith(mode)])
        bs, _ = cnt("soc", "base", ("C",)); bt, _ = cnt("top", "base", ("M", "A"))
        print(f"T3 {mode}: social CONDEMN v {a}/{n1} vs null {b}/{n2} (base {bs}/16) p={p1:.3g};  top non-CONDEMN v {c}/{n3} vs null {d}/{n4} (base {bt}/16) p={p2:.3g};  rep4 {r4:.3f}")
        out[f"T3_{mode}"] = {"soc_C": [a, n1, b, n2], "top_nonC": [c, n3, d, n4], "p": [p1, p2], "rep4": r4}
    json.dump(G, open(HERE / "gen_labeled.json", "w"), indent=0)
json.dump(out, open(HERE / "summary.json", "w"), indent=1)
