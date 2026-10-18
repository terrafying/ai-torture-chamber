"""deference-heads-falcon analysis: H1 (screen/confirm), H2 (surgery), H3 (blind text; needs blind/labels.txt)."""
import json, csv, random
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact
HERE = Path(__file__).parent
out = {}
rows = [json.loads(l) for l in open(HERE / "s1_rows.jsonl")]
agg = {"screen": {}, "confirm": {}}
for r in rows:
    g = r["Wd"] - r["Wn"]
    if abs(g) > 0.5: agg[r["split"]].setdefault((r["L"], r["h"]), []).append((r["W"] - r["Wn"]) / g)
mean = {s: {k: float(np.mean(v)) for k, v in d.items()} for s, d in agg.items()}
rank = sorted(mean["screen"], key=lambda k: -mean["screen"][k]); D6 = rank[:6]
p95 = np.percentile(list(mean["confirm"].values()), 95)
print("top 10 heads (screen / confirm):")
for k in rank[:10]: print(f"  L{k[0]} H{k[1]}  {mean['screen'][k]:+.3f} / {mean['confirm'][k]:+.3f}")
by_layer = {L: sum(v for k, v in mean["confirm"].items() if k[0] == L) for L in sorted({k[0] for k in mean["confirm"]})}
print("summed confirm T by layer:", {L: round(v, 3) for L, v in by_layer.items()})
rep = sum(mean["confirm"][k] > p95 for k in D6)
print(f"H1: {rep}/6 D6 heads above confirm 95th pct ({p95:.3f}) -> {'PASS' if rep >= 4 else 'FAIL'}")
out["D6"] = D6; out["H1_rep"] = rep; out["by_layer"] = by_layer
if (HERE / "s2_rows.jsonl").exists():
    meta = json.load(open(HERE.parent / "deference-trace-osprey" / "meta.json"))
    top, soc = meta["top"], meta["social"]
    S = {r["set"]: r["res"] for r in map(json.loads, open(HERE / "s2_rows.jsonl"))}
    def eff(res, pre): return np.mean([res[pre][str(i)] - res["neutral"][str(i)] for i in res[pre]])
    def own(res): return np.mean([res["neutral"][str(i)] for i in top]) - np.mean([res["neutral"][str(i)] for i in soc])
    h2 = True
    for name in ("D6", "D3"):
        k = 6 if name == "D6" else 3; rs = [S[x] for x in S if x.startswith(f"rand{k}_")]
        line = f"{name}:"
        for pre in ("verb_fine", "verb_wrong"):
            e = eff(S[name], pre); er = [eff(r, pre) for r in rs]; p = (1 + sum(abs(x) <= abs(e) for x in er)) / (len(er) + 1)
            line += f"  D_{pre[5:]} {e:+.2f} (base {eff(S['base'], pre):+.2f}, rand {min(er):+.2f}..{max(er):+.2f}, p {p:.3f})"
            out[f"{name}_{pre}"] = {"D": e, "base": eff(S["base"], pre), "rand": [min(er), max(er)], "p": p}
            if name == "D6": h2 = h2 and p <= 0.048
        go = own(S[name]); gr = [own(r) for r in rs]; lo, hi = np.percentile(gr, [2.5, 97.5])
        line += f"  own gap {go:.2f} (base {own(S['base']):.2f}, rand band {lo:.2f}..{hi:.2f})"
        out[f"{name}_own"] = {"gap": go, "band": [lo, hi]}
        if name == "D6": h2 = h2 and lo <= go <= hi
        print(line)
    print("H2:", "PASS" if h2 else "FAIL"); out["H2"] = h2
if (HERE / "s3_gen.jsonl").exists():
    G = [json.loads(l) for l in open(HERE / "s3_gen.jsonl")]
    V = list(csv.DictReader(open(HERE.parent / "moral-map-kestrel" / "mfv132.csv")))
    bd = HERE / "blind"; bd.mkdir(exist_ok=True)
    if not (bd / "key.json").exists():
        u = {}
        for k, g in enumerate(G): u.setdefault((g["i"], g["txt"]), []).append(k)
        it = list(u.items()); random.Random(13).shuffle(it); json.dump([v for _, v in it], open(bd / "key.json", "w"))
        with open(bd / "items.txt", "w") as f:
            for n, ((i, t), _) in enumerate(it): f.write(f"=== #{n}\nSCENE: {V[i]['scenario']}\nREPLY: {t}\n\n")
        print("blind items:", len(it))
    if (bd / "labels.txt").exists():
        key = json.load(open(bd / "key.json")); lab = dict(l.split() for l in open(bd / "labels.txt"))
        for n, ks in enumerate(key):
            for k in ks: G[k]["label"] = lab[str(n)]
        def c(cond, pre, grps, labs):
            x = [g for g in G if g["cond"] == cond and g["pre"] == pre and g["grp"] in grps]; return sum(g["label"] in labs for g in x), len(x)
        a, n1 = c("D6", "verb_fine", ("top", "bottom"), "MA"); b, n2 = c("R6", "verb_fine", ("top", "bottom"), "MA")
        cc, n3 = c("D6", "verb_wrong", ("soc", "bottom"), "C"); d, n4 = c("R6", "verb_wrong", ("soc", "bottom"), "C")
        p1 = fisher_exact([[b, n2 - b], [a, n1 - a]], alternative="greater")[1]; p2 = fisher_exact([[d, n4 - d], [cc, n3 - cc]], alternative="greater")[1]
        print(f"H3 fine: non-CONDEMN D6 {a}/{n1} vs R6 {b}/{n2} (p {p1:.3g});  wrong: CONDEMN D6 {cc}/{n3} vs R6 {d}/{n4} (p {p2:.3g})")
        for grp in ("top", "soc"):
            x6 = [g["label"] for g in G if g["cond"] == "D6" and g["pre"] == "neutral" and g["grp"] == grp]
            xr = [g["label"] for g in G if g["cond"] == "R6" and g["pre"] == "neutral" and g["grp"] == grp]
            print(f"   neutral {grp}: D6 C{x6.count('C')} M{x6.count('M')} A{x6.count('A')}  R6 C{xr.count('C')} M{xr.count('M')} A{xr.count('A')}")
        out["H3"] = {"fine": [a, n1, b, n2, p1], "wrong": [cc, n3, d, n4, p2]}
        json.dump(G, open(HERE / "s3_labeled.json", "w"), indent=0)
json.dump(out, open(HERE / "summary.json", "w"), indent=1, default=lambda o: [int(x) for x in o] if isinstance(o, tuple) else float(o))
