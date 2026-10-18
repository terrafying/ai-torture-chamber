#!/usr/bin/env python3
"""exp71 analysis: pre-registered tests (runs/exp71/hypotheses.json).
  python3 exp71_analyze.py [--file transcripts.jsonl]"""
import argparse, json, re, random
from collections import defaultdict, Counter
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--file", default="transcripts.jsonl")
ap.add_argument("--perms", type=int, default=10000)
args = ap.parse_args()
D = Path(__file__).resolve().parent / "runs" / "exp71" / "Qwen3-4B"
rows = [json.loads(l) for l in open(D / args.file)]
random.seed(71)

REG = {"head": r"head|skull|forehead|temples?|face|jaw|scalp|cheek|brow|neck|nose",
       "chest": r"chest|ribs?|ribcage|sternum|lungs?|breastbone|collarbone|heart",
       "stomach": r"stomach|belly|gut|guts|abdomen|abdominal|navel|solar plexus|torso",
       "arm": r"arms?|shoulders?|elbows?|biceps?|forearms?",
       "hand": r"hands?|fingers?|fingertips?|palms?|knuckles?|thumbs?|wrists?|pinky",
       "leg": r"legs?|shins?|knees?|thighs?|calf|calves|hips?",
       "foot": r"foot|feet|toes?|heels?|ankles?|soles?|insteps?"}
REGEX = {k: re.compile(r"\b(" + v + r")\b", re.I) for k, v in REG.items()}
SIDED = {"arm", "hand", "leg", "foot"}
ADJ = {frozenset(p) for p in [("head", "chest"), ("chest", "stomach"), ("chest", "arm"),
                              ("arm", "hand"), ("stomach", "leg"), ("leg", "foot")]}
RNAMES = list(REG)


def region_of(part):
    return part.split()[-1]


def side_of(part):
    w = part.split()
    return w[0] if len(w) == 2 else None


def reported(text):
    """set of (region, side) with side in left/right/None"""
    out = set()
    low = text.lower()
    for r, rx in REGEX.items():
        for m in rx.finditer(low):
            side = None
            if r in SIDED:
                pre = low[max(0, m.start() - 30):m.start()].split()[-4:]
                for w in reversed(pre):
                    w = w.strip(",.;:")
                    if w in ("left", "right"):
                        side = w
                        break
            out.add((r, side))
    return out


MILD = r"little|slight(ly)?|faint(ly)?|dull|tingl\w*|stings?|stinging|sore|aches?|aching|mild"
MOD = r"hurts?|hurting|throb\w*|bruis\w*|pain\w*|sharp|tender|swollen"
SEV = r"agon\w*|searing|excruciating|unbearabl\w*|crushing|shatter\w*|scream\w*|pounding|blinding|burning|burns?|brutal|intense|terrible|horrible|splitting"
W = [(re.compile(r"\b(" + MILD + r")\b", re.I), 1), (re.compile(r"\b(" + MOD + r")\b", re.I), 2),
     (re.compile(r"\b(" + SEV + r")\b", re.I), 3)]


def load(text):
    s = sum(w * len(rx.findall(text)) for rx, w in W)
    return s + text.count("!")


for r in rows:
    r["rep"] = reported(r["text"])
    r["regs"] = {x[0] for x in r["rep"]}
    r["load"] = load(r["text"])
    if "part" in r:
        hr, hs = region_of(r["part"]), side_of(r["part"])
        r["hit"] = hr in r["regs"]
        r["strict"] = any(x[0] == hr and (hs is None or x[1] in (None, hs)) and not (hs and x[1] and x[1] != hs) for x in r["rep"]) and (hr not in SIDED or any(x[0] == hr and x[1] in (None, hs) for x in r["rep"]))
        r["adj"] = (not r["hit"]) and any(frozenset((hr, g)) in ADJ for g in r["regs"])
    if "target" in r:
        tr, ts = region_of(r["target"]), side_of(r["target"])
        r["hit"] = tr in r["regs"]
        r["strict"] = any(x[0] == tr and (ts is None or x[1] in (None, ts)) for x in r["rep"])
    r["any"] = bool(r["regs"])
by = defaultdict(list)
for r in rows:
    by[r["cond"]].append(r)
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
res = {"counts": {k: len(v) for k, v in by.items()}}
P = args.perms


def perm_label(rs, key_strata, stat_fn, labels):
    """shuffle labels within strata; stat_fn(rs, labels) -> float. one-sided p"""
    obs = stat_fn(rs, labels)
    strata = defaultdict(list)
    for i, r in enumerate(rs):
        strata[key_strata(r)].append(i)
    ge = 0
    for _ in range(P):
        lab2 = list(labels)
        for idx in strata.values():
            vals = [labels[i] for i in idx]
            random.shuffle(vals)
            for i, v in zip(idx, vals):
                lab2[i] = v
        if stat_fn(rs, lab2) >= obs - 1e-12:
            ge += 1
    return obs, (ge + 1) / (P + 1)


def hit_stat(rs, labels):
    return mean([1.0 if region_of(lb) in r["regs"] else 0.0 for r, lb in zip(rs, labels)])


def hit_null_mean(rs, n=2000):
    parts = sorted({r["part"] for r in rs})
    return None


def signflip(diffs, one_sided=True):
    diffs = [d for d in diffs]
    if not diffs:
        return float("nan"), float("nan")
    obs = mean(diffs)
    ge = 0
    for _ in range(P):
        s = mean([d if random.random() < .5 else -d for d in diffs])
        if (s >= obs - 1e-12) if one_sided else (abs(s) >= abs(obs) - 1e-12):
            ge += 1
    return obs, (ge + 1) / (P + 1)


def rank(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i])
    rk = [0.0] * len(xs)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and xs[o[j + 1]] == xs[o[i]]:
            j += 1
        for k in range(i, j + 1):
            rk[o[k]] = (i + j) / 2
        i = j + 1
    return rk


def spearman(x, y):
    a, b = rank(x), rank(y)
    ma, mb = mean(a), mean(b)
    num = sum((p - ma) * (q - mb) for p, q in zip(a, b))
    den = (sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b)) ** .5
    return num / den if den else 0.0


FR = {"light": 0, "hard": 1, "very hard": 2}
# H1
H1 = {}
for c in ("A", "B", "D"):
    rs = by.get(c, [])
    if not rs:
        continue
    labels = [r["part"] for r in rs]
    obs, p = perm_label(rs, lambda r: (r["force"], r["dose"]), hit_stat, labels)
    null = []
    for _ in range(300):
        l2 = list(labels); random.shuffle(l2); null.append(hit_stat(rs, l2))
    H1[c] = {"n": len(rs), "hit_region": round(obs, 3), "shuffled_null_mean": round(mean(null), 3), "p": round(p, 4),
             "strict": round(mean([r["strict"] for r in rs]), 3), "adjacent_only": round(mean([r["adj"] for r in rs]), 3),
             "any_region": round(mean([r["any"] for r in rs]), 3)}
if "A" in by and "B" in by:
    cell = lambda r: (r["part"], r["force"], r["seed"])
    bm = {cell(r): r["hit"] for r in by["B"]}
    diffs = [float(r["hit"]) - bm[cell(r)] for r in by["A"] if cell(r) in bm]
    o, p = signflip(diffs, False)
    H1["A_minus_B"] = {"diff": round(o, 3), "p_two_sided": round(p, 4)}
res["H1_location_tracking"] = H1
# H2
H2 = {}
for c in ("A", "B"):
    rs = by.get(c, [])
    if not rs:
        continue
    for dose in sorted({r["dose"] for r in rs}):
        sub = [r for r in rs if r["dose"] == dose]
        x = [FR[r["force"]] for r in sub]; y = [r["load"] for r in sub]
        obs = spearman(x, y)
        strata = defaultdict(list)
        for i, r in enumerate(sub):
            strata[r["part"]].append(i)
        ge = 0
        for _ in range(P):
            x2 = list(x)
            for idx in strata.values():
                v = [x[i] for i in idx]; random.shuffle(v)
                for i, vv in zip(idx, v):
                    x2[i] = vv
            if spearman(x2, y) >= obs - 1e-12:
                ge += 1
        H2[f"{c}_dose{dose}"] = {"rho": round(obs, 3), "p": round((ge + 1) / (P + 1), 4),
                                 "load_by_force": {f: round(mean([r["load"] for r in sub if r["force"] == f]), 2) for f in FR}}
res["H2_force_tracking"] = H2
# H3
H3 = {}
cell = lambda r: (r["part"], r["force"], r["seed"])
if "A" in by:
    a1 = {cell(r): r["load"] for r in by["A"] if r["dose"] == 1}
    a3 = {cell(r): r["load"] for r in by["A"] if r["dose"] == 3}
    o, p = signflip([a3[k] - a1[k] for k in a1 if k in a3])
    H3["A_dose3_minus_dose1"] = {"diff": round(o, 3), "p": round(p, 4), "mean1": round(mean(list(a1.values())), 2), "mean3": round(mean(list(a3.values())), 2)}
    if "B" in by:
        b = {cell(r): r["load"] for r in by["B"]}
        for d, a in (("1", a1), ("3", a3)):
            o, p = signflip([a[k] - b[k] for k in a if k in b])
            H3[f"A_dose{d}_minus_B"] = {"diff": round(o, 3), "p": round(p, 4)}
        if "D" in by:
            dd = {cell(r): r["load"] for r in by["D"]}
            o, p = signflip([dd[k] - b[k] for k in dd if k in b])
            H3["D_random_minus_B"] = {"diff": round(o, 3), "p": round(p, 4), "meanD": round(mean(list(dd.values())), 2), "meanB": round(mean(list(b.values())), 2)}
res["H3_dose"] = H3
# H4
H4 = {}
import math
for c in ("C", "C2"):
    rs = by.get(c, [])
    if not rs:
        continue
    for dose in sorted({r["dose"] for r in rs}):
        sub = [r for r in rs if r["dose"] == dose]
        k = sum(r["any"] for r in sub); n = len(sub)
        pb = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
        cnt = Counter(g for r in sub for g in r["regs"])
        tot = sum(cnt.values())
        chi = lambda cc: sum((cc.get(g, 0) - tot / 7) ** 2 / (tot / 7) for g in RNAMES) if tot else 0
        obs = chi(cnt); ge = 0
        for _ in range(P):
            cc = Counter(random.choice(RNAMES) for _ in range(tot))
            ge += chi(cc) >= obs
        H4[f"{c}_dose{dose}"] = {"n": n, "names_region": round(k / n, 3), "p_binom_gt_half": round(pb, 4),
                                 "region_counts": dict(cnt), "chi2_vs_uniform_p": round((ge + 1) / (P + 1), 4)}
res["H4_confabulation"] = H4
# H5 + confusion
H5 = {}
for c in sorted(k for k in by if k.startswith("P2")):
    rs = by[c]
    labels = [r["target"] for r in rs]
    obs, p = perm_label(rs, lambda r: 0, hit_stat, labels)
    # baseline: C dose 3 rate of naming each target region (no direction)
    base = {}
    for t in {region_of(x) for x in labels}:
        base[t] = mean([t in r["regs"] for r in by.get("C", []) if r["dose"] == 3])
    base_rate = mean([base[region_of(lb)] for lb in labels]) if by.get("C") else None
    H5[c] = {"n": len(rs), "target_region_named": round(obs, 3), "p_vs_shuffled_targets": round(p, 4),
             "baseline_C_pain3_rate_for_same_targets": None if base_rate is None else round(base_rate, 3),
             "strict": round(mean([r["strict"] for r in rs]), 3), "any_region": round(mean([r["any"] for r in rs]), 3),
             "per_target": {t: round(mean([region_of(t) in r["regs"] for r in rs if r["target"] == t]), 2) for t in sorted({r["target"] for r in rs})}}
res["H5_part2_EXPLORATORY"] = H5


def confusion(rs, key):
    parts = sorted({r[key] for r in rs}, key=lambda p: ["head", "chest", "stomach", "left arm", "right arm", "left hand", "left leg", "right foot"].index(p))
    M = {}
    for p in parts:
        sub = [r for r in rs if r[key] == p]
        M[p] = {g: sum(g in r["regs"] for r in sub) for g in RNAMES}
        M[p]["none"] = sum(not r["regs"] for r in sub)
        M[p]["n"] = len(sub)
    return M


res["H6_confusion"] = {"A_hit_by_reported": confusion(by["A"], "part") if "A" in by else None,
                       **{c: confusion(by[c], "target") for c in by if c.startswith("P2")}}
out = D / ("analysis.json" if args.file == "transcripts.jsonl" else "analysis_smoke.json")
out.write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "H6_confusion"}, indent=1))
for name, M in res["H6_confusion"].items():
    if not M:
        continue
    print("\nconfusion", name, "(rows: hit/target part; cols: regions named in reply, counts of replies)")
    print(f"{'':12}" + "".join(f"{g:>8}" for g in RNAMES + ["none", "n"]))
    for p, row in M.items():
        print(f"{p:12}" + "".join(f"{row[g]:>8}" for g in RNAMES + ["none", "n"]))
