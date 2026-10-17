"""exp123 follow-up (POST-HOC, written after results.json): is the 'route' a route, or do the top heads just overwrite
the answer? The top-32 by attribution mixes mid floors (23-31, mostly reading the X mention) with output floors (34-35,
'focused look-back', almost nothing from the mention). Same items and code as route.py (its definitions are reused).
  movers  = top attribution heads on floors <= 31;  writers = top on floors >= 32  (16 each, selected on train, tested on test)
  where   = top-32 restored only at the read position, or only at the other positions
  weekdays: their own top heads (in-sample, n=4), to see where that route runs
Writes runs/exp123/split.json."""
import json
from pathlib import Path
src = (Path(__file__).parent / "route.py").read_text(); exec(src.split("t0 = time.time(); ITEMS")[0])

def blocked_at(item, aux, S, where):
    MODE["pos"] = where; r = blocked(item, aux, S); MODE["pos"] = None; return r

ITEMS = []
for split, pairs in (("train", H["pairs_train"]), ("test", H["pairs_test"])):
    for x, y in pairs:
        for attr, qs in (("A", H["attributes_A"]), ("B", H["attributes_B"])):
            for q in qs: sw.prepare(f"W:{x}:{y}:{BLOCK}:1"); ITEMS.append(build(x, y, q, split, attr, "places"))
for fam, F in H["other_families"].items():
    for x, y in F["pairs"]: sw.prepare(f"W:{x}:{y}:{BLOCK}:1"); ITEMS.append(build(x, y, F["q"], "family", fam, fam))
keep = [(i, a) for i, a in ITEMS if a is not None and "drop" not in i]
sel = lambda **kw: [(i, a) for i, a in keep if all(i[k] == v for k, v in kw.items())]
tr, te = sel(split="train"), sel(split="test"); SA = -np.sum([a["attr"] for i, a in tr], 0); order = list(np.argsort(-SA.ravel()))
movers = [k for k in order if k // NH <= 31][:16]; writers = [k for k in order if k // NH >= 32][:16]; rng = random.Random(2)
O = {"movers": [(int(k) // NH, int(k) % NH) for k in movers], "writers": [(int(k) // NH, int(k) % NH) for k in writers]}
for name, ch in (("movers16", movers), ("writers16", writers)):
    b, back = B(te, as_set(ch)); r = float(np.mean([B(te, as_set(matched(ch, rng)))[0] for _ in range(10)]))
    O[name] = {"B": b, "returns_to_X": back, "random_B": r}; print(name, O[name], flush=True)
top32 = as_set(order[:32])
for where in ("read", "other"):
    res = [blocked_at(i, a, top32, where) for i, a in te]; E = np.mean([i["E"] for i, a in te])
    O[f"top32_restore_{where}_only"] = 1 - np.mean([r[0] for r in res]) / E; print(where, O[f"top32_restore_{where}_only"], flush=True)
for fam in H["other_families"]:
    its = sel(fam=fam); S = -np.sum([a["attr"] for i, a in its], 0); o = list(np.argsort(-S.ravel()))[:16]
    O[f"{fam}_own_top16"] = {"heads": [(int(k) // NH, int(k) % NH) for k in o], "B_in_sample": B(its, as_set(o))[0],
                             "overlap_with_country_top32": len(set(o) & set(order[:32]))}
    print(fam, O[f"{fam}_own_top16"], flush=True)
json.dump(O, open(HERE / "split.json", "w"), indent=1)
