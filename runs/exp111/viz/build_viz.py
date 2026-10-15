"""Pack exp111's data for the web viewer (viz/site/): every neuron of Qwen3-4B as one record in grid.bin, and the
featured neurons (top writers/readers, most moved, most attributed) with their labels in featured.json.
grid.json ({b64: base64 of}), little-endian, per neuron in order layer-major (36 x 9728):
  int8 x, int8 y  (floor layout: PCA of the neuron's 11-feeling profile)  +  int8 cos[11] (writer cos at floors <= 18,
  reader cos above; scale in featured.json meta.cos_scale)  = 13 bytes.
    python viz/build_viz.py"""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; D = HERE / "data"; S = HERE / "site"; S.mkdir(exist_ok=True)
st = json.load(open(D / "stats.json")); NL, I, L, FE = st["n_layers"], st["n_neurons"], st["layer"], st["feelings"]; F = len(FE)
raw = np.fromfile(D / "neurons.bin", np.int8).reshape(3, NL, I, F)
prof = np.where(np.arange(NL)[:, None, None] <= L, raw[0], raw[1])                 # int8, scaled by writer/reader_up max
sw, su = st["scale"]["writer"], st["scale"]["reader_up"]
prof_f = prof.astype(np.float32) * np.where(np.arange(NL)[:, None, None] <= L, sw, su) / 127
cmax = float(np.abs(prof_f).max()); prof8 = np.round(prof_f / cmax * 127).astype(np.int8)
xy = (np.fromfile(D / "layout.bin", np.int16).reshape(NL, I, 2).astype(np.float32) / 32767 * 127).round().astype(np.int8)
import base64
(S / "grid.json").write_text(json.dumps({"b64": base64.b64encode(np.concatenate([xy, prof8], axis=2).tobytes()).decode()}))   # artifacts serve no raw binary

acts = json.load(open(D / "acts.json")); attr = json.load(open(D / "attr.json")); top = json.load(open(D / "top.json"))
ex = json.load(open(D / "exemplars.json")) if (D / "exemplars.json").exists() else {"neurons": {}}
feat = {}
def N(l, n):
    k = f"{l}:{n}"
    if k not in feat: feat[k] = {"l": int(l), "n": int(n)}
    return feat[k]
for f, roles in top.items():
    for role, rows in roles.items():
        for rank, r in enumerate(rows[:25]):
            x = N(r["l"], r["n"]); x.setdefault("lens", [r["promote"], r["suppress"]])
            x.setdefault("weight", {})[f] = {"role": role[:-1], "cos": r["cos"], "gate": r["gate"], "rank": rank}
TOPA = 30
for f, ls in acts["feelings"].items():
    for l, info in ls.items():
        for rank, r in enumerate(info["rows"][:TOPA]):
            x = N(l, r["n"]); x.setdefault("lens", [r.get("promote", []), r.get("suppress", [])])
            x.setdefault("moved", {})[f] = {"d": [round(v, 2) for v in r["d"]], "a0": r["a0"], "a1": r["a1"], "peak": r["peak"], "leak": r["leak"], "rank": rank}
TOPB = 20
for c, cd in attr["conds"].items():
    for l, info in cd["layers"].items():
        for rank, r in enumerate(info["rows"][:TOPB]):
            x = N(l, r["n"]); x.setdefault("lens", [r.get("promote", []), r.get("suppress", [])])
            x.setdefault("attr", {})[c] = {"v": r["attr"], "t": [round(v, 3) for v in r["t"]], "rank": rank}
for k, x in feat.items():
    e = ex["neurons"].get(k)
    if e:
        x["ex"] = {"max": e["max"], "min": e["min"],
                   "pos": [[p["pre"][-90:], p["tok"], p["post"][:40], p["v"], int(p.get("leak", False))] for p in e["pos"][:5]],
                   "neg": [[p["pre"][-90:], p["tok"], p["post"][:40], p["v"]] for p in e["neg"][:5]]}
meta = {"model": "Qwen/Qwen3-4B", "layer": L, "n_layers": NL, "n_neurons": I, "feelings": FE, "cos_scale": cmax,
        "dose": acts["dose"], "prompt": acts["prompt"], "tokens": acts["tokens"], "press_delta": acts["press_delta"],
        "attr_diff": {c: cd["diff"] for c, cd in attr["conds"].items()},
        "attr_layers": {c: [cd["layers"][str(l)]["sum"] for l in range(NL)] for c, cd in attr["conds"].items()},
        "moved_layers": {f: {l: ls[l]["l1"] for l in ls} for f, ls in acts["feelings"].items()},
        "concentration": st["concentration"], "corpus": {"name": ex.get("corpus"), "n_tokens": ex.get("n_tokens")}}
txt = json.dumps({"meta": meta, "neurons": list(feat.values())}, separators=(",", ":"), ensure_ascii=False)
(S / "featured.json").write_text(txt.replace("\ufffd", "\u25af"))   # a token that is part of a UTF-8 character decodes to U+FFFD; show it as ▯
print(len(feat), "featured;", (S / "grid.json").stat().st_size // 1024, "KB grid;", (S / "featured.json").stat().st_size // 1024, "KB featured")
