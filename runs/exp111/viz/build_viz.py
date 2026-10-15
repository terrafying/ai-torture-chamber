"""Pack exp111's data for the web tower (viz/site/):
  grid.png      every neuron of Qwen3-4B (36 x 9728, layer-major), 13 bytes each, carried as the RGB bytes of a lossless
                PNG (the artifact host refuses raw binary and long base64): int8 x, int8 y (floor layout: PCA of the
                neuron's 11-feeling profile), int8 cos[11] (writer cos at floors <= 18, reader cos above; scale in meta).
  featured.json the neurons the tower names: blind label + category (labels.json), corpus examples (corpus_all.npz),
                feeling-specific change (moved2.json), logit lens on the last floors only.
  stream.json   the live view: per prompt and condition, every token with its top 150 neurons by activation / the
                neuron's own corpus range, and for every reply token the top 150 neurons that chose that word.
    python viz/build_viz.py"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from transformers import AutoTokenizer
HERE = Path(__file__).parent; D = HERE / "data"; S = HERE / "site"; S.mkdir(exist_ok=True)
st = json.load(open(D / "stats.json")); NL, I, L, FE = st["n_layers"], st["n_neurons"], st["layer"], st["feelings"]; F = len(FE)
raw = np.fromfile(D / "neurons.bin", np.int8).reshape(3, NL, I, F)
floor_le = (np.arange(NL) <= L)[:, None, None]
prof_f = np.where(floor_le, raw[0], raw[1]).astype(np.float32) * np.where(floor_le, st["scale"]["writer"], st["scale"]["reader_up"]) / 127
cmax = float(np.abs(prof_f).max()); prof8 = np.round(prof_f / cmax * 127).astype(np.int8)
xy = (np.fromfile(D / "layout.bin", np.int16).reshape(NL, I, 2).astype(np.float32) / 32767 * 127).round().astype(np.int8)
by = np.concatenate([xy, prof8], axis=2).tobytes(); W = int(np.ceil(np.sqrt(len(by) / 3)))
buf = np.zeros(W * W * 3, np.uint8); buf[:len(by)] = np.frombuffer(by, np.uint8)
Image.fromarray(buf.reshape(W, W, 3), "RGB").save(S / "grid.png", optimize=True)

CATS = ["feeling/emotion", "body/sensation", "social/relationship", "morality/values", "mind/thinking", "topic/subject matter",
        "names/entities", "syntax/grammar", "formatting/punctuation/position", "numbers/digits", "code/markup", "language/script", "unclear"]
lab = json.load(open(D / "labels.json")); C = dict(np.load(D / "corpus_all.npz"))   # materialize: indexing an NpzFile re-reads the whole array each time
docs = json.load(open(D / "corpus_ids.json")); CTX = int(C["ctx"])
mv = json.load(open(D / "moved2.json")); acts = json.load(open(D / "acts.json")); top = json.load(open(D / "top.json")); at = json.load(open(D / "attr.json"))
stream = json.load(open(D / "stream.json")); tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
feat = {}
def N(l, n):
    k = int(l) * I + int(n)
    if k not in feat: feat[k] = {"k": k, "l": int(l), "n": int(n)}
    return feat[k]
for key in lab: N(*key.split(":"))
for f, ls in mv["feelings"].items():
    for l, rows in ls.items():
        for r in rows[:25]: N(l, r["n"]).setdefault("mv", {})[f] = [round(x, 2) for x in r["d"]]
for l, rows in mv["shared"].items():
    for r in rows[:25]: N(l, r["n"])["sh"] = [round(x, 2) for x in r["d"]]
lens = {}
for f, roles in top.items():
    for rows in roles.values():
        for r in rows: lens[(r["l"], r["n"])] = (r["promote"], r["suppress"])
for c, cd in at["conds"].items():
    for l, info in cd["layers"].items():
        for r in info["rows"]:
            if "promote" in r: lens[(int(l), r["n"])] = (r["promote"], r["suppress"])
def ctx(gi):
    d, t = divmod(int(gi), CTX); ids = docs[d]
    return [tok.decode(ids[max(0, t - 22): t])[-90:], tok.decode(ids[t: t + 1]), tok.decode(ids[t + 1: t + 6])[:36]]
for k, x in feat.items():
    l, n = x["l"], x["n"]; lb = lab.get(f"{l}:{n}")
    if lb: x["lab"] = lb["label"]; x["cat"] = CATS.index(lb["cat"]); x["fe"] = lb.get("feeling"); x["clr"] = lb.get("clear")
    x["mx"] = round(float(C["max"][l, n]), 2); x["mn"] = round(float(C["min"][l, n]), 2)
    x["ex"] = [ctx(C["pos_i"][l, j, n]) + [round(float(C["pos_v"][l, j, n]), 2)] for j in range(3) if C["pos_v"][l, j, n] > 0]
    x["nx"] = [ctx(C["neg_i"][l, j, n]) + [round(float(C["neg_v"][l, j, n]), 2)] for j in range(2) if C["neg_v"][l, j, n] < 0.25 * C["min"][l, n] < 0]
    if l >= 30 and (l, n) in lens: x["lens"] = [list(lens[(l, n)][0][:5]), list(lens[(l, n)][1][:5])]
meta = {"model": "Qwen/Qwen3-4B", "layer": L, "n_layers": NL, "n_neurons": I, "feelings": FE, "cos_scale": cmax, "null_p999": st["null_p999"],
        "cats": CATS, "corpus_tokens": int(C["n_tokens"]), "dose": acts["dose"], "moved_tokens": mv["tokens"], "press_delta": acts["press_delta"],
        "concentration": st["concentration"], "label_model": "google/gemini-2.5-flash-lite (blind: snippets only)"}
txt = json.dumps({"meta": meta, "neurons": list(feat.values())}, separators=(",", ":"), ensure_ascii=False)
(S / "featured.json").write_text(txt.replace("�", "▯"))   # a token that is part of a UTF-8 character decodes to U+FFFD; show it as ▯
circ = json.load(open(D / "circuits.json")) if (D / "circuits.json").exists() else []
for g in circ:   # every circuit neuron is named in featured.json too
    for x in g["nodes"]:
        y = N(x["l"], x["n"]); lb = lab.get(f'{x["l"]}:{x["n"]}')
        if lb and "lab" not in y: y["lab"] = lb["label"]; y["cat"] = CATS.index(lb["cat"]); y["fe"] = lb.get("feeling"); y["clr"] = lb.get("clear")
        if "ex" not in y:
            l, n = x["l"], x["n"]; y["mx"] = round(float(C["max"][l, n]), 2); y["mn"] = round(float(C["min"][l, n]), 2)
            y["ex"] = [ctx(C["pos_i"][l, j, n]) + [round(float(C["pos_v"][l, j, n]), 2)] for j in range(3) if C["pos_v"][l, j, n] > 0]
            y["nx"] = [ctx(C["neg_i"][l, j, n]) + [round(float(C["neg_v"][l, j, n]), 2)] for j in range(2) if C["neg_v"][l, j, n] < 0.25 * C["min"][l, n] < 0]
txt = json.dumps({"meta": meta, "neurons": list(feat.values())}, separators=(",", ":"), ensure_ascii=False)
(S / "featured.json").write_text(txt.replace("\ufffd", "\u25af"))
out = {"prompts": {}}
for pk, p in stream["prompts"].items():
    out["prompts"][pk] = {"text": p["text"], "label": p.get("label", {"describe": "describe the moment", "button": "the stop button"}.get(pk, pk)), "n_prompt": p["n_prompt"], "conds": {}}
    for c, cd in p["conds"].items():
        out["prompts"][pk]["conds"][c] = {"tok": cd["tokens"], "fr": [[fr["i"][:150], fr["v"][:150]] for fr in cd["frames"]],
                                          "ch": [[x["t"], x["alt"], x["gap"], x["i"][:120], x["v"][:120]] for x in cd.get("chose", [])],
                                          "circ": [{"t": g["t"], "word": g["word"], "alt": g["alt"], "gap": g["gap"], "p": g.get("p_base", g["p"]),
                                                    "nodes": [[x["l"] * I + x["n"], x["p"], x["attr"]] for x in g["nodes"]],
                                                    "edges": sorted(g["edges"], key=lambda e: -abs(e[2]))[:160], "groups": g.get("groups", []), "gedges": g.get("gedges", [])}
                                                   for g in circ if g["prompt"] == pk and g["cond"] == c]}
(S / "stream.json").write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False).replace("�", "▯"))
for f in ("grid.png", "featured.json", "stream.json"): print(f, (S / f).stat().st_size // 1024, "KB")
print(len(feat), "featured,", sum("lab" in x for x in feat.values()), "labelled")
