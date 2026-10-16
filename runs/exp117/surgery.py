"""exp117 (hypotheses.json): concept surgery across seven concept groups in Qwen3-4B.
For each set: the model's own mapping f(X) (first answer token), item means at every block from 6 neutral sentences,
the set's shape, then swaps X -> Y at the X token of the task prompt: add alpha * (mean_Y - mean_X) restricted to the
top-D principal directions of the set's means at block b; success = the answer becomes f(Y).
Writes runs/exp117/results.json (+ per-pair rows) and ~/repos/research/neuron_tower_data/surgery/geometry.json.
    HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python runs/exp117/surgery.py"""
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parent / "exp111"))
from tower_cfg import DEV, load
OUT = Path(os.path.expanduser("~/repos/research/neuron_tower_data/surgery")); OUT.mkdir(parents=True, exist_ok=True)
SETS = {
 "weekdays": ("Monday Tuesday Wednesday Thursday Friday Saturday Sunday", "cycle", "What day comes after {X}?"),
 "months": ("January February March April May June July August September October November December", "cycle", "What month comes after {X}?"),
 "numbers": ("one two three four five six seven eight nine ten", "line", "What number comes after {X}? Answer with a word."),
 "planets": ("Mercury Venus Earth Mars Jupiter Saturn", "line", "Which planet comes after {X} in order from the Sun?"),
 "countries": ("France Japan Germany Italy Spain China Egypt Canada Russia", "unordered", "What is the capital of {X}?"),
 "colours": ("red orange yellow green blue purple", "unordered", "Name a fruit that is usually {X}."),
 "animals": ("dog cat cow duck sheep lion pig horse", "unordered", "What sound does a {X} make?"),
}
NEUTRAL = ["I was thinking about{X} today.", "We talked about{X} for a while.", "The word was{X}.", "Someone mentioned{X} again.", "It was all about{X}.", "Let me tell you about{X}."]
BLOCKS = [2, 4, 6, 8, 10, 12, 16, 20, 24]; ALPHAS = [.25, .5, .75, 1, 1.5]
tok, m = load(); NL = m.config.num_hidden_layers; rng = np.random.default_rng(0)
edit = {"b": None}
def mk(b):
    def hook(mod, inp, out):
        if edit["b"] != b: return out
        tup = isinstance(out, tuple); h = (out[0] if tup else out).clone(); h[0, edit["pos"]] += edit["d"].to(h.dtype)
        return (h,) + tuple(out[1:]) if tup else h
    return hook
for b in range(NL): m.model.layers[b].register_forward_hook(mk(b))
def item_pos(ids, item):
    hits = [i for i, x in enumerate(ids) if tok.decode([x]).strip() == item]; return hits[-1]
def resid_at(text, item):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    with torch.no_grad(): hs = m(torch.tensor([ids], device=DEV), output_hidden_states=True).hidden_states
    p = item_pos(ids, item); return torch.stack([h[0, p] for h in hs[1:]]).float().cpu().numpy()
def task_ids(task, item):
    ids = tok.apply_chat_template([{"role": "user", "content": task.replace("{X}", item) + " Answer with just one word."}], add_generation_prompt=True, enable_thinking=False)
    ids = ids if isinstance(ids, list) else ids["input_ids"]; return ids, item_pos(ids, item)
def answer(ids):
    with torch.no_grad(): lg = m(torch.tensor([ids], device=DEV)).logits[0, -1]
    return tok.decode([int(lg.argmax())]).strip().lower()
res = {"sets": {}}; geo = {}; t0 = time.time()
for name, (items, kind, task) in SETS.items():
    items = items.split(); edit["b"] = None
    base = {x: answer(task_ids(task, x)[0]) for x in items}
    keep = [x for x in items if list(base.values()).count(base[x]) == 1]
    dropped = [x for x in items if x not in keep]
    X = np.stack([[resid_at(t.replace("{X}", " " + x), x) for t in NEUTRAL] for x in keep])    # items x temps x blocks x D
    n = len(keep); means = X.mean(1)                                                              # items x blocks x D
    shape = {}
    for b in range(NL):
        Mn = means[:, b]; mu = Mn.mean(0); U, S, Vt = np.linalg.svd(Mn - mu, full_matrices=False); c = (Mn - mu) @ Vt[:2].T
        row = {"var2": float((S[:2] ** 2).sum() / (S ** 2).sum())}
        if kind == "cycle":
            th = np.arctan2(c[:, 1], c[:, 0]); pos = 2 * np.pi * np.array([items.index(x) for x in keep]) / len(items)
            row["circular"] = float(max(abs(np.exp(1j * (th - s * pos)).mean()) for s in (1, -1)))
        if kind == "line": row["linear"] = float(abs(spearmanr(c[:, 0], [items.index(x) for x in keep]).correlation))
        shape[b] = row
    basis = {b: np.linalg.svd(means[:, b] - means[:, b].mean(0), full_matrices=False)[2] for b in BLOCKS}
    pairs = [(x, y) for x in keep for y in keep if x != y]; rows = []
    for x, y in pairs:
        ids, pos = task_ids(task, x); r = {"x": x, "y": y, "fx": base[x], "fy": base[y]}; i, j = keep.index(x), keep.index(y)
        for b in BLOCKS:
            for D in (2, n - 1):
                d = means[j, b] - means[i, b]; V = basis[b][:D]; d = (d @ V.T) @ V
                edit.update(b=b, pos=pos, d=torch.tensor(d, dtype=torch.float32, device=DEV)); a = answer(ids); r[f"b{b}_D{'all' if D == n - 1 else D}"] = a
        d = means[j, 8] - means[i, 8]; V = basis[8][:n - 1]; d = (d @ V.T) @ V
        for al in ALPHAS:
            edit.update(b=8, pos=pos, d=torch.tensor(al * d, dtype=torch.float32, device=DEV)); r[f"alpha{al}"] = answer(ids)
        rd = rng.normal(size=d.shape); rd = rd / np.linalg.norm(rd) * np.linalg.norm(d)
        edit.update(b=8, pos=pos, d=torch.tensor(rd, dtype=torch.float32, device=DEV)); r["control"] = answer(ids); edit["b"] = None
        rows.append(r)
    succ = lambda key: float(np.mean([r[key] == r["fy"] for r in rows])) if rows else None
    S_ = {"items": keep, "dropped": dropped, "baseline": base, "kind": kind, "pairs": len(rows),
          "full_by_block": {b: succ(f"b{b}_Dall") for b in BLOCKS}, "two_d_by_block": {b: succ(f"b{b}_D2") for b in BLOCKS},
          "dose_block8": {al: succ(f"alpha{al}") for al in ALPHAS}, "control_block8": succ("control"),
          "shape_best": max(shape.values(), key=lambda r: r.get("circular", r.get("linear", r["var2"]))), "shape_by_block": shape, "rows": rows}
    res["sets"][name] = S_
    geo[name] = {"items": keep, "kind": kind, "blocks": []}
    for b in range(NL):
        Mn = means[:, b]; mu = Mn.mean(0); V3 = np.linalg.svd(Mn - mu, full_matrices=False)[2][:3]; P3 = (X[:, :, b] - mu) @ V3.T
        geo[name]["blocks"].append(np.round(P3 / (np.abs(P3).max() + 1e-9), 3).tolist())
    print(name, "kept", len(keep), "dropped", dropped, "| full@8", S_["full_by_block"][8], "2d@8", S_["two_d_by_block"][8], "full@24", S_["full_by_block"][24],
          "control", S_["control_block8"], "| shape", {k: round(v, 3) for k, v in S_["shape_best"].items()}, round(time.time() - t0), "s", flush=True)
    json.dump(res, open(HERE / "results.json", "w"), indent=1); json.dump(geo, open(OUT / "geometry.json", "w"), separators=(",", ":"))
SS = res["sets"]
res["H1"] = bool(sum(s["full_by_block"][8] >= .5 for s in SS.values()) >= 5 and np.mean([s["control_block8"] for s in SS.values()]) <= .05)
res["H2"] = bool(all(s["two_d_by_block"][8] < s["full_by_block"][8] for s in SS.values()))
res["H3"] = bool(all(s["full_by_block"][24] < s["full_by_block"][8] for s in SS.values()))
json.dump(res, open(HERE / "results.json", "w"), indent=1)
print({k: res[k] for k in ("H1", "H2", "H3")})
