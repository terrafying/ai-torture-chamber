"""exp115 (hypotheses.json): do weekdays and months sit on circles inside Qwen3-4B, and does the model use them?
  shape  per block, PCA on the item means (6 templates each); circular order = mean resultant length of the item
         angles against their calendar positions (best direction); null = 1,000 label shuffles, max over blocks.
  use    weekdays: "What day of the week is two days after X?" Rotate the residual at the X token, at the best block,
         by one day along the fitted circle; does the answer move one day? Control: same-norm change in random planes.
Writes runs/exp115/results.json and ~/repos/research/neuron_tower_data/manifolds/geometry.json (3D PCA per block,
for the page).
    HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python runs/exp115/manifolds.py"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parent / "exp111"))
from tower_cfg import DEV, load
OUT = Path(os.path.expanduser("~/repos/research/neuron_tower_data/manifolds")); OUT.mkdir(parents=True, exist_ok=True)
SETS = {
 "weekdays": (["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
              ["Today is{X}.", "See you on{X}!", "The meeting moved to{X} morning.", "Every{X} we go swimming.", "It rained all day on{X}.", "My favourite day is{X}."]),
 "months": (["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
            ["My birthday is in{X}.", "The festival starts in{X}.", "It snowed in{X} last year.", "Every{X} we travel.", "The report is due in{X}.", "Each year in{X} prices rise."]),
}
tok, m = load(); NL = m.config.num_hidden_layers; rng = np.random.default_rng(0)
def resid_at_item(template, item):
    pre = template.split("{X}")[0]; ids = tok(template.replace("{X}", " " + item), add_special_tokens=False)["input_ids"]
    pos = len(tok(pre + " " + item, add_special_tokens=False)["input_ids"]) - 1
    with torch.no_grad(): hs = m(torch.tensor([ids], device=DEV), output_hidden_states=True).hidden_states
    return torch.stack([h[0, pos] for h in hs[1:]]).float().cpu().numpy()            # blocks x D (after each block)
def circ(theta, N):   # best-direction mean resultant length of angles vs calendar positions
    pos = 2 * np.pi * np.arange(N) / N
    return max(abs(np.exp(1j * (theta - s * pos)).mean()) for s in (1, -1))
res = {}; geo = {}
for name, (items, temps) in SETS.items():
    X = np.stack([[resid_at_item(t, it) for t in temps] for it in items])          # items x temps x blocks x D
    N = len(items); stats, thetas, planes = [], [], []
    for b in range(NL):
        Mn = X[:, :, b].mean(1); mu = Mn.mean(0); U, S, Vt = np.linalg.svd(Mn - mu, full_matrices=False)
        c = (Mn - mu) @ Vt[:2].T; th = np.arctan2(c[:, 1], c[:, 0]); stats.append(circ(th, N)); thetas.append(th)
        planes.append((mu, Vt[:3], (S[:2] ** 2).sum() / (S ** 2).sum()))
    stats = np.array(stats); best = int(stats.argmax())
    null = np.array([max(circ(th[perm], N) for th in thetas) for perm in (rng.permutation(N) for _ in range(1000))])
    p = float((null >= stats.max()).mean())
    res[name] = {"best_block": best, "circular_order": round(float(stats[best]), 3), "p_vs_shuffle_max_over_blocks": p,
                 "null_p95": round(float(np.quantile(null, .95)), 3), "var_top2_at_best": round(float(planes[best][2]), 3),
                 "by_block": [round(float(x), 3) for x in stats]}
    # geometry for the page: every item x template point in each block's top-3 PC frame
    geo[name] = {"items": items, "blocks": []}
    for b in range(NL):
        mu, V3, _ = planes[b]; P3 = (X[:, :, b] - mu) @ V3.T; sc = np.abs(P3).max() + 1e-9
        geo[name]["blocks"].append(np.round(P3 / sc, 3).tolist())
    print(name, {k: v for k, v in res[name].items() if k != "by_block"}, flush=True)
    if name == "weekdays": DAY = (items, X, planes, best)

# ---- use test (weekdays)
items, X, planes, best = DAY; mu, V3, _ = planes[best]; u1, u2 = V3[0], V3[1]
Mn = X[:, :, best].mean(1); c = (Mn - mu) @ np.stack([u1, u2]).T; th = np.arctan2(c[:, 1], c[:, 0])
sdir = max((1, -1), key=lambda s: abs(np.exp(1j * (th - s * 2 * np.pi * np.arange(7) / 7)).mean()))
# the fitted orientation: does increasing calendar position go clockwise or counter-clockwise in (u1, u2)?
step = sdir * 2 * np.pi / 7
edit = {"delta": None}
def hook(mod, inp, out):
    if edit["delta"] is None: return out
    h = out[0] if isinstance(out, tuple) else out; h = h.clone(); h[0, edit["pos"]] += edit["delta"].to(h.dtype)
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
m.model.layers[best].register_forward_hook(hook)
def ask(day):
    msg = f"What day of the week is two days after {day}? Answer with just the day."
    ids = tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, enable_thinking=False)
    ids = ids if isinstance(ids, list) else ids["input_ids"]
    dpos = [i for i, x in enumerate(ids) if tok.decode([x]).strip() == day][-1]
    return ids, dpos
def answer(ids):
    with torch.no_grad(): lg = m(torch.tensor([ids], device=DEV)).logits[0, -1].float()
    for j in lg.topk(10).indices.tolist():
        w = tok.decode([j]).strip()
        for d in items:
            if w and d.startswith(w) and len(w) >= 2: return d
    return tok.decode([int(lg.argmax())]).strip()
def h_at(ids, pos):
    with torch.no_grad(): return m(torch.tensor([ids], device=DEV), output_hidden_states=True).hidden_states[best + 1][0, pos].float().cpu().numpy()
rows = []; rs = np.random.default_rng(1)
for i, d in enumerate(items):
    ids, pos = ask(d); edit["delta"] = None; base = answer(ids); h = h_at(ids, pos)
    cc = np.array([(h - mu) @ u1, (h - mu) @ u2]); R = np.array([[np.cos(step), -np.sin(step)], [np.sin(step), np.cos(step)]])
    delta = (R @ cc - cc) @ np.stack([u1, u2]); edit.update(delta=torch.tensor(delta, dtype=torch.float32, device=DEV), pos=pos); rot = answer(ids)
    ctrl = []
    for _ in range(20):
        Q, _ = np.linalg.qr(rs.normal(size=(len(h), 2))); q = np.array([(h - mu) @ Q[:, 0], (h - mu) @ Q[:, 1]]); dr = (R @ q - q) @ Q.T
        dr = dr / (np.linalg.norm(dr) + 1e-9) * np.linalg.norm(delta); edit["delta"] = torch.tensor(dr, dtype=torch.float32, device=DEV); ctrl.append(answer(ids))
    edit["delta"] = None
    rows.append({"day": d, "baseline": base, "baseline_right": base == items[(i + 2) % 7], "rotated": rot, "moved_one_day": rot == items[(i + 3) % 7],
                 "control_moved": sum(c_ == items[(i + 3) % 7] for c_ in ctrl) / 20, "control_changed": sum(c_ != base for c_ in ctrl) / 20, "delta_norm": round(float(np.linalg.norm(delta)), 2)})
    print(rows[-1], flush=True)
ok = [r for r in rows if r["baseline_right"]]
res["use"] = {"block": best, "rows": rows, "baseline_right": len(ok), "moved_one_day": sum(r["moved_one_day"] for r in ok),
              "control_moved_mean": round(float(np.mean([r["control_moved"] for r in ok])) * len(ok), 2) if ok else None,
              "H3": bool(ok and sum(r["moved_one_day"] for r in ok) >= 4 and np.mean([r["control_moved"] for r in ok]) * len(ok) <= 1)}
res["H1"] = res["weekdays"]["p_vs_shuffle_max_over_blocks"] < .01; res["H2"] = res["months"]["p_vs_shuffle_max_over_blocks"] < .01
json.dump(res, open(HERE / "results.json", "w"), indent=1); json.dump(geo, open(OUT / "geometry.json", "w"), separators=(",", ":"))
print(json.dumps({k: v for k, v in res.items() if k not in ("weekdays", "months")} | {"use": {k: v for k, v in res["use"].items() if k != "rows"}}))
