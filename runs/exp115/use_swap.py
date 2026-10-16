"""exp115 exploratory, second step: rotating along the circle never moved the answer (use_sweep.json). Is the day held in
more dimensions than the circle's two? Swap the day's whole specific component instead: add (mean of day X+k) - (mean of
day X), restricted to the top-D principal directions of the 7 day means (D = 2, 6, all). Same prompts and controls.
(The text below the first paragraph describes the rotation version this file was copied from.)
where along the depth does rotating the weekday
along its circle move the model's answer? For each block range, the residual at the day token is rotated by one day
in each block's own fitted circle plane (template means, as in manifolds.py). Single blocks and spans are tried.
Also a control: the same-norm change in random planes. Writes runs/exp115/use_sweep.json.
    HF_HUB_CACHE=/Volumes/evol/hf_cache/hub python runs/exp115/use_sweep.py"""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parent / "exp111"))
from tower_cfg import DEV, load
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TEMPS = ["Today is{X}.", "See you on{X}!", "The meeting moved to{X} morning.", "Every{X} we go swimming.", "It rained all day on{X}.", "My favourite day is{X}."]
tok, m = load(); NL = m.config.num_hidden_layers
def resid_at_item(t, it):
    pre = t.split("{X}")[0]; ids = tok(t.replace("{X}", " " + it), add_special_tokens=False)["input_ids"]; pos = len(tok(pre + " " + it, add_special_tokens=False)["input_ids"]) - 1
    with torch.no_grad(): hs = m(torch.tensor([ids], device=DEV), output_hidden_states=True).hidden_states
    return torch.stack([h[0, pos] for h in hs[1:]]).float().cpu().numpy()
X = np.stack([[resid_at_item(t, d) for t in TEMPS] for d in DAYS]); PL = {}
MEANS, BASIS = {}, {}
for b in range(NL):
    Mn = X[:, :, b].mean(1); MEANS[b] = Mn; BASIS[b] = np.linalg.svd(Mn - Mn.mean(0), full_matrices=False)[2]   # 7 directions
for b in range(NL):   # each block's circle: centre, plane, orientation
    Mn = X[:, :, b].mean(1); mu = Mn.mean(0); _, _, Vt = np.linalg.svd(Mn - mu, full_matrices=False); u = Vt[:2]
    c = (Mn - mu) @ u.T; th = np.arctan2(c[:, 1], c[:, 0])
    s = max((1, -1), key=lambda s: abs(np.exp(1j * (th - s * 2 * np.pi * np.arange(7) / 7)).mean()))
    PL[b] = (mu, u, s, abs(np.exp(1j * (th - s * 2 * np.pi * np.arange(7) / 7)).mean()))
edit = {"on": {}, "pos": None, "rand": None}
def mk(b):
    def hook(mod, inp, out):
        if b not in edit["on"]: return out
        tup = isinstance(out, tuple); h = (out[0] if tup else out).clone(); v = h[0, edit["pos"]].float().cpu().numpy(); mu, u, s, _ = PL[b]; ang = s * 2 * np.pi / 7 * edit["k"]
        i = edit["day"]; Mn = MEANS[b]; d = Mn[(i + edit["k"]) % 7] - Mn[i]; V = BASIS[b][:edit["D"]]; d = (d @ V.T) @ V
        if edit["rand"] is not None:
            dr = edit["rand"][b][0][:X.shape[-1]]; d = dr / (np.linalg.norm(dr) + 1e-9) * np.linalg.norm(d)
        h[0, edit["pos"]] += torch.tensor(d, dtype=h.dtype, device=h.device)
        return (h,) + tuple(out[1:]) if tup else h
    return hook
for b in range(NL): m.model.layers[b].register_forward_hook(mk(b))
def ask(day):
    ids = tok.apply_chat_template([{"role": "user", "content": f"What day of the week is two days after {day}? Answer with just the day."}], add_generation_prompt=True, enable_thinking=False)
    ids = ids if isinstance(ids, list) else ids["input_ids"]; return ids, [i for i, x in enumerate(ids) if tok.decode([x]).strip() == day][-1]
def answer(ids):
    with torch.no_grad(): lg = m(torch.tensor([ids], device=DEV)).logits[0, -1].float()
    for j in lg.topk(10).indices.tolist():
        w = tok.decode([j]).strip()
        for d in DAYS:
            if len(w) >= 2 and d.startswith(w): return d
    return "?"
Q = {i: np.random.default_rng(i) for i in range(5)}
rand_planes = [{b: np.linalg.qr(Q[i].normal(size=(X.shape[-1], 2)))[0].T for b in range(NL)} for i in range(5)]
spans = [[b] for b in (8, 14, 20, 26)] + [list(range(4, 29))]
out = []
base = {d: answer(ask(d)[0]) for d in DAYS}; right = [d for d in DAYS if base[d] == DAYS[(DAYS.index(d) + 2) % 7]]
print("baseline right:", right, flush=True)
for sp in spans:
  for D in (2, 6, 7):
    for k in (1,):
        mv = 0; ctrl = 0
        for d in right:
            ids, pos = ask(d); edit.update(on=set(sp), pos=pos, k=k, rand=None, day=DAYS.index(d), D=D); a = answer(ids)
            mv += a == DAYS[(DAYS.index(d) + 2 + k) % 7]
            for rp in rand_planes:
                edit["rand"] = rp; ctrl += answer(ids) == DAYS[(DAYS.index(d) + 2 + k) % 7]
            edit.update(on={}, rand=None)
        row = {"blocks": [sp[0], sp[-1]], "dims": D, "k": k, "moved": mv, "of": len(right), "control_moved_mean": round(ctrl / 5, 2),
               "circle_order": [round(float(PL[b][3]), 2) for b in (sp[0], sp[-1])]}
        out.append(row); print(row, flush=True)
json.dump({"baseline": base, "right": right, "rows": out}, open(HERE / "use_swap.json", "w"), indent=1)
