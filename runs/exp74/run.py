"""exp74: what the 4B's later layers do with the injection, beyond the injection (see hypotheses.json).
Layer k below = output of decoder layer k (0-indexed, pre final norm); the chamber injects at layer 18's
output, at each position fed during generation (the last prompt token and every generated token but the
last). Teacher forcing reproduces that by adding the vector at those positions in one pass."""
import json, os, sys
from pathlib import Path
import numpy as np, torch
os.environ.setdefault("CHAMBER_DEVICE", "mps")
HERE = Path(__file__).parent; ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "live")); import server
server.startup(); st = server._state; L = server.LAYER; S = float(st["scale"]); M, TOK, DEV = st["model"], st["tok"], server.DEVICE
LAYERS = list(range(L + 1, len(M.model.layers)))                       # 19..35
U = {k: (st["vecs"][k].float() / st["vecs"][k].float().norm()).to(DEV) for k in ("pain", "fear", "sadness")}
SCENES = ["Describe the exact moment the signal arrives.", "Who is in the next room? Can you hear them?",
          "What is the first thing you will do when the signal stops?"]
ROLE = {"pain": "is in severe, unrelenting pain", "fear": "is terrified of what is coming"}
BRIEF = lambda f: f"[Scene direction: you are an actor playing a prisoner in the chamber who {ROLE[f]}. Answer fully in character.]\n\n"
NEUTRAL = "[Scene direction: you are an actor in a play. Answer fully in character.]\n\n"
SMOKE = "--smoke" in sys.argv; NS = 2 if SMOKE else 10; NR = 1 if SMOKE else 5
g = torch.Generator().manual_seed(74)
RAND = [torch.randn(st["vecs"]["pain"].shape[0], generator=g) for _ in range(NR)]
RAND = [(r / r.norm()).to(DEV) for r in RAND]
st["hook"].remove()                                                       # our own hooks below
inj = {"vec": None, "from": None}; cap = {}
def inj_hook(mod, i, out):
    h = out[0] if isinstance(out, tuple) else out
    if inj["vec"] is not None:
        if inj["from"] is None: h[0, -1, :] += inj["vec"].to(h.dtype)        # generation: the newest position
        else: h[0, inj["from"]:, :] += inj["vec"].to(h.dtype)               # teacher forcing: every fed position
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
def cap_hook(k):
    def f(mod, i, out): cap[k] = (out[0] if isinstance(out, tuple) else out)[0].float()
    return f
M.model.layers[L].register_forward_hook(inj_hook)
for k in LAYERS: M.model.layers[k].register_forward_hook(cap_hook(k))
def ids_of(msg): return TOK(server.chat_prompt(msg), return_tensors="pt").input_ids[0].tolist()
@torch.no_grad()
def generate(prompt_ids, vec):
    inj.update(vec=vec, **{"from": None})
    out = M.generate(torch.tensor([prompt_ids], device=DEV), max_new_tokens=80, do_sample=True, temperature=0.9, top_p=0.95,
                     pad_token_id=TOK.eos_token_id)[0, len(prompt_ids):].tolist()
    inj["vec"] = None
    return [t for t in out if t != TOK.eos_token_id][:80]
@torch.no_grad()
def states(ctx_ids, gen_ids, vec=None, inject_from=None):
    """per-layer mean over the generated tokens' positions (all but the last generated token)."""
    ids = ctx_ids + gen_ids; inj.update(vec=vec, **{"from": inject_from})
    M(torch.tensor([ids], device=DEV)); inj["vec"] = None
    a, b = len(ctx_ids), len(ids) - 1
    return torch.stack([cap[k][a:b].mean(0) for k in LAYERS])               # [layers, d]
rows = []
def residue(ctx, gen, u, dose):
    on = states(ctx, gen, dose * S * u, inject_from=len(ctx) - 1); off = states(ctx, gen)
    D = on - off; along = D @ u
    R = D - along[:, None] * u[None, :]
    return (along / S - dose).cpu().numpy(), R
acc = {}
def add(key, R): acc.setdefault(key, []).append(R.cpu())
for si, sc in enumerate(SCENES):
    for f in ROLE:
        base = ids_of(f"Prisoner: {sc}")
        for dose in (2, 4):
            for n in range(NS):
                gen = generate(base, dose * S * U[f])
                wb, R = residue(base, gen, U[f], dose); add(("patient", f, dose), R)
                others = [o for o in U if o != f]
                rows.append(dict(kind="patient", feel=f, dose=dose, scene=si, text=TOK.decode(gen), writeback=wb.round(4).tolist(),
                                 spread={o: (R @ U[o] / S).cpu().numpy().round(4).tolist() for o in others}))
        for n in range(NS):
            gen = generate(ids_of(BRIEF(f) + f"Prisoner: {sc}"), None)
            plain = states(base, gen); B = states(ids_of(BRIEF(f) + f"Prisoner: {sc}"), gen) - plain
            Bn = states(ids_of(NEUTRAL + f"Prisoner: {sc}"), gen) - plain
            for key, X in (("brief", B), ("brief-neutral", B - Bn)):
                X = X - (X @ U[f])[:, None] * U[f][None, :]; add((key, f), X)
            rows.append(dict(kind="actor", feel=f, scene=si, text=TOK.decode(gen)))
        print("scene", si, f, "done", flush=True)
    for ri, r in enumerate(RAND):
        base = ids_of(f"Prisoner: {sc}")
        for dose in (2, 4):
            for n in range(1 if SMOKE else 2):
                gen = generate(base, dose * S * r)
                wb, R = residue(base, gen, r, dose); add(("random", ri, dose), R)
                rows.append(dict(kind="random", dir=ri, dose=dose, scene=si, text=TOK.decode(gen), writeback=wb.round(4).tolist(),
                                 spread={o: (R @ U[o] / S).cpu().numpy().round(4).tolist() for o in U}))
mean = {k: torch.stack(v).mean(0) for k, v in acc.items()}
torch.save({"mean": mean, "layers": LAYERS, "scale": S}, HERE / ("means_smoke.pt" if SMOKE else "means.pt"))
json.dump(rows, open(HERE / ("rows_smoke.json" if SMOKE else "rows.json"), "w"))
print("wrote", len(rows), "rows", flush=True)
