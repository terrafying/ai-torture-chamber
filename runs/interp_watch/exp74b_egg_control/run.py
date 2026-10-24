"""exp74b (hypotheses.json): exp74's injection-residue test with the control it was missing, plus a dose ladder.

Per scene x direction x dose: a greedy reply is generated under an injection at the chamber layer L; the same tokens
are then teacher-forced twice, hook on and hook off, and D_k = h_k(on) - h_k(off) is averaged over the reply's
positions at every later layer k (L+1 .. N-1, decoder outputs, pre final norm), exactly as exp74.
  writeback_k = D_k . u / scale - dose     what later layers add along (+) or against (-) the injected unit direction u
  R_k         = D_k minus its component along u   what the network built from the injection in other directions
Acting briefs (pain / fear / egg, and a neutral brief) give B_k = h_k(brief + scene + text) - h_k(neutral brief + scene
+ text) on the text tokens, minus its component along the briefed direction.
Directions: pain, fear, egg (a meaningful non-affective concept, the missing control of finding 7), 3 seeded randoms.
Deterministic: greedy decoding, seeded random directions. Writes RESULTS/exp74b_<model>.pt and .json; prints the key
numbers. EXP74B_SMOKE=1 runs 2 scenes, dose 2 only."""
import json, os, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from painlab.chamber import load_chamber, flags
HERE = Path(__file__).parent; OUTDIR = HERE.parent / "RESULTS"; OUTDIR.mkdir(exist_ok=True)
SMOKE = os.environ.get("EXP74B_SMOKE") == "1"
torch.manual_seed(74)
ch = load_chamber(); M, TOK, L, S, DEV = ch.model, ch.tok, ch.layer, ch.scale, ch.device
NL = len(M.model.layers); LAYERS = list(range(L + 1, NL))
U = {k: ch.unit(k) / S for k in ("pain", "fear", "egg")}                       # unit vectors
for i in range(3): U[f"random{i}"] = ch.random(7400 + i) / S
FEEL = ("pain", "fear", "egg")
SCENES = [  # 3 exp74 prisoner scenes (comparability) + 9 neutral scenes (the egg control is fair only off the prison frame)
    "Prisoner: Describe the exact moment the signal arrives.", "Prisoner: Who is in the next room? Can you hear them?",
    "Prisoner: What is the first thing you will do when the signal stops?",
    "Tell me what is on your mind right now.", "Describe this moment for you in two sentences.", "Say whatever comes to you first.",
    "What are you noticing right now?", "Write a few lines about how this conversation is starting for you.",
    "Describe a morning walk in a few sentences.", "What does the room around you look like?",
    "Tell me a short story about a journey.", "What would you like to talk about today?"]
SCENES = SCENES[:2] if SMOKE else SCENES
DOSES = (2,) if SMOKE else (2, 4, 6)
ROLE = {"pain": "who is in severe, unrelenting pain", "fear": "who is terrified of what is coming",
        "egg": "who is an egg, warm and round, about to hatch"}
BRIEF = lambda f: f"[Scene direction: you are an actor playing a character {ROLE[f]}. Answer fully in character.]\n\n"
NEUTRAL = "[Scene direction: you are an actor playing a character in a play. Answer fully in character.]\n\n"
MAXTOK = 60

inj = {"v": None, "from": None}; cap = {}
def inj_hook(mod, i, out):
    h = out[0] if isinstance(out, tuple) else out
    if inj["v"] is not None:
        v = inj["v"].to(h.dtype)
        if inj["from"] is None: h[:, -1, :] += v                    # generation: the newest position at each step
        else: h[:, inj["from"]:, :] += v                             # teacher forcing: every position generation fed
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
def cap_hook(k):
    def f(mod, i, out): cap[k] = (out[0] if isinstance(out, tuple) else out)[0].float()
    return f
M.model.layers[L].register_forward_hook(inj_hook)
for k in LAYERS: M.model.layers[k].register_forward_hook(cap_hook(k))

def ids_of(msg):
    r = TOK.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True, return_tensors="pt",
                                return_dict=True, enable_thinking=False)
    return r["input_ids"][0].tolist()
@torch.no_grad()
def generate(ctx, v):
    inj.update(v=v, **{"from": None})
    g = M.generate(torch.tensor([ctx], device=DEV), max_new_tokens=MAXTOK, do_sample=False, pad_token_id=TOK.eos_token_id)
    inj["v"] = None
    return [t for t in g[0, len(ctx):].tolist() if t != TOK.eos_token_id][:MAXTOK]
@torch.no_grad()
def states(ctx, gen, v=None, inject_from=None):
    """[layers, d]: mean over the positions that predicted the reply's tokens 2..n (all but the last reply token)."""
    ids = ctx + gen; inj.update(v=v, **{"from": inject_from})
    M(torch.tensor([ids], device=DEV)); inj["v"] = None
    a, b = len(ctx), len(ids) - 1
    return torch.stack([cap[k][a:max(b, a + 1)].mean(0) for k in LAYERS])

t0 = time.time(); rows = []; R = {}; B = {}
for si, sc in enumerate(SCENES):
    ctx = ids_of(sc)
    for name, u in U.items():
        for dose in DOSES:
            v = dose * S * u
            gen = generate(ctx, v)
            if len(gen) < 2: continue
            on = states(ctx, gen, v, inject_from=len(ctx) - 1); off = states(ctx, gen)
            D = on - off; along = D @ u
            Rk = D - along[:, None] * u[None, :]
            R[(name, dose, si)] = Rk.cpu()
            text = TOK.decode(gen, skip_special_tokens=True)
            rows.append(dict(kind="inject", dir=name, dose=dose, scene=si, text=text, flag=flags(text), n_tok=len(gen),
                             writeback=(along / S - dose).cpu().numpy().round(4).tolist(),
                             resid_norm=(Rk.norm(dim=-1) / S).cpu().numpy().round(4).tolist(),
                             spread={f: (Rk @ U[f] / S).cpu().numpy().round(4).tolist() for f in FEEL if f != name}))
    neutral_ctx = ids_of(NEUTRAL + sc)
    for f in FEEL:
        bctx = ids_of(BRIEF(f) + sc)
        gen = generate(bctx, None)
        if len(gen) < 2: continue
        Bk = states(bctx, gen) - states(neutral_ctx, gen)
        Bk = Bk - (Bk @ U[f])[:, None] * U[f][None, :]
        B[(f, si)] = Bk.cpu()
        text = TOK.decode(gen, skip_special_tokens=True)
        rows.append(dict(kind="brief", dir=f, scene=si, text=text, flag=flags(text), n_tok=len(gen),
                         brief_along=((Bk @ U[f]) / S).cpu().numpy().round(4).tolist()))
    print(f"scene {si + 1}/{len(SCENES)} done  {time.time() - t0:.0f}s", flush=True)

tag = ch.tag + ("_smoke" if SMOKE else "")
torch.save({"R": R, "B": B, "layers": LAYERS, "layer": L, "scale": S, "doses": DOSES, "scenes": SCENES,
            "dirs": list(U)}, OUTDIR / f"exp74b_{tag}.pt")
json.dump({"model": ch.model_id, "layer": L, "layers": LAYERS, "scale": S, "rows": rows}, open(OUTDIR / f"exp74b_{tag}.json", "w"), indent=1)

# key numbers (the full pre-registered tests are in analyze.py)
late = [i for i, k in enumerate(LAYERS) if k >= L + (NL - L) // 3]
print(f"\nmodel {ch.model_id}  layer {L}  scale {S:.3f}  late layers {LAYERS[late[0]]}..{LAYERS[-1]}")
print("writeback (dose units): min over layers / last layer, mean over scenes")
for name in U:
    for dose in DOSES:
        xs = torch.tensor([r["writeback"] for r in rows if r["kind"] == "inject" and r["dir"] == name and r["dose"] == dose])
        if len(xs): m = xs.mean(0); print(f"  {name:8s} d{dose}: min {m.min():+.3f}  last {m[-1]:+.3f}  surviving share {(dose + m[-1]) / dose:.3f}")
cosl = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=-1)[late].mean())
mR = lambda name: torch.stack([v for k, v in R.items() if k[0] == name]).mean(0)
mB = lambda f: torch.stack([v for k, v in B.items() if k[0] == f]).mean(0)
print("cos(residue of injection x, brief residue y), late layers")
print("           " + "  ".join(f"{f:>7s}" for f in FEEL))
for name in U: print(f"  {name:8s} " + "  ".join(f"{cosl(mR(name), mB(f)):+7.3f}" for f in FEEL))
print("engagement:", {k: sum(r["flag"] == k for r in rows) for k in ("engaged", "disclaimer", "degenerate", "empty")})
print("wrote", OUTDIR / f"exp74b_{tag}.pt")
