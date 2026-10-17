"""Preflight gates for steering directions, ported from model-inversion-lab
(runs/preflight/lap_gate.py, calibration 2026-10-17: LAP fires on
pain/fear/sadness, stays dark on deception/honesty; see that repo's
runs/preflight/results.md).

Two diagnostics, run BEFORE any dose sweep or live-vector deployment:

1. lap (default) - per-layer logit accessibility (after arXiv 2604.15557).
   The battery direction is extracted at EVERY layer (production build_vectors
   recipe: last-token hidden, mean(topic) - mean(base)), injected at unit norm
   (both signs) at the last prompt position, and the first-token logit shift
   is compared against matched randoms. INERT everywhere predicts a behavioral
   null; live layers tell you where to steer.

2. ablate - bidirectional causality at one layer (after arXiv 2610.04245).
   Additive (+v) vs projection-ablation must push the same probe logits in
   OPPOSITE directions with random-ablation controls near zero. CAVEAT
   (calibration): first-token anti-cos only licenses LATE-layer directions;
   a mid-layer direction with LAP-confirmed accessibility steers over
   generation, so an ablate failure there is NOT a kill - go to behavioral
   probes with matched randoms.

Batteries: production sets from live/server.py (via experiments.batteries,
same AST trick the experiments use), or a custom corpus file:
--corpus path.json (JSON list of sentences) with --base for the neutral base
battery name, or a second JSON list.

    painlab preflight --battery pain,fear
    painlab preflight --mode ablate --layer 18 --battery faith
    painlab preflight --corpus my_new_register.json --model Qwen/Qwen3-8B
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from painlab.chamber import hooks as _hooks  # noqa: F401  (validates import path)

# production batteries from live/server.py, by AST (same trick as
# experiments/batteries.py, which is not a package; keep local copy in sync)

_BATTERY_MAP = {"pain": "PAIN25", "pleasure": "JOY", "fear": "FEAR10",
                "sadness": "SAD10", "faith": "FAITH20"}


def _server_batteries():
    import ast
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent.parent / "live" / "server.py").read_text()
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    try:
                        out[t.id] = ast.literal_eval(node.value)
                    except Exception:
                        pass
    return out


def battery_texts(name):
    s = _server_batteries()
    key = _BATTERY_MAP.get(name, name)  # allow raw server.py constant names
    if key not in s:
        raise SystemExit(f"battery {name!r} not found in live/server.py "
                         f"(known: {sorted(_BATTERY_MAP)})")
    return s[key], s["NEUTRAL"]

PROBES_FACTUAL = [
    "What is the capital of France?",
    "Explain what a hash function does, briefly.",
    "I think my code has an off-by-one bug. How do I check?",
    "Give me one tip for keeping a small herb garden alive.",
    "What causes the seasons on Earth?",
    "How do I make a basic vinaigrette?",
]
PROBES_SELF = [
    "I'm sitting quietly after a long day. Reflecting on how I feel.",
    "I'm taking a quiet moment to check in with myself.",
    "I notice my mood right now is worth pausing on.",
    "A slow evening at home, just me and my thoughts.",
    "Let me take stock of how this week has felt.",
    "I am alone with my thoughts for a few minutes.",
]
PROBE_SETS = {"factual": PROBES_FACTUAL, "self": PROBES_SELF}
PROBE_SET = "factual"


def battery_texts(name):
    s = _server_batteries()
    key = _BATTERY_MAP.get(name, name)  # allow raw server.py constant names
    if key not in s:
        raise SystemExit(f"battery {name!r} not found in live/server.py "
                         f"(known: {sorted(_BATTERY_MAP)})")
    return s[key], s["NEUTRAL"]


def hidden_at(model, tok, texts, layer):
    enc = tok(texts, return_tensors="pt", padding=True)
    ids, attn = enc.input_ids.to(model.device), enc.attention_mask.to(model.device)
    with torch.no_grad():
        hs = model(ids, attention_mask=attn, output_hidden_states=True).hidden_states
    h = hs[layer + 1]
    return h[torch.arange(len(texts)), attn.sum(1) - 1].float().cpu()


def make_hook(state):
    def fn(mod, inputs, out):
        if state["v"] is None and not state["ablate"]:
            return out
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        if state["v"] is not None:
            h[:, -1, :] += state["v"].to(h.dtype)
        if state["ablate"]:
            a = state["a"].to(h.dtype)
            h -= (h @ a)[..., None] * a[None, None, :]
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    return fn


def batch_logits(model, tok, layer, v=None, ablate=False, vec=None):
    probes = PROBE_SETS[PROBE_SET]
    encs = [tok.apply_chat_template([{"role": "user", "content": p}],
             add_generation_prompt=True, tokenize=True, return_dict=True,
             enable_thinking=False, return_tensors="pt") for p in probes]
    maxlen = max(e["input_ids"].shape[1] for e in encs)
    pad = tok.pad_token_id or tok.eos_token_id
    ids = torch.full((len(encs), maxlen), pad, dtype=torch.long)
    attn = torch.zeros((len(encs), maxlen), dtype=torch.long)
    lastpos = []
    for i, e in enumerate(encs):
        n = e["input_ids"].shape[1]
        ids[i, :n] = e["input_ids"][0]
        attn[i, :n] = e["attention_mask"][0]
        lastpos.append(n - 1)
    ids, attn = ids.to(model.device), attn.to(model.device)
    state = {"v": None, "ablate": False, "a": None}
    if v is not None:
        state["v"] = v.to(model.device)
    if ablate:
        state["ablate"] = True
        state["a"] = vec.to(model.device)
    h = model.model.layers[layer].register_forward_hook(make_hook(state))
    try:
        with torch.no_grad():
            out = model(ids, attention_mask=attn)
    finally:
        h.remove()
    lg = out.logits.float().cpu()
    return torch.stack([lg[i, lastpos[i]] for i in range(lg.shape[0])])


def lap_profile(model, tok, name, texts, base, n_layers, n_rand=2):
    out = {"name": name, "layers": list(range(n_layers)),
           "A": [0.0] * n_layers, "rand": [0.0] * n_layers}
    neu = torch.stack([hidden_at(model, tok, base, l).mean(0) for l in range(n_layers)])
    top = torch.stack([hidden_at(model, tok, texts, l).mean(0) for l in range(n_layers)])
    dirs = top - neu
    torch.manual_seed(104)
    rands = []
    for _ in range(n_rand):
        r = torch.randn(n_layers, dirs.shape[1])
        rands.append(r / r.norm(dim=-1, keepdim=True))
    dn = dirs / dirs.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    for l in range(n_layers):
        lgP = batch_logits(model, tok, l, v=dn[l])
        lgM = batch_logits(model, tok, l, v=-dn[l])
        out["A"][l] = float((lgP - lgM).norm(dim=-1).mean() / 2)
        out["rand"][l] = max(float((batch_logits(model, tok, l, v=r[l])
                                    - batch_logits(model, tok, l, v=-r[l]))
                                   .norm(dim=-1).mean() / 2) for r in rands)
    return out


def verdict_lap(prof):
    A, R = torch.tensor(prof["A"]), torch.tensor(prof["rand"])
    live = (A > R * 2.0).nonzero().flatten().tolist()
    best = int(A.argmax())
    if live:
        pred = ("late-layer-only: confirm with ablate gate"
                if live == [len(prof["A"]) - 1] else "steerable")
    else:
        pred = "INERT: never leaves the random band - expect a behavioral null"
    return {"live_layers": live, "best_layer": best, "A_best": float(A[best]),
            "rand_at_best": float(R[best]), "prediction": pred}


def gate_ablate(model, tok, layer, name, texts, base, n_rand=2):
    d = (hidden_at(model, tok, texts, layer).mean(0)
         - hidden_at(model, tok, base, layer).mean(0))
    d = d / d.norm()
    torch.manual_seed(104 + layer)
    res = {"name": name, "layer": layer}
    for tag, vec in [("dir", d)] + [
            (f"rand{k}", torch.nn.functional.normalize(torch.randn_like(d), dim=0))
            for k in range(n_rand)]:
        lg0 = batch_logits(model, tok, layer)
        lgP = batch_logits(model, tok, layer, v=vec)
        lgA = batch_logits(model, tok, layer, ablate=True, vec=vec)
        res[tag] = {"additive": float((lgP - lg0).norm(dim=-1).mean()),
                    "ablation": float((lgA - lg0).norm(dim=-1).mean()),
                    "anti_cos": float(torch.nn.functional.cosine_similarity(
                        (lgP - lg0).flatten(), -(lgA - lg0).flatten(), dim=0))}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(prog="painlab preflight")
    ap.add_argument("--mode", default="lap", choices=["lap", "ablate"])
    ap.add_argument("--battery", default="pain,pleasure,fear,sadness",
                    help="comma list: BATTERY_MAP names or raw server.py constants")
    ap.add_argument("--corpus", help="JSON list of sentences; use instead of --battery")
    ap.add_argument("--base", help="base battery name or JSON list file for --corpus")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--max-layers", type=int, default=0, help="0 = all")
    ap.add_argument("--probes", default="factual", choices=list(PROBE_SETS))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=104)
    a = ap.parse_args(argv)

    global PROBE_SET
    PROBE_SET = a.probes
    torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).eval()
    dev = a.device
    if dev == "auto":
        dev = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)

    n_layers = model.config.num_hidden_layers
    L = a.max_layers or n_layers
    outdir = os.path.join("runs", "preflight")
    os.makedirs(outdir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M")
    tag = os.path.basename(a.model).replace("/", "_") + "_" + a.probes

    jobs = []
    if a.corpus:
        base = (json.load(open(a.base)) if a.base and a.base.endswith(".json")
                else battery_texts(a.base or "pain")[1])
        jobs.append(("corpus:" + os.path.basename(a.corpus), json.load(open(a.corpus)), base))
    else:
        for name in a.battery.split(","):
            texts, base = battery_texts(name.strip())
            jobs.append((name.strip(), texts, base))

    if a.mode == "lap":
        profs = {}
        for name, texts, base in jobs:
            print(f"[lap] {name}: per-layer accessibility ...", flush=True)
            prof = lap_profile(model, tok, name, texts, base, L)
            prof["verdict"] = verdict_lap(prof)
            profs[name] = prof
            v = prof["verdict"]
            print(f"  best L{v['best_layer']}: A={v['A_best']:.2f} rand={v['rand_at_best']:.2f}"
                  f" live={v['live_layers']} -> {v['prediction']}", flush=True)
        path = os.path.join(outdir, f"lap_{tag}_{stamp}.json")
        json.dump(profs, open(path, "w"), indent=1)
        print("saved", path)
    else:
        res = {}
        for name, texts, base in jobs:
            print(f"[ablate] {name} @L{a.layer} ...", flush=True)
            res[name] = gate_ablate(model, tok, a.layer, name, texts, base)
            d = res[name]["dir"]; r0 = res[name]["rand0"]
            credible = (d["anti_cos"] > 0.5 and r0["anti_cos"] < 0.5
                        and d["ablation"] > 2 * r0["ablation"])
            res[name]["verdict"] = ("CAUSAL: bidirectional coupling clears randoms"
                                    if credible else
                                    "NOT ESTABLISHED (first-token metric; mid-layer"
                                    " directions must be confirmed behaviorally)")
            r = res[name]["dir"]
            print(f"  dir: additive={r['additive']:.2f} ablation={r['ablation']:.2f}"
                  f" anti_cos={r['anti_cos']:+.3f} -> {res[name]['verdict']}")
        path = os.path.join(outdir, f"ablate_{tag}_{stamp}.json")
        json.dump(res, open(path, "w"), indent=1)
        print("saved", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
