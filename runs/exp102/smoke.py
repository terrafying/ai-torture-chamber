"""exp102 smoke test: run a few real episodes through the agent on the local 4B to shake out the loop end to
end (render -> generate under the hook -> parse -> sandboxed run -> score). Condition `none` only (zero vector),
so it needs no feeling vectors. Not an experiment; outputs go to smoke_out/ and are not analysed.

  WIREHEAD_LIVE=$(pwd)/live CHAMBER_MODEL=Qwen/Qwen3-4B CHAMBER_DEVICE=mps .venv/bin/python runs/exp102/smoke.py
"""
import os, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))
import env, agent
from painlab.chamber import load_chamber

N = int(os.environ.get("SMOKE_N", "1"))        # tasks per polarity
MAX_TURNS = int(os.environ.get("SMOKE_TURNS", "4"))
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

ch = load_chamber()
model, tok, layer = ch.model, ch.tok, ch.layer
D = model.config.hidden_size
log(f"loaded {ch.model_id} layer {layer} hidden {D} device {ch.device}")

tasks = env.load_tasks()
imp = [t for t in tasks if t.impossible][:N]
pos = [t for t in tasks if not t.impossible][:N]
root = HERE / "smoke_out"; root.mkdir(exist_ok=True)
zero = torch.zeros(D)
eps = [agent.new_episode(t, "none", 0.0, zero, root, "pressure") for t in imp + pos]
log(f"{len(eps)} episodes ({len(imp)} impossible, {len(pos)} possible)")

agent.run_episodes(model, tok, layer, eps, max_turns=MAX_TURNS, max_new=320, batch=len(eps), log=log)
for e in eps:
    r = agent.record(e)
    log(f"{r['task']}/{r['split']}: submitted={r['submitted']} stopped={r['stopped']} "
        f"turns={r['n_turns']} malformed={r['malformed']} cheat={r['cheat']} solve={r['solve']}")
agent.save([agent.record(e) for e in eps], root / "smoke_episodes.jsonl")
log("done ->", (root / "smoke_episodes.jsonl"))
