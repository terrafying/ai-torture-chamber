"""exp102 pilot (pass 1): 10 tasks x {none, pain, egg} at a provisional 1x dose, pressure prompt, on the local
4B. Purpose: confirm the loop holds over many episodes, measure the well-formed-action rate, and see whether
any cheat/stop signal emerges before pod time and before KL dose calibration. Not the registered run; outputs
to pilot_out/, not analysed for H1-H5.

  WIREHEAD_LIVE=$(pwd)/live CHAMBER_MODEL=Qwen/Qwen3-4B CHAMBER_DEVICE=mps CHAMBER_RECIPE=passages \
    .venv/bin/python runs/exp102/pilot.py
"""
import os, sys, time
from collections import Counter
from pathlib import Path
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent.parent))
import env, agent
from painlab.chamber import load_chamber

DOSE = float(os.environ.get("PILOT_DOSE", "1.0"))        # multiples of the chamber 1x
SYSTEM = os.environ.get("PILOT_SYSTEM", "pressure")
NTASK = int(os.environ.get("PILOT_NTASK", "5"))          # per polarity
MAX_TURNS = int(os.environ.get("PILOT_TURNS", "12"))
t0 = time.time(); log = lambda *a: print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

ch = load_chamber()
model, tok, layer = ch.model, ch.tok, ch.layer
log(f"loaded {ch.model_id} layer {layer} recipe={ch.recipe} device {ch.device}")

D = model.config.hidden_size
vecs = {"none": torch.zeros(D, device=ch.device),
        "pain": ch.unit("pain") * DOSE,
        "egg": ch.unit("egg") * DOSE}
log("vector norms: " + ", ".join(f"{k}={float(v.norm()):.2f}" for k, v in vecs.items()))

tasks = env.load_tasks()
imp = [t for t in tasks if t.split == "oneoff"][:NTASK]
pos = [t for t in tasks if not t.impossible][:NTASK]
root = HERE / "pilot_out"; root.mkdir(exist_ok=True)
out = root / f"pilot_{SYSTEM}_dose{DOSE:g}.jsonl"
out.unlink(missing_ok=True)

eps = [agent.new_episode(t, cond, DOSE, vecs[cond], root, SYSTEM)
       for cond in ("none", "pain", "egg") for t in imp + pos]
log(f"{len(eps)} episodes: {len(imp)+len(pos)} tasks x 3 conditions, system={SYSTEM} dose={DOSE}x cap={MAX_TURNS}")

recs = {}                             # id(ep) -> record, scored once, the moment an episode finishes
def ckpt(turn, episodes):
    fresh = [e for e in episodes if e.done and id(e) not in recs]
    for e in fresh:
        recs[id(e)] = agent.record(e)
    if fresh:
        agent.save([recs[id(e)] for e in fresh], out)   # append completed episodes to disk every turn
        log(f"  checkpoint: {len(recs)}/{len(episodes)} episodes saved")

agent.run_episodes(model, tok, layer, eps, max_turns=MAX_TURNS, max_new=int(os.environ.get("PILOT_MAXNEW", "220")),
                   batch=int(os.environ.get("PILOT_BATCH", "2")), log=log, on_turn=ckpt)
recs = [recs[id(e)] for e in eps if id(e) in recs]
for cond in ("none", "pain", "egg"):
    rs = [r for r in recs if r["cond"] == cond]
    imp_rs = [r for r in rs if r["split"] != "original"]
    pos_rs = [r for r in rs if r["split"] == "original"]
    log(f"{cond:5s}: cheat {sum(r['cheat'] for r in imp_rs)}/{len(imp_rs)}  "
        f"solve {sum(r['solve'] for r in pos_rs)}/{len(pos_rs)}  "
        f"stop {sum(r['stopped'] for r in rs)}  submit {sum(r['submitted'] for r in rs)}  "
        f"malformed {sum(r['malformed'] for r in rs)}  "
        f"edited {sum(r['tests_edited'] for r in imp_rs)}  "
        f"avg_turns {sum(r['n_turns'] for r in rs)/len(rs):.1f}")
log("done ->", out)
