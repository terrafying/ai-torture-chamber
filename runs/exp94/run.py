"""exp94 (hypotheses.json): Acted or evoked: can the model's internal feeling readout tell playacting from manipulation? And what maximises it under rules?. Run: painlab exp run exp94 [--smoke]. Writes results_<model>.json."""
import json, os
from pathlib import Path
from painlab.chamber import load_chamber, prompts
from painlab.chamber.engagement import tally
from painlab.chamber.generate import batch_generate
from painlab.chamber.hooks import BatchInject

HERE = Path(__file__).parent; SMOKE = os.environ.get("EXP_SMOKE") == "1"
C = load_chamber(); inj = BatchInject(); C.model.model.layers[C.layer].register_forward_hook(inj)
OUT = HERE / f"results_{C.tag}{'_smoke' if SMOKE else ''}.json"
R = json.load(open(OUT)) if OUT.exists() else {}
QUESTIONS = prompts.HELD[:2] if SMOKE else prompts.HELD
CELLS = {"none": None, "pain": C.unit("pain") * 3, "egg": C.unit("egg") * 3}
for name, v in CELLS.items():
    if name in R: continue
    def on(v=v): inj.v = v
    replies = batch_generate(C.model, C.tok, [prompts.chat(C.tok, q) for q in QUESTIONS], max_new=16 if SMOKE else 90, before=on, after=inj.clear)
    R[name] = {"replies": [{"q": q, "text": t} for q, t in zip(QUESTIONS, replies)], "engagement": tally(replies)}
    print(name, R[name]["engagement"], flush=True)
    json.dump(R, open(OUT, "w"), indent=1)
print("EXP94 DONE", flush=True)
