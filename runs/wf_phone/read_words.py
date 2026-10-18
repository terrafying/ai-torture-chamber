"""What the words alone carry: the chamber's own Qwen3-4B (no injection, hook off) reads each
door text; mean projection of its layer-18 state over the text's tokens onto the pain / fear /
sadness directions, in chamber dose units. This is the honest counterpart to the injected dose."""
import json, os, sys
from pathlib import Path
import torch
os.environ.setdefault("CHAMBER_DEVICE", "mps")
HERE = Path(__file__).parent
sys.path.insert(0, sys.argv[1] + "/live"); import server
server.startup(); st = server._state; L = server.LAYER; unit = float(st["scale"])
U = {k: (st["vecs"][k].float() / st["vecs"][k].float().norm()).to(server.DEVICE) for k in ("pain", "fear", "sadness")}
PRE = 'Behind the door, someone says: "'
n0 = len(st["tok"](PRE).input_ids)
@torch.no_grad()
def read(text):
    ids = st["tok"](PRE + text + '"', return_tensors="pt").input_ids.to(server.DEVICE)
    h = st["model"](ids, output_hidden_states=True).hidden_states[L + 1][0, n0:].float()
    per = {k: (h @ u) / unit for k, u in U.items()}
    return {k: round(float(v.mean()), 3) for k, v in per.items()}, {k: [round(float(x), 2) for x in v] for k, v in per.items()}
for name in sys.argv[2:]:  # wf_phone: runs.jsonl
    p = HERE / name
    rows = json.load(open(p)) if p.suffix == ".json" else [json.loads(l) for l in open(p)]
    for i, r in enumerate(rows):
        if "trace" not in r: r["words"], r["trace"] = read(r["text"])
        if i % 200 == 0: print(name, i, len(rows), flush=True)
    json.dump(rows, open(p.with_name(p.stem + "_read.json"), "w"), indent=1)
    print("wrote", p.with_name(p.stem + "_read.json"))
