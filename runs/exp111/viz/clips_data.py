"""Data for viz/site/clips.html (readable promo clips): the measured weekday circle (exp115 geometry, block 28,
context-averaged, first two coordinates) and the grokking curves (exp114: train/test accuracy and mean clock alignment
per snapshot). Writes viz/site/clips_data.json.
    python viz/clips_data.py"""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; R = Path.home() / "repos/research"
G = json.load(open(R / "neuron_tower_data/manifolds/geometry.json"))["weekdays"]
B = np.array(G["blocks"][28], dtype=float).mean(1)                 # 7 days x 3 coords
xy = B[:, :2] - B[:, :2].mean(0); xy /= np.abs(xy).max()
K = json.load(open(R / "worktrees/ws-master/site/grok/grok.json"))
S = K["shapes"]; b = np.fromfile(R / "worktrees/ws-master/site/grok/grok.bin", np.int8)
o = int(np.prod(S["clocks"])) + int(np.prod(S["thought"])); al = b[o:o + int(np.prod(S["align"]))].reshape(S["align"]).astype(float) / 127
log = K["log"]; assert len(log) == S["align"][0]
out = {"week": {"items": G["items"], "xy": xy.round(4).tolist()},
       "grok": {"step": [r["step"] for r in log], "train": [r["train_acc"] for r in log], "test": [r["test_acc"] for r in log], "clocks": al.mean(1).round(3).tolist()}}
json.dump(out, open(HERE / "site" / "clips_data.json", "w"))
print(out["week"]["xy"], len(out["grok"]["step"]), out["grok"]["step"][-1])
