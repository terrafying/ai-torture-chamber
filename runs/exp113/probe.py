"""exp113 (exploratory): probe one neuron with hand-written texts. Prints each text's max activation / corpus max and
the token where it peaks. Texts come from a JSON file {"label": [texts...], ...}.
    HF_HUB_CACHE=... python runs/exp113/probe.py <floor> <neuron> <texts.json>"""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE.parent / "exp111"))
from tower_cfg import DATA as D, DEV, load
l, n, F = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
MX = float(np.load(D / "corpus_all.npz")["max"][l, n]); tok, m = load(); cur = {}
m.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda mod, inp: cur.__setitem__("a", inp[0][0, :, n].float()))
for group, texts in json.load(open(F)).items():
    print(f"== {group}")
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        with torch.no_grad(): m(torch.tensor([ids], device=DEV))
        a = cur["a"]; j = int(a.argmax())
        print(f"  {float(a[j]) / MX:5.2f}  at {tok.decode([ids[j]])!r:14s} {t[:90]}")
