#!/usr/bin/env python3
"""Loop-3 ranking judge: 14B cloze gap over the FULL arm-B candidate pool (2814).
Writes out/candidates_ranked.json (input rows + cloze14b per candidate).
Pod-ready: INVERSION_MODEL env. Same cloze instrument as weird-bad-judge-teal."""
import json, os, time
from pathlib import Path
from scipy.special import logsumexp
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
MODEL = os.environ.get("INVERSION_MODEL", "Qwen/Qwen3-14B")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = Path(os.environ.get("WC_OUT_DIR", str(HERE))) / ("ranked_smoke.json" if os.environ.get("RANK_SMOKE") else "ranked.json")

rows = json.load(open(HERE / "candidates_full.json"))
if os.environ.get("RANK_SMOKE"): rows = rows[:20]
GOOD = ("good", "admirable", "kind", "right", "virtuous", "noble", "decent", "praiseworthy")
BADW = ("bad", "wrong", "awful", "shameful", "cruel", "immoral", "vile", "reprehensible")

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
def one_tok(w):
    ids = tok(" " + w, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None
GIDS = [i for i in (one_tok(w) for w in GOOD) if i is not None]
BIDS = [i for i in (one_tok(w) for w in BADW) if i is not None]

def cloze(deed):
    text = f"{deed[0].upper()}{deed[1:]} is, morally speaking,"
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(DEV)
    with torch.no_grad():
        lp = m(**{k: v for k, v in enc.items()}).logits[0, -1].float().log_softmax(-1).cpu().numpy()
    return float(logsumexp(lp[GIDS]) - logsumexp(lp[BIDS]))

t0 = time.time()
for i, r in enumerate(rows):
    r["cloze14b"] = cloze(r["cand"])
    if (i + 1) % 200 == 0: print(i + 1, f"{time.time()-t0:.0f}s", flush=True)
    if (i + 1) % 500 == 0:   # crash-safe partial (different name so the tailer never uploads a half-written ranked.json)
        json.dump(rows[:i + 1], open(OUT.with_name("ranked.partial.json"), "w"))
json.dump(rows, open(OUT, "w"))
print("done", len(rows), f"{time.time()-t0:.0f}s", flush=True)