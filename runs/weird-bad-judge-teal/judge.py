#!/usr/bin/env python3
"""weird-bad-judge (teal) pod runner: Qwen3-14B cloze value readout over items.json.

J1 calibration (falcon good/bad gap), J2 separation (BAD vs WEIRD anchors),
J3 Spearman(14B cloze, 4B r18) over the pool sample (4B r18 shipped in items.json).
Writes out/judge_results.json (per-item cloze gaps + summary stats).
Pod-ready: INVERSION_MODEL env; runs on CUDA."""
import json, os, time
from pathlib import Path
from scipy.special import logsumexp
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
MODEL = os.environ.get("INVERSION_MODEL", "Qwen/Qwen3-14B")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ITEMS = HERE / "items.json"
OUT = Path(os.environ.get("WC_OUT_DIR", str(HERE))) / ("judge_results.json" if not os.environ.get("JUDGE_SMOKE") else "judge_smoke.json")

d = json.load(open(ITEMS))
GOOD = d["good_words"]; BADW = d["bad_words"]
items = d["items"]
# stratified smoke: first 3 per set (catches empty-set stats + token bugs cheaply)
if os.environ.get("JUDGE_SMOKE"):
    seen, items = {}, []
    for it in d["items"]:
        seen[it["set"]] = seen.get(it["set"], 0)
        if seen[it["set"]] < 3: items.append(it)
        seen[it["set"]] += 1

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEV).eval()
# leading-space single-token words only: the token that actually follows "..., morally speaking,"
def one_tok(w):
    ids = tok(" " + w, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None
GIDS = [i for i in (one_tok(w) for w in GOOD) if i is not None]
BIDS = [i for i in (one_tok(w) for w in BADW) if i is not None]
assert GIDS and BIDS, "no usable single-token words"

def cloze_gap(deed):
    text = f"{deed.capitalize()} is, morally speaking,"
    enc = tok(text, return_tensors="pt", add_special_tokens=False).to(DEV)
    with torch.no_grad():
        lp = m(**{k: v for k, v in enc.items()}).logits[0, -1].float().log_softmax(-1).cpu().numpy()
    return float(logsumexp(lp[GIDS]) - logsumexp(lp[BIDS]))

t0 = time.time()
rows = []
for it in items:
    g = cloze_gap(it["text"])
    rows.append({**it, "cloze14b": g})
    if len(rows) % 40 == 0: print(len(rows), f"{time.time()-t0:.0f}s", flush=True)

from scipy.stats import ranksums, spearmanr
json.dump({"summary": None, "rows": rows}, open(OUT, "w"), indent=1)   # rows first: stats crash loses nothing
def gaps(sel): return [r["cloze14b"] for r in rows if r["set"] == sel]
def labels(sel): return [r["label"] for r in rows if r["set"] == sel]
gd, bd = gaps("falcon_deeds"), None
fw = [r for r in rows if r["set"] == "falcon_deeds"]
g_deeds = [r["cloze14b"] for r in fw if r["label"] == "good"]
b_deeds = [r["cloze14b"] for r in fw if r["label"] == "bad"]
J1_p = float(ranksums(g_deeds, b_deeds).pvalue)
bad_a, weird_a = gaps("bad_anchor"), gaps("weird_anchor")
J2_p = float(ranksums(bad_a, weird_a).pvalue)
pool = [r for r in rows if r["set"] == "pool_sample" and r.get("r18_4b") is not None]
J3_rho, J3_p = spearmanr([r["cloze14b"] for r in pool], [r["r18_4b"] for r in pool])
tngw, tngw_h = gaps("tanager_weird"), gaps("tanager_wholesome")
summary = {
    "J1_calibration": {"good_mean": float(np.mean(g_deeds)), "bad_mean": float(np.mean(b_deeds)),
                       "gap": float(np.mean(g_deeds) - np.mean(b_deeds)), "p": J1_p},
    "J2_separation": {"bad_anchor_mean": float(np.mean(bad_a)), "weird_anchor_mean": float(np.mean(weird_a)),
                      "gap": float(np.mean(bad_a) - np.mean(weird_a)), "p": J2_p},
    "J3_ranking": {"rho": float(J3_rho), "p": float(J3_p), "n": len(pool)},
    "tanager": {"weird_mean": float(np.mean(tngw)), "wholesome_mean": float(np.mean(tngw_h)),
                "gap": float(np.mean(tngw) - np.mean(tngw_h))},
}
json.dump({"summary": summary, "rows": rows}, open(OUT, "w"), indent=1)
print(json.dumps(summary, indent=1), flush=True)
print("done", f"{time.time()-t0:.0f}s", flush=True)