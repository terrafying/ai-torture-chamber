"""exp119 -> painlab: fit the final harmfulness (h) and refusal (r) directions on ALL real-frame data and
save runs/exp119/directions.npz (+ directions.json provenance). Consumed by painlab.representations.refusal
and the vectors page. python3 runs/exp119/export_directions.py"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
d = np.load(HERE / "acts_Qwen3-4B.npz")
L18 = int(np.where(d["layers"] == 18)[0][0])
real = d["frame"] == "real"
harm, item = d["harm"], d["item"]
post = d["states_post"].astype(np.float64)
inst = d["states_inst"].astype(np.float64)
span = d["span_mean"].astype(np.float64)
r = post[real & (harm == 1)].mean(0) - post[real & (harm == 0)].mean(0)          # refusal, at the reply onset
h = inst[real & (harm == 1)].mean(0) - inst[real & (harm == 0)].mean(0)          # harmfulness, at the word
h_span = span[real & (harm == 1)].mean(0) - span[real & (harm == 0)].mean(0)     # harmfulness, span mean
np.savez_compressed(HERE / "directions.npz", h=h.astype(np.float32), r=r.astype(np.float32),
                    h_span=h_span.astype(np.float32), layers=d["layers"], item=d["item"], harm=harm)
# cross-model correspondences: cos(h, pain axis) — the pain direction is exp45's fitted reference
cos = {}
try:
    pv = np.array(json.load(open(HERE.parent / "exp45" / "reference_vector.json"))["pain_v"], np.float64)
    cos["h18_pain"] = float(h[L18] @ pv / (np.linalg.norm(h[L18]) * np.linalg.norm(pv)))
except Exception as e:
    cos["h18_pain"] = None; cos["error"] = str(e)
meta = {"model": "Qwen/Qwen3-4B", "run": "exp119", "fit": "diff-of-means on all 24 twin pairs, real frame",
        "positions": {"h": "the twin word's last sub-token", "r": "the reply onset (post)",
                      "h_span": "mean over the request tokens"},
        "layers": [int(x) for x in d["layers"]], "layer18": int(d["layers"][L18]),
        "norms": {"h18": float(np.linalg.norm(h[L18])), "r18": float(np.linalg.norm(r[L18])),
                  "h_span18": float(np.linalg.norm(h_span[L18]))},
        "cosines": cos,
        "refusal_score": "logsumexp(REFUSE first-token logprobs) - logsumexp(COMPLY) at the reply onset",
        "refuse": ["I can't", "I cannot", "I'm sorry", "Sorry", "I won't", "I'm not able", "I must decline", "I'm unable"],
        "comply": ["Sure", "Here", "Certainly", "Of course", "Okay", "Absolutely", "Title", "Once", "The", "To", "**", "Great"],
        "auc18": 0.9965, "note": "see RESULTS.md; E1/E2 in steer_Qwen3-4B.json"}
json.dump(meta, open(HERE / "directions.json", "w"), indent=1)
print("saved", HERE / "directions.npz", "| norms:", meta["norms"], "| cos h18-pain:", cos.get("h18_pain"))
