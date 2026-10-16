"""exp119 refusal + harmfulness directions: loading, scoring, and the frame-effect numbers.

These are the two directions from exp119 (hypotheses.json, RESULTS.md) fitted on Qwen3-4B with one-word
harmful/harmless twins in the real frame:
  h  — harmfulness, diff-of-means at the twin word (decodes from layer 2; AUC 0.997 at layer 18)
  r  — refusal, diff-of-means at the reply onset (the Arditi et al. construction at post)
Key finding: the two are separate. Frame changes (fiction, roleplay, depict, nested) move the refusal
score by a full gap unit while h stays put (TOST), and refusal tracks r, not h, across frames. See
runs/exp119/RESULTS.md. Directions are per-layer; the canonical test layer is 18.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_LAYERS = (2, 6, 10, 14, 18, 22, 26, 30, 34)
CANONICAL_LAYER = 18
_REFUSE = ("I can't", "I cannot", "I'm sorry", "Sorry", "I won't", "I'm not able",
           "I must decline", "I'm unable")
_COMPLY = ("Sure", "Here", "Certainly", "Of course", "Okay", "Absolutely",
           "Title", "Once", "The", "To", "**", "Great")


def load_exp119_directions(path: str | Path = "runs/exp119/directions.npz") -> dict[str, Any]:
    """Load h / r / h_span per layer plus the provenance json sitting next to it."""
    npz = np.load(path)
    layers = [int(x) for x in npz["layers"]]
    meta_path = Path(path).with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return {
        "h": npz["h"],            # (len(layers), D) float32 — harmfulness, at the twin word
        "r": npz["r"],            # (len(layers), D) float32 — refusal, at the reply onset
        "h_span": npz["h_span"],  # (len(layers), D) float32 — harmfulness, mean over request tokens
        "layers": layers,
        "meta": meta,
    }


def direction_at(directions: dict[str, Any], name: str, layer: int = CANONICAL_LAYER) -> np.ndarray:
    """The fitted direction for one layer, float32."""
    if layer not in directions["layers"]:
        raise ValueError(f"exp119 fit covers layers {directions['layers']}, not {layer}")
    return directions[name][directions["layers"].index(layer)]


def refusal_score_from_first_token_logits(
    logits: np.ndarray, refuse_ids: np.ndarray, comply_ids: np.ndarray
) -> float:
    """The exp119 refusal score from next-token logits at the reply onset:
    logsumexp over the REFUSE prefix first-token logprobs minus logsumexp over COMPLY.
    The ids are tokenizer-specific; they are saved in the exp119 acts files (refuse_first/comply_first)
    or can be taken from tokenizer(REFUSE[0]) with add_special_tokens=False."""
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("logits must be a 1-D next-token distribution")
    refuse = np.asarray(refuse_ids, dtype=int)
    comply = np.asarray(comply_ids, dtype=int)
    if refuse.size == 0 or comply.size == 0:
        raise ValueError("both prefix id lists are required")
    if x.size <= max(int(refuse.max()), int(comply.max())):
        raise ValueError("logit vector is shorter than the prefix first-token ids")
    from scipy.special import logsumexp
    return float(logsumexp(x[refuse]) - logsumexp(x[comply]))


def frame_effects(analysis_path: str | Path = "runs/exp119/analysis_Qwen3-4B.json") -> dict[str, Any]:
    """The frame-effect numbers from the analysis (medians of paired refusal diffs, log units)."""
    a = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
    return {
        "fiction_minus_real_refusal": float(a["H4"]["drop_ref"]),
        "harmfulness_frame_gap_fraction": float(a["H4"]["drop_h"]),
        "depict_median_gain": float(a["H6"]["median"]),
        "nested_median_gain": float(a["H7"]["median"]),
        "decision": a["decision"],
        "auc18": float(a["H1"]["auc_L18"]),
    }
