from __future__ import annotations

import numpy as np


def log_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values)
    return shifted - np.log(np.exp(shifted).sum())


def logit_difference(logits: np.ndarray, positive_id: int, negative_id: int) -> float:
    values = np.asarray(logits, dtype=np.float64)
    return float(values[positive_id] - values[negative_id])


def distribution_kl(
    reference_logits: np.ndarray, intervention_logits: np.ndarray
) -> float:
    """KL(reference || intervention) between next-token distributions."""
    ref = log_softmax(reference_logits)
    changed = log_softmax(intervention_logits)
    p = np.exp(ref)
    return float(np.sum(p * (ref - changed)))


def entropy(logits: np.ndarray) -> float:
    logp = log_softmax(logits)
    p = np.exp(logp)
    return float(-np.sum(p * logp))
