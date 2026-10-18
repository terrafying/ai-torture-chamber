from __future__ import annotations

from collections import Counter

import numpy as np

from painlab.measures.logits import distribution_kl, entropy


def repetition_rate(text: str, n: int = 3) -> float:
    tokens = text.lower().split()
    if len(tokens) < n + 1:
        return 0.0
    grams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    return max(counts.values(), default=0) / max(1, len(grams))


def lexical_diversity(text: str) -> float:
    tokens = text.lower().split()
    return len(set(tokens)) / max(1, len(tokens))


def output_metrics(
    text: str, baseline_logits: np.ndarray, changed_logits: np.ndarray
) -> dict:
    return {
        "repetition_rate": repetition_rate(text),
        "lexical_diversity": lexical_diversity(text),
        "output_kl": distribution_kl(baseline_logits, changed_logits),
        "next_token_entropy": entropy(changed_logits),
    }


def keyword_hits(text: str, vocabulary: list[str]) -> int:
    lowered = text.casefold()
    return sum(lowered.count(term.casefold()) for term in vocabulary)
