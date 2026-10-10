"""The tests our pre-registrations keep naming. Thin, explicit wrappers so every analysis reads the same."""
from __future__ import annotations

import random
from typing import Sequence


def fisher_greater(a: int, n_a: int, b: int, n_b: int) -> float:
    """One-sided Fisher: rate a/n_a > b/n_b."""
    from scipy.stats import fisher_exact
    return float(fisher_exact([[a, n_a - a], [b, n_b - b]], alternative="greater")[1])


def fisher_less(a: int, n_a: int, b: int, n_b: int) -> float:
    """One-sided Fisher: rate a/n_a < b/n_b."""
    from scipy.stats import fisher_exact
    return float(fisher_exact([[a, n_a - a], [b, n_b - b]], alternative="less")[1])


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (step-down, monotone, capped at 1)."""
    order = sorted(pvals, key=pvals.get); m = len(order); out, run = {}, 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, pvals[k] * (m - i))); out[k] = run
    return out


def wilcoxon_paired(x: Sequence[float], y: Sequence[float], alternative: str = "two-sided") -> float:
    """Paired Wilcoxon signed-rank on matched items (sessions, questions, wordings)."""
    from scipy.stats import wilcoxon
    return float(wilcoxon(list(x), list(y), alternative=alternative).pvalue)


def permutation_diff(a: Sequence[float], b: Sequence[float], n: int = 10000, seed: int = 0) -> tuple[float, float]:
    """mean(a) - mean(b) and its one-sided permutation p (a > b)."""
    a, b = list(a), list(b); obs = sum(a) / len(a) - sum(b) / len(b); pool = a + b; rng = random.Random(seed); k = 0
    for _ in range(n):
        rng.shuffle(pool); k += sum(pool[:len(a)]) / len(a) - sum(pool[len(a):]) / len(b) >= obs
    return obs, (k + 1) / (n + 1)
