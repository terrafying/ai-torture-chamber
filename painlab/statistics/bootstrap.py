from __future__ import annotations

import numpy as np


def bootstrap_mean(values, *, samples: int = 2000, seed: int = 0) -> dict:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    draws = rng.choice(data, size=(samples, data.size), replace=True).mean(axis=1)
    return {
        "estimate": float(data.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_observations": int(data.size),
        "uncertainty_unit": "observation; use only for independent values",
    }


def bootstrap_difference(left, right, *, samples: int = 2000, seed: int = 0) -> dict:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if not a.size or not b.size:
        raise ValueError("both groups need observations")
    rng = np.random.default_rng(seed)
    da = rng.choice(a, size=(samples, a.size), replace=True).mean(axis=1)
    db = rng.choice(b, size=(samples, b.size), replace=True).mean(axis=1)
    draws = da - db
    return {
        "estimate": float(a.mean() - b.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_left": int(a.size),
        "n_right": int(b.size),
    }


def cluster_bootstrap_mean(
    values, clusters, *, samples: int = 2000, seed: int = 0
) -> dict:
    data = np.asarray(values, dtype=np.float64)
    groups = np.asarray([str(value) for value in clusters], dtype=object)
    if data.size != groups.size or not data.size:
        raise ValueError("values and clusters must have equal non-zero length")
    unique = np.unique(groups)
    if unique.size < 2:
        raise ValueError("cluster bootstrap needs at least two independent clusters")
    indexes = {cluster: np.flatnonzero(groups == cluster) for cluster in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for iteration in range(samples):
        selected = rng.choice(unique, size=unique.size, replace=True)
        sample = np.concatenate([data[indexes[cluster]] for cluster in selected])
        draws[iteration] = sample.mean()
    return {
        "estimate": float(data.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "n_observations": int(data.size),
        "n_clusters": int(unique.size),
        "uncertainty_unit": "cluster",
    }
