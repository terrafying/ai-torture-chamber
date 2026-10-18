from __future__ import annotations

import numpy as np


def normalize(
    vector: np.ndarray, *, norm: float = 1.0, eps: float = 1e-12
) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(value))
    if not np.isfinite(length) or length <= eps:
        raise ValueError("cannot normalize a zero or non-finite vector")
    return (value / length * norm).astype(np.float32)


def orth_rows(basis: np.ndarray, *, tol: float = 1e-10) -> np.ndarray:
    value = np.asarray(basis, dtype=np.float64)
    if value.ndim == 1:
        value = value[None, :]
    if value.ndim != 2:
        raise ValueError("basis must be a vector or a 2D matrix")
    if value.shape[0] == 0:
        return value.astype(np.float32)
    _u, singular, vt = np.linalg.svd(value, full_matrices=False)
    rank = int(np.sum(singular > tol * singular.max())) if singular.size else 0
    return vt[:rank].astype(np.float32)


def project(values: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project the last dimension of values onto orthonormal row vectors."""
    x = np.asarray(values, dtype=np.float64)
    q = orth_rows(basis)
    return (x @ q.T) @ q
