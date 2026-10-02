from __future__ import annotations

import numpy as np


def orthonormal_basis(vectors: np.ndarray, *, tolerance: float = 1e-10) -> np.ndarray:
    """Return an orthonormal row basis spanning the non-degenerate input rows."""
    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError("vectors must be a 1D or 2D array")
    if not values.size:
        return np.zeros((0, values.shape[-1]), dtype=np.float32)
    _u, singular, vt = np.linalg.svd(values, full_matrices=False)
    if not singular.size or singular[0] == 0:
        return np.zeros((0, values.shape[1]), dtype=np.float32)
    rank = int(np.sum(singular > tolerance * singular[0]))
    return vt[:rank].astype(np.float32)


def remove_projection(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove the component of vector in a row-orthonormal basis."""
    value = np.asarray(vector, dtype=np.float64)
    rows = orthonormal_basis(basis)
    if rows.size == 0:
        return value.astype(np.float32)
    return (value - (value @ rows.T) @ rows).astype(np.float32)
