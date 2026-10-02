from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from painlab.representations.directions import normalize
from painlab.representations.subspaces import orthonormal_basis, remove_projection


@dataclass
class ControlVector:
    name: str
    vector: np.ndarray
    method: str
    target_norm: float
    metadata: dict


def random_norm_matched(target: np.ndarray, seed: int) -> ControlVector:
    value = np.asarray(target, dtype=np.float32)
    vector = normalize(
        np.random.default_rng(seed).normal(size=value.shape),
        norm=float(np.linalg.norm(value)),
    )
    return ControlVector(
        "random_norm_matched",
        vector,
        "isotropic_gaussian",
        float(np.linalg.norm(value)),
        {"seed": seed},
    )


def covariance_matched(
    target: np.ndarray, activations: np.ndarray, seed: int
) -> ControlVector:
    value = np.asarray(target, dtype=np.float64)
    samples = np.asarray(activations, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != value.size or samples.shape[0] < 2:
        raise ValueError("activations must be (at least 2 examples, target width)")
    centered = samples - samples.mean(axis=0, keepdims=True)
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    eig = singular**2 / max(1, len(samples) - 1)
    rank = int(np.count_nonzero(eig > 1e-12))
    if rank == 0:
        raise ValueError("activation covariance has zero rank")
    rng = np.random.default_rng(seed)
    draw = (rng.normal(size=rank) * np.sqrt(eig[:rank])) @ vt[:rank]
    vector = normalize(draw, norm=float(np.linalg.norm(value)))
    return ControlVector(
        "covariance_matched",
        vector,
        "empirical_covariance_gaussian",
        float(np.linalg.norm(value)),
        {"seed": seed, "empirical_rank": rank},
    )


def orthogonal_random(
    target: np.ndarray, *, against: np.ndarray | None = None, seed: int = 0
) -> ControlVector:
    value = np.asarray(target, dtype=np.float32)
    basis = orthonormal_basis(value[None, :] if against is None else against)
    rng = np.random.default_rng(seed)
    draw = rng.normal(size=value.shape).astype(np.float32)
    if basis.size:
        draw = remove_projection(draw, basis)
    vector = normalize(draw, norm=float(np.linalg.norm(value)))
    return ControlVector(
        "orthogonal_random",
        vector,
        "gaussian_projected_out",
        float(np.linalg.norm(value)),
        {"seed": seed, "basis_rank": len(basis)},
    )


def shuffled_coordinates(target: np.ndarray, seed: int = 0) -> ControlVector:
    value = np.asarray(target, dtype=np.float32)
    shuffled = value[np.random.default_rng(seed).permutation(value.size)]
    return ControlVector(
        "shuffled_coordinates",
        shuffled.copy(),
        "coordinate_permutation",
        float(np.linalg.norm(value)),
        {"seed": seed},
    )


def semantic_controls(
    target: np.ndarray, directions: Mapping[str, np.ndarray]
) -> dict[str, ControlVector]:
    norm = float(np.linalg.norm(target))
    result = {}
    for name, direction in directions.items():
        vector = normalize(np.asarray(direction), norm=norm)
        result[name] = ControlVector(
            name, vector, "provided_semantic_direction", norm, {}
        )
    return result


def standard_controls(
    target: np.ndarray,
    *,
    activations: np.ndarray | None = None,
    seed: int = 0,
    against: np.ndarray | None = None,
) -> dict[str, ControlVector]:
    result = {
        "random_norm_matched": random_norm_matched(target, seed),
        "orthogonal_random": orthogonal_random(target, against=against, seed=seed + 1),
        "shuffled_coordinates": shuffled_coordinates(target, seed + 2),
    }
    if activations is not None:
        result["covariance_matched"] = covariance_matched(target, activations, seed + 3)
    return result


def match_perturbation(
    target_metric: float,
    measure: Callable[[float], float],
    *,
    strengths: list[float],
) -> dict:
    """Choose the tested strength whose measured perturbation is closest to target."""
    if not strengths:
        raise ValueError("strength grid must be non-empty")
    rows = [
        {"strength": float(dose), "metric": float(measure(float(dose)))}
        for dose in strengths
    ]
    best = min(rows, key=lambda row: abs(row["metric"] - target_metric))
    return {
        "target_metric": float(target_metric),
        "matched_strength": best["strength"],
        "matched_metric": best["metric"],
        "absolute_error": abs(best["metric"] - target_metric),
        "grid": rows,
    }
