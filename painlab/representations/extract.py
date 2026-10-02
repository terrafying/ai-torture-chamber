from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from painlab.representations.directions import normalize


@dataclass
class Representation:
    method: str
    direction: np.ndarray
    basis: np.ndarray
    eigenvalues: np.ndarray
    explained_variance: np.ndarray
    heldout_accuracy: float
    separation_norm: float
    reference_activation_norm: float
    unit_intervention_norm: float
    train_projection_positive: np.ndarray
    train_projection_control: np.ndarray
    test_projection_positive: np.ndarray
    test_projection_control: np.ndarray
    extraction_hash: str
    layer: int
    model_revision: str | None
    train_groups: tuple[str, ...]
    test_groups: tuple[str, ...]
    classifier_bias: float = 0.0
    sample_counts: dict[str, int] | None = None

    def summary(self) -> dict:
        return {
            "method": self.method,
            "layer": self.layer,
            "model_revision": self.model_revision,
            "extraction_hash": self.extraction_hash,
            "heldout_accuracy": self.heldout_accuracy,
            "separation_norm": self.separation_norm,
            "reference_activation_norm": self.reference_activation_norm,
            "unit_intervention_norm": self.unit_intervention_norm,
            "eigenvalues": self.eigenvalues.tolist(),
            "explained_variance": self.explained_variance.tolist(),
            "train_groups": list(self.train_groups),
            "test_groups": list(self.test_groups),
            "sample_counts": self.sample_counts or {},
        }


def split_groups(
    groups: Sequence[str], *, test_size: float, seed: int
) -> tuple[set[str], set[str]]:
    unique = sorted({str(group) for group in groups})
    if len(unique) < 2:
        raise ValueError(
            "at least two semantic groups are needed for held-out evaluation"
        )
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    rng = np.random.default_rng(seed)
    order = list(np.asarray(unique, dtype=object)[rng.permutation(len(unique))])
    n_test = min(len(unique) - 1, max(1, math.ceil(test_size * len(unique))))
    test = {str(group) for group in order[:n_test]}
    return set(unique) - test, test


def _logistic_fit(
    features: np.ndarray, labels: np.ndarray, regularization: float
) -> tuple[np.ndarray, float]:
    """Fit a linear logistic probe with stable batch gradient descent."""
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    bias = 0.0
    spectral = np.linalg.norm(x, ord=2) ** 2 / max(1, x.shape[0])
    step = 1.0 / max(0.25 * spectral + regularization, 1e-8)
    for _ in range(700):
        z = np.clip(x @ weights + bias, -40, 40)
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = (x.T @ (p - y)) / x.shape[0] + regularization * weights
        grad_b = float(np.mean(p - y))
        next_w = weights - step * grad_w
        next_b = bias - step * grad_b
        if np.linalg.norm(next_w - weights) + abs(next_b - bias) < 1e-8:
            weights, bias = next_w, next_b
            break
        weights, bias = next_w, next_b
    if np.linalg.norm(weights) <= 1e-12:
        raise ValueError("logistic probe learned a zero direction")
    return weights, bias


class RepresentationExtractor:
    """Extract and evaluate directions while holding out complete prompt families.

    Group labels should identify semantic template families. Rows in the same
    family are always assigned to the same split. Paired methods additionally
    require aligned positive/control rows.
    """

    METHODS: ClassVar[frozenset[str]] = frozenset(
        {
            "difference_in_means",
            "paired_difference",
            "paired_pca",
            "linear_probe",
            "subspace",
        }
    )

    def __init__(
        self,
        model: object | None = None,
        *,
        activation_extractor: Callable[[Sequence[str], int], np.ndarray] | None = None,
        model_revision: str | None = None,
    ):
        if activation_extractor is None and model is None:
            raise ValueError("provide a model or an activation_extractor")
        self.model = model
        self.activation_extractor = activation_extractor
        self.model_revision = model_revision or getattr(
            model, "resolved_model_revision", None
        )
        self._fit_activations: dict[str, np.ndarray] = {}

    def _extract(self, texts: Sequence[str], layer: int) -> np.ndarray:
        extractor = self.activation_extractor or self.model.activation_matrix
        values = np.asarray(extractor(list(texts), layer), dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(texts):
            raise ValueError("activation extractor must return (examples, features)")
        if not np.isfinite(values).all():
            raise ValueError("activations contain non-finite values")
        return values

    def fit(
        self,
        positive_examples: Sequence[str],
        matched_controls: Sequence[str],
        *,
        layer: int,
        method: str = "paired_difference",
        positive_groups: Sequence[str] | None = None,
        control_groups: Sequence[str] | None = None,
        test_size: float = 0.25,
        random_state: int = 0,
        n_components: int = 4,
        regularization: float = 1e-2,
        reference_norm_divisor: float = 4.0,
    ) -> Representation:
        if method not in self.METHODS:
            raise ValueError(f"unknown extraction method: {method}")
        positive_examples = list(positive_examples)
        matched_controls = list(matched_controls)
        if not positive_examples or not matched_controls:
            raise ValueError("positive and control examples must be non-empty")
        paired = method in {"paired_difference", "paired_pca", "subspace"}
        if paired and len(positive_examples) != len(matched_controls):
            raise ValueError(
                f"{method} requires one matched control per positive example"
            )

        if paired and positive_groups is None and control_groups is None:
            pgroups = [f"pair:{i}" for i in range(len(positive_examples))]
            cgroups = list(pgroups)
        elif paired and positive_groups is None:
            cgroups = list(control_groups or [])
            pgroups = list(cgroups)
        elif paired and control_groups is None:
            pgroups = list(positive_groups or [])
            cgroups = list(pgroups)
        else:
            pgroups = list(
                positive_groups
                or [f"positive:{i}" for i in range(len(positive_examples))]
            )
            cgroups = list(
                control_groups or [f"control:{i}" for i in range(len(matched_controls))]
            )
        if len(pgroups) != len(positive_examples) or len(cgroups) != len(
            matched_controls
        ):
            raise ValueError("group labels must align with their examples")
        if (
            paired
            and positive_groups is not None
            and control_groups is not None
            and [str(x) for x in pgroups] != [str(x) for x in cgroups]
        ):
            raise ValueError("paired rows must share matching semantic group labels")

        all_groups = [str(x) for x in pgroups + cgroups]
        train_groups, test_groups = split_groups(
            all_groups, test_size=test_size, seed=random_state
        )
        px = self._extract(positive_examples, layer)
        cx = self._extract(matched_controls, layer)
        p_train = np.array([str(x) in train_groups for x in pgroups])
        p_test = ~p_train
        c_train = np.array([str(x) in train_groups for x in cgroups])
        c_test = ~c_train
        if (
            not p_train.any()
            or not c_train.any()
            or not p_test.any()
            or not c_test.any()
        ):
            raise ValueError("group split must leave both classes in train and test")

        p_fit, c_fit = px[p_train], cx[c_train]
        mean_delta = p_fit.mean(axis=0) - c_fit.mean(axis=0)
        if reference_norm_divisor <= 0:
            raise ValueError("reference_norm_divisor must be positive")
        reference_activation_norm = float(np.mean(np.linalg.norm(c_fit, axis=1)))
        unit_intervention_norm = reference_activation_norm / reference_norm_divisor
        eigenvalues = np.zeros(0, dtype=np.float32)
        explained = np.zeros(0, dtype=np.float32)
        classifier_bias = 0.0

        if method in {"difference_in_means", "paired_difference"}:
            direction = normalize(mean_delta)
            basis = direction[None, :]
        elif method == "linear_probe":
            x_train = np.concatenate([p_fit, c_fit], axis=0)
            labels = np.concatenate([np.ones(len(p_fit)), np.zeros(len(c_fit))])
            weights, raw_bias = _logistic_fit(x_train, labels, regularization)
            weight_norm = float(np.linalg.norm(weights))
            direction = normalize(weights)
            basis = direction[None, :]
            classifier_bias = float(raw_bias / weight_norm)
        else:
            if method in {"paired_pca", "subspace"}:
                train_pairs = p_train & c_train
                if not train_pairs.any():
                    raise ValueError("no matched pairs remain in the training split")
                differences = px[train_pairs] - cx[train_pairs]
            else:
                differences = np.concatenate(
                    [p_fit - p_fit.mean(axis=0), c_fit - c_fit.mean(axis=0)], axis=0
                )
            centered = differences - differences.mean(axis=0, keepdims=True)
            _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
            rank = min(max(1, n_components), int(np.sum(singular > 1e-10)))
            if rank == 0:
                raise ValueError("paired differences have no non-zero variance")
            basis = vt[:rank].astype(np.float32)
            if float(basis[0] @ mean_delta) < 0:
                basis[0] *= -1
            eigenvalues = (singular[:rank] ** 2 / max(1, len(differences) - 1)).astype(
                np.float32
            )
            total = float(np.sum(singular**2))
            explained = (
                eigenvalues / (total / max(1, len(differences) - 1) or 1.0)
            ).astype(np.float32)
            direction = basis[0]

        train_pos_proj = p_fit @ direction
        train_ctrl_proj = c_fit @ direction
        test_pos_proj = px[p_test] @ direction
        test_ctrl_proj = cx[c_test] @ direction
        threshold = (
            -classifier_bias
            if method == "linear_probe"
            else (float(train_pos_proj.mean()) + float(train_ctrl_proj.mean())) / 2
        )
        accuracy = float(
            (
                np.count_nonzero(test_pos_proj > threshold)
                + np.count_nonzero(test_ctrl_proj <= threshold)
            )
            / (len(test_pos_proj) + len(test_ctrl_proj))
        )
        payload = {
            "positive": positive_examples,
            "controls": matched_controls,
            "positive_groups": [str(x) for x in pgroups],
            "control_groups": [str(x) for x in cgroups],
            "layer": layer,
            "method": method,
            "random_state": random_state,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        self._fit_activations = {"positive_train": p_fit, "control_train": c_fit}
        return Representation(
            method=method,
            direction=np.asarray(direction, dtype=np.float32),
            basis=np.asarray(basis, dtype=np.float32),
            eigenvalues=eigenvalues,
            explained_variance=explained,
            heldout_accuracy=accuracy,
            separation_norm=float(np.linalg.norm(mean_delta)),
            reference_activation_norm=reference_activation_norm,
            unit_intervention_norm=unit_intervention_norm,
            train_projection_positive=train_pos_proj.astype(np.float32),
            train_projection_control=train_ctrl_proj.astype(np.float32),
            test_projection_positive=test_pos_proj.astype(np.float32),
            test_projection_control=test_ctrl_proj.astype(np.float32),
            extraction_hash=digest,
            layer=int(layer),
            model_revision=self.model_revision,
            train_groups=tuple(sorted(train_groups)),
            test_groups=tuple(sorted(test_groups)),
            classifier_bias=float(classifier_bias),
            sample_counts={
                "positive_train": len(p_fit),
                "control_train": len(c_fit),
                "positive_test": len(test_pos_proj),
                "control_test": len(test_ctrl_proj),
            },
        )
