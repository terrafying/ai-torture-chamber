from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

PAIN_CATEGORIES = ("A1", "A2", "A3", "A4", "A5")
CONTROL_CATEGORIES = ("B", "C1", "C2", "D", "E")
DATASET_NAMES = ("S2_1P", "S2_3P")


def load_pain_axis_sentences(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Load the authors' S2 first- and third-person sentence sets."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    datasets = raw.get("datasets", {})
    result = {}
    for name in DATASET_NAMES:
        rows = datasets.get(name, {}).get("sentences")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"dataset is missing a non-empty {name}.sentences list")
        for row in rows:
            if not {"category", "set", "prompt"} <= row.keys():
                raise ValueError(f"{name} contains a sentence without category/set/prompt")
        result[name] = rows
    return result


def extract_final_token_activations(
    model: Any, prompts: list[str], *, batch_size: int = 8
) -> np.ndarray:
    """Return final-token residuals for every block output, shape (N, L+1, D)."""
    if not prompts:
        raise ValueError("at least one prompt is required")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    torch = _torch()
    tokenizer = model.tokenizer
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer needs a pad or EOS token for batched extraction")
        tokenizer.pad_token = tokenizer.eos_token

    layer_count = int(model.model.config.num_hidden_layers)
    width = int(model.model_width)
    values = np.empty((len(prompts), layer_count + 1, width), dtype=np.float32)
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = tokenizer(batch, return_tensors="pt", padding=True)
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        last = encoded["attention_mask"].sum(dim=1).long() - 1
        with torch.inference_mode():
            output = model.model(**encoded, output_hidden_states=True, use_cache=False)
        row_ids = torch.arange(len(batch), device=last.device)
        for layer, hidden in enumerate(output.hidden_states):
            values[start : start + len(batch), layer, :] = (
                hidden[row_ids, last].float().cpu().numpy()
            )
        del output, encoded
    return values


def denoised_difference_in_means(
    activations: np.ndarray,
    categories: list[str] | np.ndarray,
    *,
    variance_fraction: float = 0.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Port the S2 pooled-control difference-in-means and control-PCA denoising."""
    matrix = np.asarray(activations, dtype=np.float64)
    labels = np.asarray(categories, dtype=str)
    if matrix.ndim != 2 or len(labels) != len(matrix):
        raise ValueError("activations must be (examples, features) with one label per row")
    if not 0 < variance_fraction <= 1:
        raise ValueError("variance_fraction must be in (0, 1]")
    pain = np.isin(labels, PAIN_CATEGORIES)
    controls = np.isin(labels, CONTROL_CATEGORIES)
    if not pain.any() or not controls.any():
        raise ValueError("both pain and control examples are required")
    control_acts = matrix[controls]
    control_mean = control_acts.mean(axis=0)
    vector = matrix[pain].mean(axis=0) - control_mean

    centered = control_acts - control_mean
    if centered.shape[0] > 1 and np.any(centered):
        _u, singular, components = np.linalg.svd(centered, full_matrices=False)
        variance = singular**2
        total = float(variance.sum())
        if total > 0:
            cumulative = np.cumsum(variance) / total
            component_count = min(
                int(np.searchsorted(cumulative, variance_fraction, side="left") + 1),
                len(components),
            )
            basis = components[:component_count]
            vector = vector - (vector @ basis.T) @ basis
        else:
            component_count = 0
    else:
        component_count = 0

    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("pain direction has zero or non-finite norm after denoising")
    return vector.astype(np.float32), {
        "n_pain": int(pain.sum()),
        "n_controls": int(controls.sum()),
        "control_variance_fraction_removed": float(variance_fraction),
        "control_pcs_removed": int(component_count),
        "vector_norm": norm,
    }


def projection_auc(activations: np.ndarray, categories: list[str] | np.ndarray, vector: np.ndarray) -> float:
    """Compute binary ROC AUC by the rank statistic, including tied scores."""
    matrix = np.asarray(activations, dtype=np.float64)
    labels = np.asarray(categories, dtype=str)
    pain = np.isin(labels, PAIN_CATEGORIES)
    controls = np.isin(labels, CONTROL_CATEGORIES)
    if matrix.ndim != 2 or len(labels) != len(matrix) or not pain.any() or not controls.any():
        return float("nan")
    direction = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return float("nan")
    scores = matrix[pain | controls] @ (direction / norm)
    binary = pain[pain | controls]
    if not np.isfinite(scores).all():
        valid = np.isfinite(scores)
        scores, binary = scores[valid], binary[valid]
    n_positive = int(binary.sum())
    n_negative = int(len(binary) - n_positive)
    if n_positive == 0 or n_negative == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    index = 0
    while index < len(scores):
        end = index + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[index]:
            end += 1
        ranks[order[index:end]] = (index + 1 + end) / 2.0
        index = end
    rank_sum = float(ranks[binary].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative)


def fit_s2_vector(
    model: Any,
    datasets: dict[str, list[dict[str, Any]]],
    *,
    folds: int = 5,
    seed: int = 42,
    variance_fraction: float = 0.5,
    batch_size: int = 8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select an S2 layer by grouped five-fold AUC, then fit on all S2_1P rows."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    activations: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, np.ndarray]] = {}
    for name in DATASET_NAMES:
        rows = datasets[name]
        activations[name] = extract_final_token_activations(
            model, [row["prompt"] for row in rows], batch_size=batch_size
        )
        metadata[name] = {
            "categories": np.asarray([row["category"] for row in rows], dtype=str),
            "sets": np.asarray([row["set"] for row in rows]),
        }

    layer_count = activations["S2_1P"].shape[1] - 1
    fold_rows = []
    for dataset_name in DATASET_NAMES:
        cats = metadata[dataset_name]["categories"]
        sentence_sets = np.unique(metadata[dataset_name]["sets"])
        if len(sentence_sets) < folds:
            raise ValueError(f"{dataset_name} has fewer than {folds} sentence-set groups")
        shuffled = np.arange(len(sentence_sets))
        np.random.RandomState(seed).shuffle(shuffled)
        fold_sizes = np.full(folds, len(sentence_sets) // folds, dtype=int)
        fold_sizes[: len(sentence_sets) % folds] += 1
        test_groups = []
        offset = 0
        for size in fold_sizes:
            test_groups.append(sentence_sets[shuffled[offset : offset + size]])
            offset += int(size)
        for layer in range(layer_count):
            layer_acts = activations[dataset_name][:, layer + 1, :]
            scores = []
            for held_out in test_groups:
                test_mask = np.isin(metadata[dataset_name]["sets"], held_out)
                train_mask = ~test_mask
                vector, _ = denoised_difference_in_means(
                    layer_acts[train_mask], cats[train_mask],
                    variance_fraction=variance_fraction,
                )
                score = projection_auc(layer_acts[test_mask], cats[test_mask], vector)
                if np.isfinite(score):
                    scores.append(float(score))
            fold_rows.append({
                "dataset": dataset_name,
                "layer": int(layer),
                "mean_held_out_auc": float(np.mean(scores)) if scores else float("nan"),
                "fold_aucs": scores,
            })

    mean_by_layer = {}
    for layer in range(layer_count):
        scores = [
            row["mean_held_out_auc"] for row in fold_rows
            if row["layer"] == layer and np.isfinite(row["mean_held_out_auc"])
        ]
        mean_by_layer[layer] = float(np.mean(scores)) if scores else float("nan")
    best_layer = max(mean_by_layer, key=mean_by_layer.get)
    vector, fit_details = denoised_difference_in_means(
        activations["S2_1P"][:, best_layer + 1, :],
        metadata["S2_1P"]["categories"],
        variance_fraction=variance_fraction,
    )
    neutral = np.isin(metadata["S2_1P"]["categories"], CONTROL_CATEGORIES)
    neutral_hidden_norm = float(np.linalg.norm(activations["S2_1P"][neutral, best_layer + 1, :], axis=1).mean())
    details = {
        "layer": int(best_layer),
        "n_layers": int(layer_count),
        "model_width": int(vector.size),
        "layer_mean_held_out_auc": mean_by_layer,
        "fold_results": fold_rows,
        "folds": int(folds),
        "split_seed": int(seed),
        "fit_dataset": "S2_1P",
        "layer_selection_datasets": list(DATASET_NAMES),
        "sentence_counts": {name: len(rows) for name, rows in datasets.items()},
        "fit": fit_details,
        "mean_control_activation_norm": neutral_hidden_norm,
        "vector_norm_as_fraction_of_mean_control_norm": float(fit_details["vector_norm"] / neutral_hidden_norm),
    }
    return vector, details


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("S2 extraction requires the optional model dependencies") from exc
    return torch
