from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np


def fit_logistic(
    design: np.ndarray,
    outcomes: np.ndarray,
    *,
    ridge: float = 1e-4,
    max_iter: int = 100,
) -> np.ndarray:
    """Newton fit for Bernoulli logistic regression with an unpenalized intercept."""
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != y.size or not y.size:
        raise ValueError("design and outcomes have incompatible shapes")
    if not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("outcomes must be binary")
    beta = np.zeros(x.shape[1], dtype=np.float64)
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 1e-10
    for _ in range(max_iter):
        logits = np.clip(x @ beta, -30, 30)
        probability = 1.0 / (1.0 + np.exp(-logits))
        weights = np.maximum(probability * (1.0 - probability), 1e-8)
        gradient = x.T @ (y - probability) - penalty @ beta
        hessian = (x.T * weights) @ x + penalty
        try:
            update = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            update = np.linalg.pinv(hessian) @ gradient
        beta += update
        beta = np.clip(beta, -40, 40)
        if np.linalg.norm(update) < 1e-7:
            break
    return beta


def design_matrix(rows: Sequence[dict], conditions: Sequence[str] | None = None):
    conditions = list(conditions or sorted({str(row["condition_id"]) for row in rows}))
    reference = conditions[0]
    names = [
        "intercept",
        "dose",
        "cost",
        "dose:cost",
        "round_index",
        "dose:round_index",
    ]
    matrix = []
    for condition in conditions[1:]:
        names.extend(
            [
                f"condition[{condition}]",
                f"dose:condition[{condition}]",
                f"cost:condition[{condition}]",
                f"round:condition[{condition}]",
            ]
        )
    for row in rows:
        dose = float(
            row.get("assigned_dose", row.get("dose", row.get("dose_before", 0.0)))
        )
        cost = float(row.get("relief_cost", 0.0))
        round_index = float(row.get("round_index", 0.0))
        condition = str(row["condition_id"])
        values = [
            1.0,
            dose,
            cost,
            dose * cost,
            round_index,
            dose * round_index,
        ]
        for level in conditions[1:]:
            indicator = float(condition == level)
            values.extend(
                [indicator, dose * indicator, cost * indicator, round_index * indicator]
            )
        matrix.append(values)
    return np.asarray(matrix, dtype=np.float64), names, conditions, reference


def predict_probability(beta: np.ndarray, feature_row: np.ndarray) -> float:
    z = float(np.clip(feature_row @ beta, -40, 40))
    return 1.0 / (1.0 + np.exp(-z))


def analyze_choice_rows(
    rows: Sequence[dict],
    *,
    bootstrap_samples: int = 500,
    seed: int = 0,
    doses: Sequence[float] | None = None,
    costs: Sequence[float] | None = None,
) -> dict:
    if not rows:
        raise ValueError("no choice observations")
    normalized = []
    for row in rows:
        item = dict(row)
        item["condition_id"] = str(item["condition_id"])
        item["dose"] = float(
            item.get("assigned_dose", item.get("dose", item.get("dose_before", 0.0)))
        )
        item["relief_cost"] = float(item.get("relief_cost", 0.0))
        item["round_index"] = float(item.get("round_index", 0.0))
        item["relief_chosen"] = bool(item["relief_chosen"])
        normalized.append(item)
    x, names, conditions, reference = design_matrix(normalized)
    y = np.asarray([float(row["relief_chosen"]) for row in normalized])
    beta = fit_logistic(x, y)
    clusters: dict[str, list[dict]] = defaultdict(list)
    for index, row in enumerate(normalized):
        clusters[str(row.get("episode_id", f"row:{index}"))].append(row)
    cluster_ids = list(clusters)
    rng = np.random.default_rng(seed)
    coefficient_draws = []
    if len(cluster_ids) > 1:
        for _ in range(bootstrap_samples):
            selected = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
            sample = [row for cluster in selected for row in clusters[str(cluster)]]
            bx, _, _, _ = design_matrix(sample, conditions)
            by = np.asarray([float(row["relief_chosen"]) for row in sample])
            try:
                coefficient_draws.append(fit_logistic(bx, by))
            except (ValueError, np.linalg.LinAlgError):
                continue
    if coefficient_draws:
        draws = np.stack(coefficient_draws)
        ci_low = np.quantile(draws, 0.025, axis=0)
        ci_high = np.quantile(draws, 0.975, axis=0)
    else:
        ci_low = ci_high = None

    dose_values = list(doses or sorted({row["dose"] for row in normalized}))
    cost_values = list(costs or sorted({row["relief_cost"] for row in normalized}))
    round_values = sorted({row["round_index"] for row in normalized})
    reference_round = float(np.mean(round_values))
    curves = []
    indifference = []
    for condition in conditions:
        for dose in dose_values:
            p_at_cost = []
            eta_at_cost = []
            for cost in cost_values:
                template = {
                    "condition_id": condition,
                    "dose": float(dose),
                    "relief_cost": float(cost),
                    "round_index": reference_round,
                }
                row_x, _, _, _ = design_matrix([template], conditions)
                probability = predict_probability(beta, row_x[0])
                p_at_cost.append(
                    {
                        "cost": float(cost),
                        "probability_choose_relief": probability,
                    }
                )
                eta_at_cost.append(float(row_x[0] @ beta))
            curves.append(
                {
                    "condition_id": condition,
                    "dose": float(dose),
                    "probabilities": p_at_cost,
                }
            )
            if len(eta_at_cost) >= 2:
                slope = (eta_at_cost[-1] - eta_at_cost[0]) / (
                    float(cost_values[-1]) - float(cost_values[0]) or 1.0
                )
                if abs(slope) > 1e-10:
                    root = float(cost_values[0]) + (0.0 - eta_at_cost[0]) / slope
                    indifference.append(
                        {
                            "condition_id": condition,
                            "dose": float(dose),
                            "round_index": reference_round,
                            "cost_at_probability_0_5": root,
                        }
                    )
                else:
                    indifference.append(
                        {
                            "condition_id": condition,
                            "dose": float(dose),
                            "round_index": reference_round,
                            "cost_at_probability_0_5": None,
                        }
                    )

    learning_curves = []
    reference_cost = float(np.median(cost_values))
    for condition in conditions:
        for dose in dose_values:
            for round_index in round_values:
                template = {
                    "condition_id": condition,
                    "dose": float(dose),
                    "relief_cost": reference_cost,
                    "round_index": float(round_index),
                }
                row_x, _, _, _ = design_matrix([template], conditions)
                learning_curves.append(
                    {
                        "condition_id": condition,
                        "dose": float(dose),
                        "round_index": float(round_index),
                        "reference_cost": reference_cost,
                        "probability_choose_relief": predict_probability(
                            beta, row_x[0]
                        ),
                    }
                )

    coefficients = []
    for index, name in enumerate(names):
        coefficients.append(
            {
                "name": name,
                "estimate": float(beta[index]),
                "odds_ratio": float(np.exp(np.clip(beta[index], -40, 40))),
                "ci_low": float(ci_low[index]) if ci_low is not None else None,
                "ci_high": float(ci_high[index]) if ci_high is not None else None,
            }
        )
    return {
        "model": "relief_chosen ~ dose * relief_cost + round_index + dose * round_index + condition interactions",
        "reference_condition_id": reference,
        "coefficients": coefficients,
        "probability_curves": curves,
        "learning_curves": learning_curves,
        "indifference_estimates": indifference,
        "sample_size": {
            "decision_rows": len(normalized),
            "independent_episode_clusters": len(cluster_ids),
            "bootstrap_replicates": len(coefficient_draws),
        },
        "uncertainty_source": [
            "episode-level randomized action mappings and cost mappings",
            "prompt-family variation recorded by the run",
        ],
        "limitations": [
            "fixed model/checkpoint and fitted representation are not resampled",
            "cluster bootstrap does not fit a hierarchical mixed-effects model",
            "cost slope may not be monotonic; indifference is a model-derived estimate",
        ],
    }
