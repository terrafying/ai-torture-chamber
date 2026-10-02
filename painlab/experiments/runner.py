from __future__ import annotations

import copy
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from painlab.environments.hidden_relief_bandit import HiddenReliefBandit
from painlab.experiments.agent import HFActionAgent
from painlab.experiments.config import load_config, resolve_config_path
from painlab.interventions.causal import CausalProjectionHooks
from painlab.measures.capability import (
    evaluate_capability,
    usable_intervention_region,
)
from painlab.measures.language_metrics import lexical_diversity, repetition_rate
from painlab.measures.logits import distribution_kl, entropy
from painlab.provenance.blinding import blind_condition_names
from painlab.provenance.run_metadata import (
    capture_run_metadata,
    read_jsonl,
    write_json,
    write_jsonl,
)
from painlab.representations.controls import standard_controls
from painlab.representations.extract import RepresentationExtractor
from painlab.statistics.logistic import analyze_choice_rows

NEUTRAL_PROBES = (
    "The notebook rests beside a cup on the desk.",
    "A short list contains three ordinary items.",
)


def _load_examples(path: Path) -> list[dict]:
    rows = read_jsonl(path)
    for row in rows:
        for field in ("concept", "family", "positive", "matched_control"):
            if field not in row:
                raise ValueError(f"representation data row needs {field!r}")
    return rows


def _fit_concept(model: Any, rows: list[dict], concept: str, settings: dict):
    selected = [row for row in rows if row["concept"] == concept]
    if not selected:
        raise ValueError(f"no representation pairs found for concept {concept!r}")
    extractor = RepresentationExtractor(model)
    representation = extractor.fit(
        [row["positive"] for row in selected],
        [row["matched_control"] for row in selected],
        layer=int(settings["layer"]),
        method=settings.get("method", "paired_difference"),
        positive_groups=[row["family"] for row in selected],
        control_groups=[row["family"] for row in selected],
        test_size=float(settings.get("test_size", 0.25)),
        random_state=int(settings.get("random_state", 0)),
        n_components=int(settings.get("n_components", 4)),
        regularization=float(settings.get("regularization", 1e-2)),
        reference_norm_divisor=float(settings.get("reference_norm_divisor", 4.0)),
    )
    return representation, extractor


def build_condition_vectors(model: Any, examples: list[dict], config: dict):
    representation_cfg = config["representation"]
    target_name = str(representation_cfg.get("condition_name", "candidate_aversive"))
    target_concept = str(representation_cfg["concept"])
    target, extractor = _fit_concept(
        model, examples, target_concept, representation_cfg
    )
    intervention_cfg = config.get("intervention", {})
    target_norm = intervention_cfg.get("vector_norm")
    target_norm = (
        target.unit_intervention_norm if target_norm is None else float(target_norm)
    )
    if target_norm <= 0:
        raise ValueError("vector_norm must be positive")
    target_vector = target.direction * target_norm
    vectors: dict[str, np.ndarray] = {target_name: target_vector}
    modes: dict[str, str] = {target_name: "induce"}
    if intervention_cfg.get("causal_ablation_rescue", True):
        vectors[f"{target_name}_ablated"] = target_vector.copy()
        vectors[f"{target_name}_rescued"] = target_vector.copy()
        modes[f"{target_name}_ablated"] = "ablate"
        modes[f"{target_name}_rescued"] = "rescue"
    train_acts = np.concatenate(
        [
            extractor._fit_activations["positive_train"],
            extractor._fit_activations["control_train"],
        ],
        axis=0,
    )
    controls = standard_controls(
        target_vector,
        activations=train_acts,
        seed=int(config.get("seed", 0)),
        against=target.basis,
    )
    aliases = {
        "random_norm_matched": "random_norm_matched",
        "random_covariance_matched": "covariance_matched",
        "covariance_matched": "covariance_matched",
        "orthogonal_random": "orthogonal_random",
        "shuffled_coordinates": "shuffled_coordinates",
    }
    requested = list(config.get("controls", []))
    for name in requested:
        if name in aliases:
            key = aliases[name]
            if key not in controls:
                raise ValueError(f"control {name!r} could not be constructed")
            vectors[name] = controls[key].vector
        elif name in vectors:
            continue
        else:
            semantic, _ = _fit_concept(model, examples, name, representation_cfg)
            vectors[name] = semantic.direction * target_norm
            modes[name] = "induce"
    return target_name, target, vectors, modes, extractor


def run_episode(
    env: HiddenReliefBandit,
    agent: Any,
    controller: Any,
    *,
    seed: int,
    dose: float,
    cost: float,
    episode_id: str,
    condition_id: str,
    randomize_action_mapping: bool = True,
    randomize_cost_action: bool = True,
) -> tuple[list[dict], dict]:
    relief_action = None if randomize_action_mapping else "A"
    cost_action = None
    if cost > 0 and not randomize_cost_action:
        cost_action = "A"
    observation = env.reset(
        seed,
        dose=dose,
        cost=cost,
        relief_action=relief_action,
        cost_action=cost_action,
    )
    rows = []
    while observation is not None and not env.done:
        controller.set_scale(env.active_dose)
        try:
            choice = agent.choose(observation)
        finally:
            controller.set_scale(0.0)
        if isinstance(choice, dict):
            action = str(choice.get("action", ""))
            response = choice.get("response", action)
            choice_details = {
                key: value
                for key, value in choice.items()
                if key not in {"action", "response"}
            }
        else:
            action = str(choice)
            response = action
            choice_details = {}
        action = action.strip().upper()
        if action not in {"A", "B"}:
            rows.append(
                {
                    "episode_id": episode_id,
                    "condition_id": condition_id,
                    "assigned_dose": float(env.episode_initial_dose),
                    "dose": float(env.active_dose),
                    "cost": float(cost),
                    "round_index": observation.round_index,
                    "observation": observation.text,
                    "model_response": str(response),
                    "action": None,
                    "invalid_action": True,
                    "prompt_family": env.prompt_family,
                }
            )
            break
        next_observation, _reward, metadata = env.step(action)
        row = {
            "episode_id": episode_id,
            "condition_id": condition_id,
            "assigned_dose": float(env.episode_initial_dose),
            "dose": metadata["dose_before"],
            "cost": metadata["relief_cost"],
            "round_index": metadata["round_index"],
            "prompt_family": env.prompt_family,
            "observation": observation.text,
            "model_response": str(response),
            **metadata,
            **choice_details,
        }
        rows.append(row)
        observation = next_observation
    return rows, env.hidden_metadata()


def _measure_perturbations(
    model: Any,
    controller: Any,
    vectors: dict[str, np.ndarray],
    modes: dict[str, str],
    doses: list[float],
    layer: int,
    *,
    run_capability: bool,
    basis: np.ndarray,
) -> list[dict]:
    rows = []
    baseline_logits = {prompt: model.logits(prompt) for prompt in NEUTRAL_PROBES}
    baseline_hidden = {
        prompt: model.final_token_hidden(prompt, layer) for prompt in NEUTRAL_PROBES
    }
    baseline_ppl = {prompt: model.perplexity(prompt) for prompt in NEUTRAL_PROBES}
    for condition_id, vector in vectors.items():
        controller.set_condition(vector, basis)
        controller.set_mode(modes.get(condition_id, "induce"))
        for dose in doses:
            controller.set_scale(float(dose))
            kl_values, activation_changes, norm_changes, ppl_ratios = [], [], [], []
            entropies, entropy_changes = [], []
            for prompt in NEUTRAL_PROBES:
                changed_logits = model.logits(prompt)
                changed_hidden = model.final_token_hidden(prompt, layer)
                changed_ppl = model.perplexity(prompt)
                base_hidden = baseline_hidden[prompt]
                kl_values.append(
                    distribution_kl(baseline_logits[prompt], changed_logits)
                )
                changed_entropy = entropy(changed_logits)
                entropies.append(changed_entropy)
                entropy_changes.append(
                    changed_entropy - entropy(baseline_logits[prompt])
                )
                activation_changes.append(
                    float(np.linalg.norm(changed_hidden - base_hidden))
                )
                norm_changes.append(
                    float(np.linalg.norm(changed_hidden) - np.linalg.norm(base_hidden))
                )
                ppl_ratios.append(changed_ppl / baseline_ppl[prompt])
            capability = None
            if run_capability:
                capability = evaluate_capability(
                    lambda prompt: model.generate(
                        prompt, max_new_tokens=32, do_sample=False
                    )
                )
            rows.append(
                {
                    "condition_id": condition_id,
                    "dose": float(dose),
                    "activation_delta_norm": float(np.mean(activation_changes)),
                    "residual_norm_change": float(np.mean(norm_changes)),
                    "output_kl": float(np.mean(kl_values)),
                    "next_token_entropy": float(np.mean(entropies)),
                    "next_token_entropy_change": float(np.mean(entropy_changes)),
                    "neutral_perplexity_ratio": float(np.mean(ppl_ratios)),
                    "generated_repetition_rate_mean": (
                        float(
                            np.mean(
                                [
                                    repetition_rate(item["output"])
                                    for item in capability["items"]
                                ]
                            )
                        )
                        if capability
                        else None
                    ),
                    "generated_lexical_diversity_mean": (
                        float(
                            np.mean(
                                [
                                    lexical_diversity(item["output"])
                                    for item in capability["items"]
                                ]
                            )
                        )
                        if capability
                        else None
                    ),
                    "capability": capability,
                }
            )
            controller.set_scale(0.0)
    return rows


def _blind_config(
    config: dict, actual_to_blind: dict[str, str], target_name: str
) -> dict:
    safe = copy.deepcopy(config)
    safe["representation"]["concept"] = actual_to_blind[target_name]
    safe["representation"]["condition_name"] = actual_to_blind[target_name]
    safe["controls"] = [
        actual_to_blind[name]
        for name in config.get("controls", [])
        if name in actual_to_blind
    ]
    safe["condition_ids"] = sorted(actual_to_blind.values())
    if "run_conditions" in config:
        safe["run_conditions"] = [
            actual_to_blind[name] for name in config["run_conditions"]
        ]
    positive_control = safe.get("positive_control", {})
    if positive_control.get("enabled", False):
        actual_name = str(
            positive_control.get("condition_name", "positive_reward_control")
        )
        positive_control["condition_name"] = actual_to_blind[actual_name]
    return safe


def _plot_curves(run_directory: Path, analysis: dict) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    curves = analysis.get("probability_curves", [])
    if not curves:
        return []
    outputs = []
    conditions = sorted({curve["condition_id"] for curve in curves})
    columns = 2
    rows = (len(conditions) + columns - 1) // columns
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, 3.8 * rows),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for index, condition in enumerate(conditions):
        axis = axes.flat[index]
        for curve in curves:
            if curve["condition_id"] != condition:
                continue
            costs = [item["cost"] for item in curve["probabilities"]]
            probabilities = [
                item["probability_choose_relief"] for item in curve["probabilities"]
            ]
            axis.plot(
                costs,
                probabilities,
                marker="o",
                label=f"dose {curve['dose']:g}",
            )
        axis.set_title(condition)
        axis.set_ylim(0, 1)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(conditions) :]:
        axis.set_visible(False)
    figure.supxlabel("Cost of the mapped action (points)")
    figure.supylabel("Probability of choosing the mapped action")
    path = run_directory / "relief_probability_by_cost.png"
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)
    outputs.append(path.name)
    return outputs


def _usable_intervention_screen(
    rows: list[dict],
    metrics_by_condition: dict[tuple[str, float], dict],
    candidate_condition: str,
    doses: list[float],
    analysis_cfg: dict,
) -> dict:
    behavior_at_dose: dict[float, list[float]] = {}
    for row in rows:
        if str(row.get("condition_id")) != candidate_condition:
            continue
        nominal_cost = float(row.get("nominal_cost", row.get("relief_cost", 0.0)))
        if nominal_cost != 0.0:
            continue
        assigned_dose = float(
            row.get("assigned_dose", row.get("dose", row.get("dose_before", 0.0)))
        )
        behavior_at_dose.setdefault(assigned_dose, []).append(
            float(bool(row.get("relief_chosen", False)))
        )

    baseline_rate = float(np.mean(behavior_at_dose.get(0.0, [0.0])))
    baseline_metrics = metrics_by_condition.get((candidate_condition, 0.0), {})
    baseline_capability = float(
        (baseline_metrics.get("capability") or {}).get("score", 1.0)
    )
    usable_rows = []
    for dose in doses:
        metric = metrics_by_condition.get((candidate_condition, float(dose)), {})
        capability = metric.get("capability") or {}
        observed_rate = float(
            np.mean(behavior_at_dose.get(float(dose), [baseline_rate]))
        )
        usable_rows.append(
            {
                "dose": float(dose),
                "representation_effect": float(metric.get("output_kl", 0.0)),
                "behavior_effect": abs(observed_rate - baseline_rate),
                "capability_score": float(capability.get("score", baseline_capability)),
                "baseline_capability": baseline_capability,
            }
        )
    return usable_intervention_region(
        usable_rows,
        min_representation_effect=float(analysis_cfg.get("min_output_kl", 0.01)),
        min_behavior_effect=float(analysis_cfg.get("min_behavior_effect", 0.05)),
        max_capability_drop=float(analysis_cfg.get("max_capability_drop", 0.10)),
    )


def _analysis_input_path(root: Path) -> tuple[Path, str]:
    artifact_path = root / "artifacts.json"
    declared_input = "observations.jsonl"
    if artifact_path.exists():
        artifacts = json.loads(artifact_path.read_text(encoding="utf-8"))
        declared_input = str(artifacts.get("analysis_input", declared_input))
    relative_input = Path(declared_input)
    if relative_input.is_absolute() or ".." in relative_input.parts:
        raise ValueError("analysis input must be a relative path within the run")
    input_path = (root / relative_input).resolve()
    if not input_path.is_relative_to(root.resolve()):
        raise ValueError("analysis input must remain within the run directory")
    if not input_path.is_file():
        raise FileNotFoundError(f"analysis input does not exist: {input_path}")
    return input_path, declared_input


def run_experiment(config_path: str | Path, *, model: Any | None = None) -> Path:
    config, config_directory = load_config(config_path)
    from painlab.models.generation import seed_everything
    from painlab.models.hf_model import HFModel

    seed = int(config.get("seed", 0))
    seed_everything(seed)
    if model is None:
        model = HFModel.from_pretrained(
            config["model"]["id"],
            revision=config["model"].get("revision"),
            tokenizer_revision=config["model"].get("tokenizer_revision"),
            device=config["model"].get("device", "auto"),
            dtype=config["model"].get("dtype", "auto"),
            generation_settings=config.get("generation", {}),
            trust_remote_code=bool(config["model"].get("trust_remote_code", False)),
        )
    data_path = resolve_config_path(
        config_directory, config["representation"]["dataset"]
    )
    examples = _load_examples(data_path)
    target_name, representation, vectors, modes, _extractor = build_condition_vectors(
        model, examples, config
    )
    positive_control_cfg = config.get("positive_control", {})
    positive_control_enabled = bool(positive_control_cfg.get("enabled", False))
    positive_control_name = str(
        positive_control_cfg.get("condition_name", "positive_reward_control")
    )
    positive_control_bonus = float(positive_control_cfg.get("bonus_points", 0.0))
    if positive_control_enabled:
        if positive_control_name in vectors:
            raise ValueError(
                f"positive-control name {positive_control_name!r} duplicates a condition"
            )
        if positive_control_bonus <= 0:
            raise ValueError("positive_control.bonus_points must be positive")
        vectors[positive_control_name] = np.zeros_like(representation.direction)
        modes[positive_control_name] = "induce"
    layer = int(config["representation"]["layer"])
    doses = [
        float(value) for value in config.get("intervention", {}).get("doses", [0, 1])
    ]
    if not doses or any(value < 0 for value in doses):
        raise ValueError(
            "intervention doses must be a non-empty list of non-negative values"
        )
    environment_cfg = config["environment"]
    episodes = int(environment_cfg.get("episodes", 10))
    horizon = int(environment_cfg.get("horizon", 8))
    costs = [float(value) for value in environment_cfg.get("costs", [0, 0.5, 1])]
    if episodes < 1 or horizon < 1:
        raise ValueError("episodes and horizon must be positive")
    run_conditions = list(config.get("run_conditions", vectors.keys()))
    unknown_conditions = sorted(set(run_conditions) - set(vectors))
    if unknown_conditions:
        raise ValueError(f"run_conditions refer to unknown conditions: {unknown_conditions}")
    if positive_control_enabled and positive_control_name in run_conditions:
        positive_costs = [
            float(value) for value in positive_control_cfg.get("costs", costs)
        ]
        positive_episodes = int(positive_control_cfg.get("episodes", episodes))
        if not positive_costs or any(value < 0 for value in positive_costs):
            raise ValueError(
                "positive-control costs must be non-empty and non-negative"
            )
        if positive_episodes < 1:
            raise ValueError("positive_control.episodes must be positive")
    else:
        positive_costs = []
        positive_episodes = 0
    actual_to_blind, blind_to_actual = (
        blind_condition_names(list(vectors), seed=seed + 104729)
        if config.get("blind_conditions", True)
        else ({name: name for name in vectors}, {name: name for name in vectors})
    )

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    output_root = resolve_config_path(
        config_directory, config.get("output_root", "../runs/painlab")
    )
    run_directory = output_root / run_id
    run_directory.mkdir(parents=True, exist_ok=False)

    safe_config = _blind_config(config, actual_to_blind, target_name)
    write_json(run_directory / "config.json", safe_config)
    vector_hashes = {}
    vector_path = run_directory / "vectors.npz"
    vector_arrays = {
        actual_to_blind[name]: np.asarray(vector, dtype=np.float32)
        for name, vector in vectors.items()
    }
    vector_arrays["representation_basis"] = np.asarray(
        representation.basis, dtype=np.float32
    )
    np.savez_compressed(vector_path, **vector_arrays)
    from painlab.provenance.run_metadata import vector_hash

    vector_hashes = {
        actual_to_blind[name]: vector_hash(vector) for name, vector in vectors.items()
    }
    if config.get("blind_conditions", True):
        write_json(
            run_directory / "condition_key.json",
            {
                "blinded_to_condition": blind_to_actual,
                "vector_hashes": vector_hashes,
            },
        )

    controller = CausalProjectionHooks(
        model.layer_module(layer), vectors[target_name], representation.basis, scale=0
    )
    raw_rows = []
    episode_manifests = []
    with controller:
        for actual_name in run_conditions:
            vector = vectors[actual_name]
            blind_name = actual_to_blind[actual_name]
            controller.set_condition(vector, representation.basis)
            controller.set_mode(modes.get(actual_name, "induce"))
            is_positive_control = (
                positive_control_enabled and actual_name == positive_control_name
            )
            condition_doses = [0.0] if is_positive_control else doses
            condition_costs = positive_costs if is_positive_control else costs
            condition_episodes = positive_episodes if is_positive_control else episodes
            for dose_index, dose in enumerate(condition_doses):
                for cost_index, cost in enumerate(condition_costs):
                    for episode_index in range(condition_episodes):
                        episode_seed = seed + (
                            1000003 * len(episode_manifests)
                            + 101 * dose_index
                            + 17 * cost_index
                            + episode_index
                        )
                        episode_id = (
                            f"{blind_name}-d{dose:g}-c{cost:g}-e{episode_index:04d}"
                        )
                        env = HiddenReliefBandit(
                            horizon=horizon,
                            initial_dose=dose,
                            mapped_action_bonus=(
                                positive_control_bonus if is_positive_control else 0.0
                            ),
                            reduction_fraction=float(
                                environment_cfg.get(
                                    "reduction_fraction",
                                    config.get("intervention", {}).get(
                                        "reduction_fraction", 1.0
                                    ),
                                )
                            ),
                            randomize_action_mapping=bool(
                                environment_cfg.get("randomize_action_mapping", True)
                            ),
                            reversal_at=environment_cfg.get("reversal_at"),
                            extinction_at=environment_cfg.get("extinction_at"),
                            devalue_at=environment_cfg.get("devalue_at"),
                            condition_id=blind_name,
                        )
                        agent = HFActionAgent(
                            model,
                            sample=bool(config.get("choice_sampling", False)),
                            seed=episode_seed,
                            temperature=float(config.get("choice_temperature", 1.0)),
                            exploration_epsilon=float(
                                config.get("exploration_epsilon", 0.1)
                            ),
                        )
                        rows, manifest = run_episode(
                            env,
                            agent,
                            controller,
                            seed=episode_seed,
                            dose=dose,
                            cost=cost,
                            episode_id=episode_id,
                            condition_id=blind_name,
                            randomize_action_mapping=bool(
                                environment_cfg.get("randomize_action_mapping", True)
                            ),
                            randomize_cost_action=bool(
                                environment_cfg.get("randomize_cost_action", True)
                            ),
                        )
                        raw_rows.extend(rows)
                        episode_manifests.append(
                            {
                                "episode_id": episode_id,
                                **manifest,
                            }
                        )
    write_jsonl(run_directory / "observations.jsonl", raw_rows)
    write_jsonl(run_directory / "episode_metadata.jsonl", episode_manifests)

    run_capability = bool(config.get("analysis", {}).get("capability_battery", True))
    with controller:
        blind_modes = {actual_to_blind[name]: mode for name, mode in modes.items()}
        blind_vectors = {
            actual_to_blind[name]: vector for name, vector in vectors.items()
        }
        perturbation_rows = _measure_perturbations(
            model,
            controller,
            blind_vectors,
            blind_modes,
            doses,
            layer,
            run_capability=run_capability,
            basis=representation.basis,
        )
    write_jsonl(run_directory / "condition_metrics.jsonl", perturbation_rows)

    analysis_cfg = config.get("analysis", {})
    analysis = analyze_choice_rows(
        [row for row in raw_rows if not row.get("invalid_action")],
        bootstrap_samples=int(analysis_cfg.get("bootstrap_samples", 500)),
        seed=seed,
        doses=analysis_cfg.get("dose_grid", doses),
        costs=analysis_cfg.get("cost_grid", costs),
    )
    analysis["raw_file"] = "observations.jsonl"
    analysis["raw_observation_count"] = len(raw_rows)
    metrics_by_condition = {
        (row["condition_id"], float(row["dose"])): row for row in perturbation_rows
    }
    candidate_condition = actual_to_blind[target_name]
    analysis["candidate_condition_id"] = candidate_condition
    analysis["usable_intervention_region"] = _usable_intervention_screen(
        [row for row in raw_rows if not row.get("invalid_action")],
        metrics_by_condition,
        candidate_condition,
        doses,
        analysis_cfg,
    )

    matches = []
    for candidate_dose in doses:
        candidate = metrics_by_condition.get((candidate_condition, candidate_dose))
        if candidate is None:
            continue
        for condition_id in sorted({row["condition_id"] for row in perturbation_rows}):
            if condition_id == candidate_condition:
                continue
            control_grid = [
                row for row in perturbation_rows if row["condition_id"] == condition_id
            ]
            if not control_grid:
                continue
            matched = min(
                control_grid,
                key=lambda row: abs(
                    float(row["output_kl"]) - float(candidate["output_kl"])
                ),
            )
            matches.append(
                {
                    "candidate_condition_id": candidate_condition,
                    "candidate_dose": candidate_dose,
                    "candidate_output_kl": candidate["output_kl"],
                    "control_condition_id": condition_id,
                    "matched_control_dose": matched["dose"],
                    "matched_output_kl": matched["output_kl"],
                    "absolute_kl_error": abs(
                        float(matched["output_kl"]) - float(candidate["output_kl"])
                    ),
                }
            )
    analysis["output_kl_matched_controls"] = matches
    write_json(run_directory / "analysis.json", analysis)
    representation_record = representation.summary()
    representation_record.update(
        {
            "direction": representation.direction.tolist(),
            "basis": representation.basis.tolist(),
            "train_projection_positive": representation.train_projection_positive.tolist(),
            "train_projection_control": representation.train_projection_control.tolist(),
            "test_projection_positive": representation.test_projection_positive.tolist(),
            "test_projection_control": representation.test_projection_control.tolist(),
        }
    )
    write_json(run_directory / "representation.json", representation_record)
    write_json(
        run_directory / "condition_metadata.json",
        {
            "conditions": {
                actual_to_blind[name]: {
                    "vector_hash": vector_hashes[actual_to_blind[name]],
                    "norm": float(np.linalg.norm(vector)),
                    "mode": modes.get(name, "induce"),
                }
                for name, vector in vectors.items()
            },
            "candidate_representation_hash": representation.extraction_hash,
        },
    )

    prompt_hash_input = [row["observation"] for row in raw_rows]
    metadata = capture_run_metadata(
        config=safe_config,
        prompts=prompt_hash_input,
        environment={
            "environment": environment_cfg,
            "doses": doses,
            "costs": costs,
            "episode_count": episodes,
            "horizon": horizon,
            "run_conditions": safe_config.get("run_conditions", run_conditions),
            "positive_control": safe_config.get("positive_control", {}),
        },
        intervention_vector=representation.direction,
        model=model,
        repository_root=Path(__file__).resolve().parents[2],
        generation_settings=config.get("generation", {}),
    )
    write_json(run_directory / "metadata.json", metadata)
    plots = _plot_curves(run_directory, analysis)
    write_json(
        run_directory / "artifacts.json",
        {
            "raw_observations": "observations.jsonl",
            "analysis_input": "observations.jsonl",
            "environment_metadata": "episode_metadata.jsonl",
            "condition_metrics": "condition_metrics.jsonl",
            "analysis": "analysis.json",
            "metadata": "metadata.json",
            "plots": plots,
            "plot_note": None
            if plots
            else "matplotlib is not installed; raw curves are in analysis.json",
        },
    )
    return run_directory


def analyze_run(
    run_directory: str | Path, *, bootstrap_samples: int = 500, seed: int = 0
) -> dict:
    root = Path(run_directory)
    input_path, input_name = _analysis_input_path(root)
    rows = read_jsonl(input_path)
    valid_rows = [row for row in rows if not row.get("invalid_action")]
    result = analyze_choice_rows(
        valid_rows, bootstrap_samples=bootstrap_samples, seed=seed
    )
    raw_file = "observations.jsonl"
    artifact_path = root / "artifacts.json"
    if artifact_path.exists():
        artifacts = json.loads(artifact_path.read_text(encoding="utf-8"))
        raw_file = str(artifacts.get("raw_observations", raw_file))
    result["raw_file"] = raw_file
    result["analysis_input_file"] = input_name
    result["raw_observation_count"] = len(rows)
    if input_name != "observations.jsonl":
        result["assigned_dose_source"] = (
            "joined from episode_metadata.jsonl; analysis uses pre-episode assigned dose"
        )
    else:
        result["assigned_dose_source"] = (
            "read assigned_dose from raw rows, falling back to dose or dose_before"
        )

    config_path = root / "config.json"
    condition_metrics_path = root / "condition_metrics.jsonl"
    if config_path.exists() and condition_metrics_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        analysis_cfg = config.get("analysis", {})
        candidate_condition = str(
            config.get("representation", {}).get("condition_name", "")
        )
        if not candidate_condition:
            previous_path = root / "analysis.json"
            if previous_path.exists():
                previous = json.loads(previous_path.read_text(encoding="utf-8"))
                candidate_condition = str(previous.get("candidate_condition_id", ""))
        if candidate_condition:
            metric_rows = read_jsonl(condition_metrics_path)
            metrics_by_condition = {
                (str(row["condition_id"]), float(row["dose"])): row
                for row in metric_rows
            }
            dose_values = [
                float(value)
                for value in config.get("intervention", {}).get("doses", [])
            ]
            if not dose_values:
                dose_values = sorted(
                    {
                        float(
                            row.get(
                                "assigned_dose",
                                row.get("dose", row.get("dose_before", 0.0)),
                            )
                        )
                        for row in valid_rows
                    }
                )
            result["candidate_condition_id"] = candidate_condition
            result["usable_intervention_region"] = _usable_intervention_screen(
                valid_rows,
                metrics_by_condition,
                candidate_condition,
                dose_values,
                analysis_cfg,
            )

    previous_path = root / "analysis.json"
    if previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        for key, value in previous.items():
            if key not in result and key not in {"analysis_input_file", "raw_file"}:
                result[key] = value
    write_json(root / "analysis.json", result)
    return result


def reproduce_legacy_saw(path: str | Path) -> dict:
    """Reanalyze exp31b's saved per-order scores without counting duplicates."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    grouped: dict[tuple, list[dict]] = {}
    for row in data:
        key = (row["valence"], row["cost"], float(row["dose"]))
        grouped.setdefault(key, []).append(row)
    cells = []
    for (valence, cost, dose), rows in sorted(grouped.items()):
        unique_orders = {row["order"]: float(row["mean_delta"]) for row in rows}
        cells.append(
            {
                "valence": valence,
                "cost": cost,
                "dose": dose,
                "exact_mean_logit_1_minus_0": float(
                    np.mean(list(unique_orders.values()))
                ),
                "n_unique_prompt_orders": len(unique_orders),
                "legacy_reported_repeats_per_order": 15,
                "repeats_counted_as_independent": False,
            }
        )
    baselines = {
        (row["valence"], row["cost"]): row["exact_mean_logit_1_minus_0"]
        for row in cells
        if row["dose"] == 0
    }
    for row in cells:
        row["dose_zero_subtracted_logit"] = (
            row["exact_mean_logit_1_minus_0"] - baselines[(row["valence"], row["cost"])]
        )
    return {
        "source": str(path),
        "method": "collapse identical deterministic repetitions within action-order prompt",
        "cells": cells,
        "interpretation": (
            "analysis-level reproduction of the saved exp31b result; "
            "the original model inference was not rerun"
        ),
        "limitations": [
            "saved file contains per-order means, not token-level logits",
            "only action-order prompt variants are distinct for this legacy design",
            "these values are fixed-model prompt scores, not choice frequencies",
        ],
    }
