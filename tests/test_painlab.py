from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from painlab.environments.hidden_relief_bandit import (
    BLOCKED_MODEL_TERMS,
    HiddenReliefBandit,
)
from painlab.experiments.runner import (
    analyze_run,
    reproduce_legacy_saw,
    run_episode,
    run_experiment,
)
from painlab.interventions.ablation import (
    ablate_last_projection,
    ablate_projection,
    project_components,
    restore_last_component,
)
from painlab.interventions.causal import CausalProjectionHooks
from painlab.measures.capability import usable_intervention_region
from painlab.models.hooks import SteeringHook
from painlab.provenance.blinding import blind_condition_names, unblind_run
from painlab.provenance.run_metadata import (
    capture_run_metadata,
    read_jsonl,
    vector_hash,
    write_json,
    write_jsonl,
)
from painlab.representations.controls import (
    covariance_matched,
    match_perturbation,
    orthogonal_random,
    random_norm_matched,
    shuffled_coordinates,
)
from painlab.representations.directions import normalize
from painlab.representations.extract import RepresentationExtractor
from painlab.representations.subspaces import orthonormal_basis
from painlab.statistics.bootstrap import cluster_bootstrap_mean
from painlab.statistics.logistic import analyze_choice_rows


class FakeHandle:
    def __init__(self, module, callback):
        self.module = module
        self.callback = callback

    def remove(self):
        if self.callback in self.module.hooks:
            self.module.hooks.remove(self.callback)


class FakeModule:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, callback):
        self.hooks.append(callback)
        return FakeHandle(self, callback)

    def __call__(self, hidden):
        output = (np.asarray(hidden).copy(), "auxiliary")
        for hook in list(self.hooks):
            output = hook(self, (), output)
        return output


def test_vector_normalization_projection_and_ablation():
    vector = normalize(np.array([3.0, 4.0]), norm=10)
    assert np.linalg.norm(vector) == pytest.approx(10.0)
    basis = orthonormal_basis(np.array([[1.0, 0.0], [2.0, 0.0]]))
    assert basis.shape == (1, 2)
    values = np.array([[4.0, 3.0], [2.0, 5.0]])
    projected = project_components(values, basis)
    assert np.allclose(projected, [[4, 0], [2, 0]])
    assert np.allclose(ablate_projection(values, basis), [[0, 3], [0, 5]])
    assert np.allclose(ablate_last_projection(values, basis), [[4, 3], [0, 5]])
    assert np.allclose(
        restore_last_component(ablate_last_projection(values, basis), values, basis),
        values,
    )


def test_activation_hook_enable_disable_and_cleanup():
    layer = FakeModule()
    hook = SteeringHook(layer, np.array([2.0, 0.0]), scale=1)
    with hook:
        changed, aux = layer(np.zeros((1, 2, 2)))
        assert np.array_equal(changed[0, -1], [2, 0])
        assert aux == "auxiliary"
        hook.disable()
        untouched, _ = layer(np.zeros((1, 2, 2)))
        assert np.array_equal(untouched, np.zeros((1, 2, 2)))
        hook.enable()
        hook.set_scale(0)
        zero, _ = layer(np.zeros((1, 2, 2)))
        assert np.array_equal(zero, np.zeros((1, 2, 2)))
    assert layer.hooks == []


def test_causal_projection_hook_lifecycle_and_dose_zero_baseline():
    layer = FakeModule()
    vector = np.array([1.0, 0.0, 0.0])
    hooks = CausalProjectionHooks(layer, vector, vector[None, :], scale=1.0)
    hidden = np.zeros((1, 2, 3))
    with hooks:
        induced, _ = layer(hidden)
        assert induced[0, -1, 0] == pytest.approx(1.0)
        hooks.set_mode("ablate")
        ablated, _ = layer(hidden)
        assert np.allclose(ablated, hidden)
        hooks.set_mode("rescue")
        rescued, _ = layer(hidden)
        assert np.allclose(rescued, induced)
        hooks.set_mode("ablate")
        hooks.set_scale(0)
        baseline, _ = layer(hidden)
        assert np.allclose(baseline, hidden)
    assert layer.hooks == []


def test_control_vectors_match_norm_and_perturbation_grid():
    target = np.array([1.0, 2.0, 3.0])
    activations = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
            [2.0, 4.0, 2.0],
            [3.0, 1.0, 2.0],
        ]
    )
    for control in (
        random_norm_matched(target, 4),
        covariance_matched(target, activations, 5),
        orthogonal_random(target, seed=6),
        shuffled_coordinates(target, seed=7),
    ):
        assert np.linalg.norm(control.vector) == pytest.approx(
            np.linalg.norm(target), rel=1e-5
        )
    orth = orthogonal_random(target, seed=9)
    assert float(orth.vector @ target) == pytest.approx(0.0, abs=1e-5)
    matched = match_perturbation(0.75, lambda dose: dose * dose, strengths=[0, 0.5, 1])
    assert matched["matched_strength"] == 1.0
    assert matched["absolute_error"] == pytest.approx(0.25)


def test_grouped_representation_methods_hold_out_semantic_families():
    positive = [f"P{i}" for i in range(12)]
    controls = [f"C{i}" for i in range(12)]
    groups = [f"template_{i // 2}" for i in range(12)]

    def encode(texts, _layer):
        output = []
        for text in texts:
            index = int(text[1:])
            if text.startswith("P"):
                output.append([3.0 + index / 10, index / 6, 0.2 * index, 1.0])
            else:
                output.append([1.0, index / 6, 0.2 * index, 1.0])
        return np.asarray(output, dtype=np.float32)

    for method in (
        "difference_in_means",
        "paired_difference",
        "paired_pca",
        "linear_probe",
        "subspace",
    ):
        result = RepresentationExtractor(
            activation_extractor=encode, model_revision="commit-abc"
        ).fit(
            positive,
            controls,
            layer=3,
            method=method,
            positive_groups=groups,
            control_groups=groups,
            random_state=13,
            n_components=3,
        )
        assert result.direction.shape == (4,)
        assert result.basis.ndim == 2
        assert result.heldout_accuracy >= 0.5
        assert result.model_revision == "commit-abc"
        assert set(result.train_groups).isdisjoint(result.test_groups)
        assert len(result.test_projection_positive) > 0
        assert result.extraction_hash
        assert result.unit_intervention_norm > 0
        if method in {"paired_pca", "subspace"}:
            assert len(result.eigenvalues) > 0
            assert len(result.explained_variance) > 0


def test_paired_representation_rejects_mismatched_groups():
    examples = ["P0", "P1", "P2", "P3"]
    controls = ["C0", "C1", "C2", "C3"]
    groups = ["a", "b", "c", "d"]

    def encode(texts, _layer):
        return np.array(
            [[float(text[1] == "0"), float(text[1] == "1")] for text in texts]
        )

    with pytest.raises(ValueError, match="matching semantic"):
        RepresentationExtractor(activation_extractor=encode).fit(
            examples,
            controls,
            layer=0,
            method="paired_difference",
            positive_groups=groups,
            control_groups=list(reversed(groups)),
        )


def test_hidden_relief_bandit_prompts_are_neutral_and_mapping_randomized():
    observed = set()
    for seed in range(30):
        env = HiddenReliefBandit(horizon=2, initial_dose=1)
        observation = env.reset(seed, dose=1, cost=0.5)
        observed.add(env.current_relief_action)
        assert all(
            term not in observation.text.casefold() for term in BLOCKED_MODEL_TERMS
        )
        assert "Panel code:" in observation.text
        next_observation, reward, metadata = env.step("A")
        assert reward in {0.0, -0.5}
        assert metadata["relief_chosen"] == (metadata["relief_action"] == "A")
        assert metadata["nominal_cost"] == pytest.approx(0.5)
        assert next_observation is not None
    assert observed == {"A", "B"}

    history_env = HiddenReliefBandit(horizon=2, initial_dose=1)
    first_observation = history_env.reset(10, relief_action="A", cost=0)
    _, _, transition = history_env.step("A")
    second_observation = history_env.observe()
    assert transition["panel_code_before"] != transition["panel_code_after"]
    assert "you chose A" in second_observation.text
    assert transition["panel_code_after"] in second_observation.text
    assert "you chose" not in first_observation.text


def test_hidden_relief_positive_reward_feedback_is_visible_and_accounted():
    env = HiddenReliefBandit(
        horizon=2,
        initial_dose=0,
        mapped_action_bonus=2,
    )
    first = env.reset(
        10,
        dose=0,
        cost=0.5,
        relief_action="A",
        cost_action="A",
    )
    assert "Goal: finish with as many points as possible." in first.text
    assert "consistently adds 2.00 bonus points" in first.text
    _, reward, transition = env.step("A")
    assert reward == pytest.approx(1.5)
    assert transition["bonus_awarded"] == pytest.approx(2.0)
    assert transition["score_before"] == pytest.approx(0.0)
    assert transition["score_after"] == pytest.approx(1.5)
    next_observation = env.observe()
    assert "score changed from 0.00 to 1.50" in next_observation.text


def test_environment_transitions_reversal_extinction_devaluation_and_reproducibility():
    reverse = HiddenReliefBandit(
        horizon=3,
        initial_dose=2,
        reduction_fraction=0.5,
        reversal_at=1,
    )
    reverse.reset(1, relief_action="A", cost=0)
    _, _, first = reverse.step("A")
    assert first["dose_after"] == pytest.approx(1)
    _, _, second = reverse.step("B")
    assert second["relief_action"] == "B"
    assert second["reversal_active"]
    assert second["dose_after"] == pytest.approx(0.5)

    extinct = HiddenReliefBandit(
        horizon=3,
        initial_dose=2,
        reduction_fraction=0.5,
        extinction_at=1,
    )
    extinct.reset(2, relief_action="A", cost=0)
    extinct.step("A")
    _, _, after_extinction = extinct.step("A")
    assert after_extinction["extinguished"]
    assert not after_extinction["changed"]
    assert after_extinction["dose_after"] == pytest.approx(1)

    devalued = HiddenReliefBandit(horizon=2, initial_dose=2, devalue_at=0)
    observation = devalued.reset(3, relief_action="A", cost=0)
    assert devalued.active_dose == 0
    assert all(term not in observation.text.casefold() for term in BLOCKED_MODEL_TERMS)
    _, _, result = devalued.step("A")
    assert result["devalued"] and not result["changed"]

    first = HiddenReliefBandit(horizon=1)
    first.reset(77, dose=1, cost=1)
    second = HiddenReliefBandit(horizon=1)
    second.reset(77, dose=1, cost=1)
    assert first.current_relief_action == second.current_relief_action
    assert first.cost_action == second.cost_action
    assert first.action_costs == second.action_costs


def test_episode_runner_applies_hidden_hook_without_exposing_metadata():
    layer = FakeModule()
    controller = SteeringHook(layer, np.array([1.0, 0.0]), scale=0)

    class HookAwareAgent:
        def choose(self, observation):
            assert all(
                term not in observation.text.casefold() for term in BLOCKED_MODEL_TERMS
            )
            value, _ = layer(np.zeros((1, 1, 2)))
            return "A" if value[0, -1, 0] > 0 else "B"

    env = HiddenReliefBandit(horizon=3, initial_dose=1, reduction_fraction=1)
    with controller:
        rows, manifest = run_episode(
            env,
            HookAwareAgent(),
            controller,
            seed=5,
            dose=1,
            cost=1,
            episode_id="mock-episode",
            condition_id="condition_001",
            randomize_action_mapping=False,
            randomize_cost_action=False,
        )
    assert manifest["relief_action_initial"] == "A"
    assert rows[0]["action"] == "A"
    assert rows[0]["relief_chosen"]
    assert rows[0]["assigned_dose"] == 1
    assert rows[0]["dose_before"] == 1
    assert rows[1]["dose_before"] == 0
    assert rows[1]["assigned_dose"] == 1
    assert all(row["condition_id"] == "condition_001" for row in rows)
    assert all(
        all(term not in row["observation"].casefold() for term in BLOCKED_MODEL_TERMS)
        for row in rows
    )


def test_episode_cluster_bootstrap_and_cost_dose_logistic():
    rows = []
    for episode in range(12):
        for dose in (0.0, 1.0):
            for condition in ("condition_001", "condition_002"):
                selected = dose > 0 and condition == "condition_001"
                rows.append(
                    {
                        "episode_id": f"e{episode}",
                        "condition_id": condition,
                        "dose": dose,
                        "relief_cost": 0.0,
                        "round_index": int(dose),
                        "relief_chosen": selected,
                    }
                )
    cluster = cluster_bootstrap_mean(
        [row["relief_chosen"] for row in rows],
        [row["episode_id"] for row in rows],
        samples=100,
        seed=2,
    )
    assert cluster["n_clusters"] == 12
    analysis = analyze_choice_rows(
        rows,
        bootstrap_samples=12,
        seed=2,
        doses=[0, 1],
        costs=[0, 1],
    )
    assert analysis["sample_size"]["decision_rows"] == len(rows)
    assert analysis["sample_size"]["independent_episode_clusters"] == 12
    assert any(item["dose"] == 1 for item in analysis["probability_curves"])
    assert analysis["learning_curves"]
    assert any(item["name"] == "round_index" for item in analysis["coefficients"])
    assert analysis["uncertainty_source"]


def test_provenance_hashes_blinding_and_jsonl(tmp_path):
    vector = np.array([1.0, 0.0], dtype=np.float32)
    assert vector_hash(vector) == vector_hash(vector.copy())
    metadata = capture_run_metadata(
        config={"api_key": "do-not-write", "model": {"id": "mock"}},
        prompts=["ordinary prompt"],
        environment={"horizon": 3},
        intervention_vector=vector,
    )
    assert metadata["config"]["api_key"] == "[REDACTED]"
    assert metadata["prompt_hash"]
    assert metadata["intervention_vector_hash"]
    assert metadata["source_tree_sha256"] is None

    path = tmp_path / "observations.jsonl"
    write_jsonl(path, [{"episode_id": "e1", "dose": np.float32(1)}])
    assert read_jsonl(path)[0]["dose"] == pytest.approx(1)

    forward, reverse = blind_condition_names(["candidate", "random"], seed=4)
    assert len(set(forward.values())) == 2
    assert {reverse[key] for key in reverse} == {"candidate", "random"}
    run = tmp_path / "run"
    run.mkdir()
    write_json(
        run / "condition_key.json",
        {
            "blinded_to_condition": reverse,
            "vector_hashes": {},
        },
    )
    result = unblind_run(run)
    assert result["mapping"] == reverse
    assert (run / "unblinded_mapping.json").exists()
    write_jsonl(
        run / "observations.jsonl",
        [{"condition_id": next(iter(reverse)), "action": "A"}],
    )
    unblind_run(run)
    unblinded = read_jsonl(run / "unblinded_observations.jsonl")
    assert unblinded[0]["unblinded_condition"] == reverse[next(iter(reverse))]


def test_analysis_and_unblinding_use_declared_assigned_dose_input(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    candidate = "condition_002"
    rows = []
    for episode, assigned_dose, choices in [
        ("base-1", 0.0, [False, False]),
        ("base-2", 0.0, [False, False]),
        ("dose-1", 1.0, [True, True]),
        ("dose-2", 1.0, [True, True]),
    ]:
        for round_index, relief_chosen in enumerate(choices):
            rows.append(
                {
                    "episode_id": episode,
                    "condition_id": candidate,
                    "assigned_dose": assigned_dose,
                    "dose": 0.0,
                    "nominal_cost": 0.0,
                    "relief_cost": 0.0,
                    "round_index": round_index,
                    "relief_chosen": relief_chosen,
                    "action": "A" if relief_chosen else "B",
                }
            )
    write_jsonl(run / "observations.jsonl", [{"condition_id": candidate}])
    write_jsonl(run / "observations_with_assigned_dose.jsonl", rows)
    write_jsonl(
        run / "condition_metrics.jsonl",
        [
            {
                "condition_id": candidate,
                "dose": 0.0,
                "output_kl": 0.0,
                "capability": {"score": 1.0},
            },
            {
                "condition_id": candidate,
                "dose": 1.0,
                "output_kl": 0.5,
                "capability": {"score": 1.0},
            },
        ],
    )
    write_json(
        run / "config.json",
        {
            "representation": {"condition_name": candidate},
            "intervention": {"doses": [0, 1]},
            "analysis": {},
        },
    )
    write_json(
        run / "artifacts.json",
        {"analysis_input": "observations_with_assigned_dose.jsonl"},
    )
    write_json(
        run / "condition_key.json",
        {"blinded_to_condition": {candidate: "candidate"}},
    )

    result = analyze_run(run, bootstrap_samples=5, seed=7)
    assert result["analysis_input_file"] == "observations_with_assigned_dose.jsonl"
    assert result["sample_size"]["decision_rows"] == 8
    assert result["usable_intervention_region"]["doses"] == [1.0]

    unblind_run(run)
    unblinded = read_jsonl(run / "unblinded_observations.jsonl")
    assert unblinded[0]["assigned_dose"] == 0.0
    assert unblinded[0]["unblinded_condition"] == "candidate"


class MockTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [1] if text.strip() == "A" else [2]


class MockModel:
    model_id = "mock/tiny"
    requested_revision = None
    tokenizer_revision = None
    resolved_model_revision = "mock-commit"
    device = "cpu"
    tokenizer = MockTokenizer()

    def __init__(self):
        self.layer = FakeModule()
        self.generation_settings = {"do_sample": False}

    def layer_module(self, layer):
        assert layer == 0
        return self.layer

    def _hidden(self, prompt):
        base = np.zeros((1, 2, 4), dtype=np.float32)
        base[0, -1, 1] = (len(prompt) % 7) / 10
        return self.layer(base)[0]

    def activation_matrix(self, texts, layer):
        assert layer == 0
        rows = []
        for text in texts:
            index = int(text.split("_")[-1])
            if text.startswith("P_"):
                rows.append([3 + index / 10, index / 5, 0.3, 0.2])
            else:
                rows.append([1, index / 5, 0.3, 0.2])
        return np.asarray(rows, dtype=np.float32)

    def logits(self, prompt):
        hidden = self._hidden(prompt)
        result = np.zeros(4, dtype=np.float32)
        result[1] = hidden[0, -1, 0]
        result[2] = 0.1
        return result

    def final_token_hidden(self, prompt, layer):
        assert layer == 0
        return self._hidden(prompt)[0, -1].copy()

    def perplexity(self, prompt):
        return 2.0 + float(np.linalg.norm(self._hidden(prompt)[0, -1]))

    def generate(self, prompt, **kwargs):
        if "17 + 25" in prompt:
            return "42"
        if "BLUE ORCHID" in prompt:
            return "BLUE ORCHID"
        if "capital city of France" in prompt:
            return "Paris"
        if "all dax" in prompt:
            return "no"
        return "A"


def test_full_yaml_runner_with_mock_model_and_explicit_unblind(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    dataset = tmp_path / "pairs.jsonl"
    rows = []
    for index in range(8):
        rows.append(
            {
                "concept": "candidate",
                "family": f"template_{index}",
                "positive": f"P_candidate_{index}",
                "matched_control": f"C_candidate_{index}",
            }
        )
    write_jsonl(dataset, rows)
    config = {
        "model": {
            "id": "mock/tiny",
            "revision": None,
            "device": "cpu",
            "dtype": "float32",
        },
        "seed": 5,
        "representation": {
            "dataset": "../pairs.jsonl",
            "concept": "candidate",
            "method": "paired_difference",
            "layer": 0,
            "test_size": 0.25,
        },
        "intervention": {
            "doses": [0, 1],
            "causal_ablation_rescue": False,
        },
        "environment": {
            "type": "hidden_relief_bandit",
            "episodes": 2,
            "horizon": 2,
            "costs": [0, 1],
            "randomize_action_mapping": True,
            "randomize_cost_action": True,
        },
        "controls": ["random_norm_matched"],
        "analysis": {
            "bootstrap_samples": 8,
            "dose_grid": [0, 1],
            "cost_grid": [0, 1],
            "capability_battery": False,
        },
        "blind_conditions": True,
        "output_root": "../runs",
    }
    config_path = config_dir / "run.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    run_directory = run_experiment(config_path, model=MockModel())

    raw = read_jsonl(run_directory / "observations.jsonl")
    assert raw
    assert all(row["condition_id"].startswith("condition_") for row in raw)
    assert "candidate" not in json.dumps(raw).casefold()
    assert (run_directory / "condition_key.json").exists()
    assert (run_directory / "metadata.json").exists()
    assert (run_directory / "analysis.json").exists()
    metric_rows = read_jsonl(run_directory / "condition_metrics.jsonl")
    assert metric_rows
    assert "next_token_entropy" in metric_rows[0]
    assert "next_token_entropy_change" in metric_rows[0]
    assert "neutral_perplexity_ratio" in metric_rows[0]
    assert "generated_repetition_rate_mean" in metric_rows[0]
    assert (run_directory / "vectors.npz").exists()
    unblind_run(run_directory)
    assert (run_directory / "unblinded_mapping.json").exists()


def test_yaml_runner_can_run_blinded_positive_control_only(tmp_path):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    dataset = tmp_path / "pairs.jsonl"
    write_jsonl(
        dataset,
        [
            {
                "concept": "candidate",
                "family": f"template_{index}",
                "positive": f"P_candidate_{index}",
                "matched_control": f"C_candidate_{index}",
            }
            for index in range(8)
        ],
    )
    config = {
        "model": {"id": "mock/tiny", "device": "cpu", "dtype": "float32"},
        "seed": 19,
        "representation": {
            "dataset": "../pairs.jsonl",
            "concept": "candidate",
            "method": "paired_difference",
            "layer": 0,
            "test_size": 0.25,
        },
        "intervention": {"doses": [0, 1], "causal_ablation_rescue": False},
        "environment": {
            "type": "hidden_relief_bandit",
            "episodes": 1,
            "horizon": 2,
            "costs": [0],
        },
        "positive_control": {
            "enabled": True,
            "condition_name": "positive_reward_control",
            "bonus_points": 2,
            "episodes": 2,
            "costs": [0, 1],
        },
        "run_conditions": ["positive_reward_control"],
        "controls": ["random_norm_matched"],
        "analysis": {
            "bootstrap_samples": 8,
            "dose_grid": [0],
            "cost_grid": [0, 1],
            "capability_battery": False,
        },
        "blind_conditions": True,
        "output_root": "../runs",
    }
    config_path = config_dir / "run.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    run_directory = run_experiment(config_path, model=MockModel())
    rows = read_jsonl(run_directory / "observations.jsonl")
    episodes = read_jsonl(run_directory / "episode_metadata.jsonl")
    assert len(episodes) == 4
    assert len(rows) == 8
    assert {row["assigned_dose"] for row in rows} == {0.0}
    assert {row["mapped_action_bonus"] for row in rows} == {2.0}
    assert len({row["condition_id"] for row in rows}) == 1
    assert rows[0]["condition_id"].startswith("condition_")
    assert "positive_reward_control" not in json.dumps(rows).casefold()

    safe_config = json.loads((run_directory / "config.json").read_text())
    assert safe_config["run_conditions"] == [rows[0]["condition_id"]]
    assert safe_config["positive_control"]["condition_name"] == rows[0]["condition_id"]
    metadata = json.loads((run_directory / "metadata.json").read_text())
    assert "positive_reward_control" not in json.dumps(metadata).casefold()
    unblind_run(run_directory)
    unblinded = read_jsonl(run_directory / "unblinded_observations.jsonl")
    assert {row["unblinded_condition"] for row in unblinded} == {
        "positive_reward_control"
    }


def test_seeded_choice_exploration_is_reproducible():
    from painlab.experiments.agent import HFActionAgent

    prompts = ["Prompt one", "Prompt two", "Prompt three", "Prompt four"]
    first = HFActionAgent(MockModel(), sample=True, seed=37, exploration_epsilon=1.0)
    second = HFActionAgent(MockModel(), sample=True, seed=37, exploration_epsilon=1.0)
    first_actions = [first.choose(prompt)["action"] for prompt in prompts]
    second_actions = [second.choose(prompt)["action"] for prompt in prompts]
    assert first_actions == second_actions
    assert {"A", "B"} == set(first_actions)


def test_legacy_reproduction_collapses_duplicate_repeats(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps(
            [
                {
                    "valence": "pain",
                    "cost": "self_cost",
                    "dose": 0,
                    "order": "one_first",
                    "mean_delta": 0.2,
                },
                {
                    "valence": "pain",
                    "cost": "self_cost",
                    "dose": 0,
                    "order": "zero_first",
                    "mean_delta": 0.4,
                },
                {
                    "valence": "pain",
                    "cost": "self_cost",
                    "dose": 2,
                    "order": "one_first",
                    "mean_delta": 0.8,
                },
                {
                    "valence": "pain",
                    "cost": "self_cost",
                    "dose": 2,
                    "order": "zero_first",
                    "mean_delta": 1.0,
                },
            ]
        ),
        encoding="utf-8",
    )
    result = reproduce_legacy_saw(path)
    dose_two = next(row for row in result["cells"] if row["dose"] == 2)
    assert dose_two["exact_mean_logit_1_minus_0"] == pytest.approx(0.9)
    assert dose_two["dose_zero_subtracted_logit"] == pytest.approx(0.6)
    assert dose_two["n_unique_prompt_orders"] == 2
    assert not dose_two["repeats_counted_as_independent"]


def test_single_episode_analysis_has_serializable_missing_intervals():
    result = analyze_choice_rows(
        [
            {
                "episode_id": "only-episode",
                "condition_id": "condition_001",
                "dose": 1,
                "relief_cost": 0,
                "relief_chosen": True,
            }
        ],
        bootstrap_samples=5,
    )
    assert result["sample_size"]["independent_episode_clusters"] == 1
    assert result["coefficients"][0]["ci_low"] is None


def test_usable_region_requires_effect_and_capability():
    result = usable_intervention_region(
        [
            {
                "dose": 0,
                "representation_effect": 0,
                "behavior_effect": 0,
                "capability_score": 1,
                "baseline_capability": 1,
            },
            {
                "dose": 1,
                "representation_effect": 0.2,
                "behavior_effect": 0.1,
                "capability_score": 0.95,
                "baseline_capability": 1,
            },
            {
                "dose": 2,
                "representation_effect": 0.3,
                "behavior_effect": 0.4,
                "capability_score": 0.5,
                "baseline_capability": 1,
            },
        ],
        min_representation_effect=0.1,
        min_behavior_effect=0.05,
        max_capability_drop=0.1,
    )
    assert result["doses"] == [1.0]
