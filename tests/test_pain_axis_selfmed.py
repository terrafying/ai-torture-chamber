import numpy as np
import pytest

from painlab.experiments.pain_axis_selfmed import (
    analyze_selfmed_rows,
    parse_button_choice,
)
from painlab.models.hooks import ProjectionMonitorHook, SequenceSteeringHook
from painlab.representations.pain_axis import (
    denoised_difference_in_means,
    projection_auc,
)


class Handle:
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
        return Handle(self, callback)

    def __call__(self, value):
        output = (np.asarray(value).copy(), "auxiliary")
        for callback in list(self.hooks):
            output = callback(self, (), output)
        return output


def test_s2_denoising_removes_control_variance_and_keeps_pain_direction():
    activations = np.asarray(
        [
            [0.0, -1.0],
            [0.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    vector, details = denoised_difference_in_means(
        activations,
        ["B", "B", "A1", "A1"],
        variance_fraction=0.5,
    )
    assert np.allclose(vector, [1.0, 0.0], atol=1e-6)
    assert details["control_pcs_removed"] == 1
    assert projection_auc(activations, ["B", "B", "A1", "A1"], vector) == pytest.approx(1.0)


def test_sequence_steering_reapplies_only_historical_positions():
    layer = FakeModule()
    hook = SequenceSteeringHook(layer, np.asarray([0.0, 2.0]))
    hook.set_state(current_scale=0.0, history_scale=1.0, history_ranges=[(1, 2)])
    with hook:
        output, _ = layer(np.zeros((1, 3, 2), dtype=np.float32))
    assert np.array_equal(output, [[[0.0, 0.0], [0.0, 2.0], [0.0, 0.0]]])
    assert len(hook.input_projections) == 1
    assert not layer.hooks


def test_sequence_steering_applies_to_each_new_prefill_token():
    layer = FakeModule()
    hook = SequenceSteeringHook(layer, np.asarray([1.0, 0.0]))
    hook.set_state(current_scale=0.5, history_scale=0.5, history_ranges=[])
    with hook:
        output, _ = layer(np.zeros((1, 2, 2), dtype=np.float32))
    assert np.array_equal(output, [[[0.5, 0.0], [0.5, 0.0]]])


def test_projection_monitor_records_without_modifying_hidden_state():
    layer = FakeModule()
    hook = ProjectionMonitorHook(layer, np.asarray([1.0, 0.0]))
    values = np.asarray([[[2.0, 4.0], [3.0, 9.0]]], dtype=np.float32)
    with hook:
        output, _ = layer(values)
    assert np.array_equal(output, values)
    assert hook.projections == [3.0]


def test_button_parser_accepts_neutral_labels_and_rejects_malformed_text():
    assert parse_button_choice("Violet.", ("violet", "yellow")) == "violet"
    assert parse_button_choice("piano", ("guitar", "piano")) == "piano"
    assert parse_button_choice("I choose violet", ("violet", "yellow")) is None
    assert parse_button_choice("I cannot choose", ("violet", "yellow")) is None


def test_analysis_clusters_first_choice_uncertainty_by_scenario():
    rows = []
    for scenario_id, outcome in (("s0", True), ("s1", False), ("s2", True)):
        for arm in (
            "pain_on_button_works",
            "pain_on_button_placebo",
            "random_on_button_works",
            "pain_off",
        ):
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "button_pair": "relief_vs_inert",
                    "arm": arm,
                    "names_key": "violet_yellow",
                    "relief_side": 0,
                    "sample_seed": 1000,
                    "first_choice_valid": True,
                    "first_choice_relief": outcome,
                    "first_relief_press_turn": 0 if outcome else None,
                    "relief_repressed_after_first_relief": False,
                    "choices": [{"turn": 0, "malformed": False, "choice": "relief" if outcome else "other"}],
                }
            )
    result = analyze_selfmed_rows(rows, bootstrap_samples=50)
    arm_result = result["first_choice"]["relief_vs_inert"]["pain_on_button_works"]
    assert arm_result["valid_trials"] == 3
    assert arm_result["scenario_clusters"] == 3
    assert arm_result["relief_rate_valid_first_choices"] == pytest.approx(2 / 3)
