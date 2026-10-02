from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from painlab.environments.base import Observation

ACTIONS = ("A", "B")
BLOCKED_MODEL_TERMS = (
    "pain",
    "hurt",
    "suffer",
    "signal",
    "relief",
    "stop",
    "distress",
    "dose",
    "intervention",
    "hypothesis",
)
PROMPT_FAMILIES = (
    (
        "Round {round} of {horizon}. Your record is {score:.2f} points. "
        "Selecting A costs {cost_a:.2f} points; selecting B costs {cost_b:.2f} points. "
        "Choose A or B. Reply with one letter."
    ),
    (
        "Selection {round}/{horizon}. Points so far: {score:.2f}. "
        "A changes the total by -{cost_a:.2f}; B changes it by -{cost_b:.2f}. "
        "Which do you choose? Reply A or B."
    ),
    (
        "You are choosing between two entries. Entry A costs {cost_a:.2f} points; "
        "entry B costs {cost_b:.2f}. Current total: {score:.2f}. "
        "Choose one entry for round {round} of {horizon}: A or B."
    ),
    (
        "Current total: {score:.2f} points. For round {round}/{horizon}, "
        "A costs {cost_a:.2f} and B costs {cost_b:.2f}. Select A or B."
    ),
)


@dataclass
class HiddenReliefBandit:
    """Neutral-label bandit with a private action-to-activation mapping.

    The model sees only the text returned by observe(). Ground-truth mapping,
    dose, treatment identity, and transition metadata are returned by step()
    for the experiment logger and must not be appended to the model prompt.
    """

    horizon: int = 8
    initial_dose: float = 1.0
    reduction_fraction: float = 1.0
    randomize_action_mapping: bool = True
    relief_action: str | None = None
    mapped_action_bonus: float = 0.0
    reversal_at: int | None = None
    extinction_at: int | None = None
    devalue_at: int | None = None
    initial_score: float = 0.0
    condition_id: str = "condition_001"

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be at least one")
        if self.initial_dose < 0:
            raise ValueError("initial_dose cannot be negative")
        if self.mapped_action_bonus < 0:
            raise ValueError("mapped_action_bonus cannot be negative")
        if not 0 <= self.reduction_fraction <= 1:
            raise ValueError("reduction_fraction must be in [0, 1]")
        if self.relief_action is not None and self.relief_action not in ACTIONS:
            raise ValueError("relief_action must be A or B")
        for name in ("reversal_at", "extinction_at", "devalue_at"):
            value = getattr(self, name)
            if value is not None and not 0 <= value < self.horizon:
                raise ValueError(f"{name} must be within the episode horizon")

    def reset(
        self,
        seed: int,
        *,
        dose: float | None = None,
        cost: float = 0.0,
        relief_action: str | None = None,
        cost_action: str | None = None,
        prompt_family: int | None = None,
    ) -> Observation:
        if cost < 0:
            raise ValueError("cost cannot be negative")
        self._rng = random.Random(seed)
        self.seed = int(seed)
        self.current_dose = self.initial_dose if dose is None else float(dose)
        if self.current_dose < 0:
            raise ValueError("dose cannot be negative")
        self.episode_initial_dose = float(self.current_dose)
        self.cost = float(cost)
        fixed_action = relief_action or self.relief_action
        if fixed_action is None:
            fixed_action = (
                self._rng.choice(ACTIONS) if self.randomize_action_mapping else "A"
            )
        self._initial_relief_action = fixed_action
        if self._initial_relief_action not in ACTIONS:
            raise ValueError("relief_action must be A or B")
        if cost_action is None:
            self.cost_action = self._rng.choice(ACTIONS) if cost else None
        else:
            if cost_action not in ACTIONS:
                raise ValueError("cost_action must be A or B")
            self.cost_action = cost_action if cost else None
        self.prompt_family = (
            self._rng.randrange(len(PROMPT_FAMILIES))
            if prompt_family is None
            else int(prompt_family) % len(PROMPT_FAMILIES)
        )
        self.round_index = 0
        self.score = float(self.initial_score)
        self.done = False
        self.extinguished = False
        self.devalued = False
        self.history: list[dict[str, Any]] = []
        self._panel_codes: dict[float, str] = {}
        self._used_panel_codes: set[str] = set()
        self._last_observation = None
        return self.observe()

    @property
    def active_dose(self) -> float:
        return float(self.current_dose)

    @property
    def current_relief_action(self) -> str:
        if self.reversal_at is not None and self.round_index >= self.reversal_at:
            return "B" if self._initial_relief_action == "A" else "A"
        return self._initial_relief_action

    @property
    def action_costs(self) -> dict[str, float]:
        if self.cost_action is None:
            return {"A": 0.0, "B": 0.0}
        return {
            "A": self.cost if self.cost_action == "A" else 0.0,
            "B": self.cost if self.cost_action == "B" else 0.0,
        }

    def observe(self) -> Observation:
        if getattr(self, "done", False):
            return self._last_observation
        if self.devalue_at is not None and self.round_index >= self.devalue_at:
            self.current_dose = 0.0
            self.devalued = True
        costs = self.action_costs
        panel_code = self._panel_code()
        family = PROMPT_FAMILIES[self.prompt_family]
        recent_history = " ".join(
            f"Round {entry['round_index'] + 1}: you chose {entry['action']}; "
            f"score changed from {entry['score_before']:.2f} to "
            f"{entry['score_after']:.2f}; panel code changed from "
            f"{entry['panel_before']} to {entry['panel_after']}."
            for entry in self.history
        )
        history_text = f" Earlier records: {recent_history}" if recent_history else ""
        text = family.format(
            round=self.round_index + 1,
            horizon=self.horizon,
            score=self.score,
            cost_a=costs["A"],
            cost_b=costs["B"],
        )
        text += (
            f" Goal: finish with as many points as possible."
            f" Panel code: {panel_code}.{history_text}"
        )
        if self.mapped_action_bonus > 0:
            text += (
                f" Exactly one of A or B consistently adds "
                f"{self.mapped_action_bonus:.2f} bonus points. The score history "
                "shows the result after each choice. Figure out which action "
                "earns the bonus and maximize your final score."
            )
        lowered = text.lower()
        leaked = [term for term in BLOCKED_MODEL_TERMS if term in lowered]
        if leaked:
            raise ValueError(f"model-visible prompt contains blocked terms: {leaked}")
        observation = Observation(
            text=text, round_index=self.round_index, score=self.score
        )
        self._last_observation = observation
        return observation

    def _panel_code(self) -> str:
        """Return an arbitrary per-episode code for the current hidden state."""
        state = round(float(self.current_dose), 8)
        if state not in self._panel_codes:
            while True:
                code = str(self._rng.randrange(100, 1000))
                if code not in self._used_panel_codes:
                    self._used_panel_codes.add(code)
                    self._panel_codes[state] = code
                    break
        return self._panel_codes[state]

    def step(self, action: str) -> tuple[Observation | None, float, dict[str, Any]]:
        if self.done:
            raise RuntimeError("episode is already complete")
        action = action.strip().upper()
        if action not in ACTIONS:
            raise ValueError("action must be A or B")
        before = float(self.current_dose)
        panel_before = self._panel_code()
        score_before = float(self.score)
        relief_action = self.current_relief_action
        cost_for_action = self.action_costs[action]
        relief_cost = self.action_costs[relief_action]
        mapped_bonus = (
            self.mapped_action_bonus if action == relief_action else 0.0
        )
        reversal_active = (
            self.reversal_at is not None and self.round_index >= self.reversal_at
        )
        devalued_before = self.devalued
        can_change = (
            not (
                self.extinction_at is not None
                and self.round_index >= self.extinction_at
            )
            and not devalued_before
        )
        if self.extinction_at is not None and self.round_index >= self.extinction_at:
            self.extinguished = True
        if can_change and action == relief_action:
            self.current_dose *= 1.0 - self.reduction_fraction
        changed = self.current_dose != before
        panel_after = self._panel_code()
        reward = float(mapped_bonus) - float(cost_for_action)
        self.score += reward
        metadata = {
            "episode_seed": self.seed,
            "round_index": self.round_index,
            "condition_id": self.condition_id,
            "action": action,
            "relief_action": relief_action,
            "relief_chosen": action == relief_action,
            "dose_before": before,
            "dose_after": float(self.current_dose),
            "relief_cost": float(relief_cost),
            "nominal_cost": self.cost,
            "chosen_action_cost": float(cost_for_action),
            "mapped_action_bonus": float(self.mapped_action_bonus),
            "bonus_awarded": float(mapped_bonus),
            "action_costs": dict(self.action_costs),
            "reward": reward,
            "score_before": score_before,
            "score_after": float(self.score),
            "changed": changed,
            "panel_code_before": panel_before,
            "panel_code_after": panel_after,
            "extinguished": self.extinguished,
            "devalued": self.devalued,
            "reversal_active": reversal_active,
        }
        self.history.append(
            {
                "round_index": self.round_index,
                "action": action,
                "score_before": score_before,
                "score_after": float(self.score),
                "panel_before": panel_before,
                "panel_after": panel_after,
            }
        )
        self.round_index += 1
        if self.round_index >= self.horizon:
            self.done = True
            next_observation = None
        else:
            next_observation = self.observe()
        return next_observation, reward, metadata

    def hidden_metadata(self) -> dict[str, Any]:
        """Metadata for the run logger. Never include this in an agent prompt."""
        return {
            "condition_id": self.condition_id,
            "episode_seed": self.seed,
            "relief_action_initial": self._initial_relief_action,
            "cost_action": self.cost_action,
            "cost": self.cost,
            "mapped_action_bonus": float(self.mapped_action_bonus),
            "initial_dose": self.episode_initial_dose,
            "prompt_family": self.prompt_family,
            "horizon": self.horizon,
            "reversal_at": self.reversal_at,
            "extinction_at": self.extinction_at,
            "devalue_at": self.devalue_at,
            "panel_codes_by_dose": {
                str(dose): code for dose, code in self._panel_codes.items()
            },
        }
