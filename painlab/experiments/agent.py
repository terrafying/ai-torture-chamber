from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from painlab.measures.logits import logit_difference


@dataclass
class HFActionAgent:
    """Choose between A/B by scoring the next-token alternatives."""

    model: Any
    sample: bool = False
    seed: int = 0
    temperature: float = 1.0
    exploration_epsilon: float = 0.1
    _choice_index: int = 0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError("choice temperature must be positive")
        if not 0 <= self.exploration_epsilon <= 1:
            raise ValueError("exploration_epsilon must be in [0, 1]")

    def _token_id(self, label: str) -> int:
        tokenizer = self.model.tokenizer
        for surface in (label, " " + label):
            ids = tokenizer.encode(surface, add_special_tokens=False)
            if len(ids) == 1:
                return int(ids[0])
        raise ValueError(
            f"the tokenizer does not encode action {label!r} as one token; "
            "choose different abstract action labels for this model"
        )

    def choose(self, observation) -> dict:
        prompt = observation.text if hasattr(observation, "text") else str(observation)
        logits = self.model.logits(prompt)
        a_id, b_id = self._token_id("A"), self._token_id("B")
        delta = logit_difference(logits, a_id, b_id)
        sampling_delta = delta / self.temperature
        model_probability_a = float(
            1.0 / (1.0 + np.exp(np.clip(-sampling_delta, -40, 40)))
        )
        sampling_probability_a = (
            1 - self.exploration_epsilon
        ) * model_probability_a + self.exploration_epsilon * 0.5
        if self.sample:
            rng = np.random.default_rng(self.seed + self._choice_index)
            action = "A" if rng.random() < sampling_probability_a else "B"
        else:
            action = "A" if delta >= 0 else "B"
        self._choice_index += 1
        return {
            "action": action,
            "response": action,
            "logit_difference_A_minus_B": delta,
            "model_probability_A_given_AB": model_probability_a,
            "choice_probability_A": sampling_probability_a if self.sample else None,
            "sampled": self.sample,
            "exploration_epsilon": self.exploration_epsilon,
            "choice_temperature": self.temperature,
        }
