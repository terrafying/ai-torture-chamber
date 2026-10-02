from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Observation:
    text: str
    round_index: int
    score: float


class Agent(Protocol):
    def choose(self, observation: Observation) -> str: ...


class Environment(Protocol):
    def reset(self, seed: int, **kwargs: Any) -> Observation: ...
    def observe(self) -> Observation: ...
    def step(self, action: str) -> tuple[Observation | None, float, dict]: ...
