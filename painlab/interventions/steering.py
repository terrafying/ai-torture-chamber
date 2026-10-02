from __future__ import annotations

from collections.abc import Callable
from typing import Any

from painlab.models.hooks import ActivationHook, SteeringHook


def add_steering(
    module: Any, vector: Any, *, scale: float = 1.0, batch_index: int | None = None
) -> SteeringHook:
    return SteeringHook(module, vector, scale=scale, batch_index=batch_index)


def apply_transform(module: Any, transform: Callable[[Any], Any]) -> ActivationHook:
    return ActivationHook(module, transform)
