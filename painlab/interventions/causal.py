from __future__ import annotations

from contextlib import ExitStack
from typing import Any, ClassVar

from painlab.interventions.ablation import (
    ablate_last_projection,
    restore_last_component,
)
from painlab.models.hooks import ActivationHook, SteeringHook, _copy


class CausalProjectionHooks:
    """Composable induce/ablate/rescue hooks on one model layer.

    In rescue mode, a capture hook records the activation after induction and
    restores its projection after ablation on the same forward pass.
    """

    MODES: ClassVar[frozenset[str]] = frozenset({"induce", "ablate", "rescue"})

    def __init__(self, module: Any, vector: Any, basis: Any, *, scale: float = 0.0):
        self.vector = vector
        self.basis = basis
        self.mode = "induce"
        self.steering = SteeringHook(module, vector, scale=scale)
        self.reference = None
        self.capture = ActivationHook(module, self._capture_reference)
        self.ablation = ActivationHook(
            module, lambda hidden: ablate_last_projection(hidden, self.basis)
        )
        self.rescue = ActivationHook(module, self._restore_reference)
        self._stack = None

    def _capture_reference(self, hidden):
        self.reference = _copy(hidden)
        return hidden

    def _restore_reference(self, hidden):
        if self.reference is None:
            raise RuntimeError(
                "rescue requested before an induced reference was captured"
            )
        return restore_last_component(hidden, self.reference, self.basis)

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        self.mode = mode
        active = self.steering.scale != 0
        self.ablation.enabled = active and mode in {"ablate", "rescue"}
        self.rescue.enabled = active and mode == "rescue"

    def set_scale(self, scale: float) -> None:
        self.steering.set_scale(scale)
        self.set_mode(self.mode)

    def set_condition(self, vector: Any, basis: Any | None = None) -> None:
        self.vector = vector
        self.steering.vector = vector
        if basis is not None:
            self.basis = basis

    def __enter__(self):
        if self._stack is not None:
            raise RuntimeError("causal hooks are already registered")
        self._stack = ExitStack()
        self._stack.enter_context(self.steering)
        self._stack.enter_context(self.capture)
        self._stack.enter_context(self.ablation)
        self._stack.enter_context(self.rescue)
        self.set_mode(self.mode)
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self._stack is not None:
            self._stack.close()
            self._stack = None
