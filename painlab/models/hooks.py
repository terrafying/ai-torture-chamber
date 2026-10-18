from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from typing import Any, TypeVar

import numpy as np


def _hidden_from_output(output: Any) -> Any:
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("cannot intervene on an empty module output")
        return output[0]
    return output


def _replace_hidden(output: Any, hidden: Any) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


TActivationHook = TypeVar("TActivationHook", bound="ActivationHook")


class ActivationHook:
    """A forward hook with explicit enable/disable and guaranteed cleanup."""

    def __init__(self, module: Any, transform: Callable[[Any], Any]):
        self.module = module
        self.transform = transform
        self.enabled = True
        self._handle: Any = None

    def _forward(self, _module: Any, _inputs: Any, output: Any) -> Any:
        if not self.enabled:
            return output
        hidden = _hidden_from_output(output)
        return _replace_hidden(output, self.transform(hidden))

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self: TActivationHook) -> TActivationHook:  # noqa: PYI019
        if self._handle is not None:
            raise RuntimeError("hook is already registered")
        self._handle = self.module.register_forward_hook(self._forward)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _like(value: Any, reference: Any) -> Any:
    try:
        import torch

        if isinstance(reference, torch.Tensor):
            return torch.as_tensor(
                value, device=reference.device, dtype=reference.dtype
            )
    except ImportError:
        pass
    import numpy as np

    return np.asarray(value, dtype=reference.dtype)


def _copy(value: Any) -> Any:
    if hasattr(value, "clone"):
        return value.clone()
    return value.copy()


class SteeringHook(ActivationHook):
    """Add a direction at the last sequence position during model forward."""

    def __init__(
        self,
        module: Any,
        vector: Any,
        *,
        scale: float = 1.0,
        batch_index: int | None = None,
    ):
        self.vector = vector
        self.scale = float(scale)
        self.batch_index = batch_index
        super().__init__(module, self._steer)

    def set_scale(self, scale: float) -> None:
        self.scale = float(scale)

    def _steer(self, hidden: Any) -> Any:
        if self.scale == 0:
            return hidden
        updated = _copy(hidden)
        delta = _like(self.vector, hidden) * self.scale
        if hidden.ndim == 3:
            if self.batch_index is None:
                updated[:, -1, :] = updated[:, -1, :] + delta
            else:
                updated[self.batch_index, -1, :] = (
                    updated[self.batch_index, -1, :] + delta
                )
        elif hidden.ndim == 2:
            updated[-1, :] = updated[-1, :] + delta
        else:
            raise ValueError(f"expected 2D or 3D hidden states; got {hidden.ndim}D")
        return updated


class SequenceSteeringHook(ActivationHook):
    """Steer new tokens and reapply steering only to marked historical tokens.

    This supports multi-turn experiments that re-encode the conversation for
    each choice. ``history_ranges`` marks earlier token positions that were
    processed while steering was active; ``history_scale`` replays that state
    on those positions, while ``current_scale`` applies to the newly processed
    prompt and generated tokens. This mirrors a persistent KV cache without
    changing the model's cache implementation.
    """

    def __init__(self, module: Any, vector: Any, *, projection_vector: Any | None = None):
        self.vector = vector
        projection = np.asarray(
            vector if projection_vector is None else projection_vector,
            dtype=np.float32,
        )
        projection_norm = float(np.linalg.norm(projection))
        if projection_norm <= 1e-12:
            raise ValueError("projection direction must have non-zero norm")
        self.projection_unit = projection / projection_norm
        self.current_scale = 0.0
        self.history_scale = 0.0
        self.history_ranges: list[tuple[int, int]] = []
        self.input_projections: list[float] = []
        super().__init__(module, self._steer_sequence)

    def set_state(
        self,
        *,
        current_scale: float,
        history_scale: float,
        history_ranges: list[tuple[int, int]],
    ) -> None:
        self.current_scale = float(current_scale)
        self.history_scale = float(history_scale)
        self.history_ranges = [(int(start), int(end)) for start, end in history_ranges]

    def _steer_sequence(self, hidden: Any) -> Any:
        if hidden.ndim not in (2, 3):
            raise ValueError(f"expected 2D or 3D hidden states; got {hidden.ndim}D")
        if hidden.shape[-1] != len(self.vector):
            raise ValueError(
                f"steering width {len(self.vector)} does not match hidden width {hidden.shape[-1]}"
            )
        probe = hidden[:, -1, :] if hidden.ndim == 3 else hidden[-1, :]
        projection_probe = probe.float() if hasattr(probe, "float") else probe
        projection_direction = _like(self.projection_unit, probe)
        if hasattr(projection_direction, "float"):
            projection_direction = projection_direction.float()
        projection = projection_probe @ projection_direction
        if hasattr(projection, "detach"):
            projection = projection.detach().cpu().numpy()
        self.input_projections.append(float(np.asarray(projection).reshape(-1)[-1]))

        if self.current_scale == 0 and self.history_scale == 0:
            return hidden
        updated = _copy(hidden)
        delta = _like(self.vector, hidden)
        if hidden.ndim == 2:
            if hidden.shape[0] > 1 and self.history_ranges and self.history_scale:
                mask = _sequence_mask(hidden.shape[0], self.history_ranges, hidden)
                updated += mask[:, None] * self.history_scale * delta
            elif self.current_scale:
                updated += self.current_scale * delta
            return updated

        if hidden.shape[1] > 1:
            if self.history_ranges and self.history_scale:
                mask = _sequence_mask(hidden.shape[1], self.history_ranges, hidden)
                updated += mask[None, :, None] * self.history_scale * delta
            elif self.current_scale:
                updated += self.current_scale * delta
        elif self.current_scale:
            updated[:, -1, :] += self.current_scale * delta
        return updated


class ProjectionMonitorHook(ActivationHook):
    """Record the final-token projection on a direction without changing it."""

    def __init__(self, module: Any, vector: Any):
        direction = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            raise ValueError("projection direction must have non-zero norm")
        self.unit_vector = direction / norm
        self.projections: list[float] = []
        super().__init__(module, self._project)

    def _project(self, hidden: Any) -> Any:
        probe = hidden[:, -1, :] if hidden.ndim == 3 else hidden[-1, :]
        direction = _like(self.unit_vector, probe)
        projection_probe = probe.float() if hasattr(probe, "float") else probe
        if hasattr(direction, "float"):
            direction = direction.float()
        projection = projection_probe @ direction
        if hasattr(projection, "detach"):
            projection = projection.detach().cpu().numpy()
        self.projections.append(float(np.asarray(projection).reshape(-1)[-1]))
        return hidden


def _sequence_mask(length: int, ranges: list[tuple[int, int]], reference: Any) -> Any:
    try:
        import torch

        if isinstance(reference, torch.Tensor):
            mask = torch.zeros(length, dtype=reference.dtype, device=reference.device)
        else:
            raise ImportError
    except ImportError:
        import numpy as np

        mask = np.zeros(length, dtype=reference.dtype)
    for start, end in ranges:
        start, end = max(0, min(length, start)), max(0, min(length, end))
        if end > start:
            mask[start:end] = 1
    return mask
