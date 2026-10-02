from __future__ import annotations

from collections.abc import Callable
from types import TracebackType
from typing import Any, TypeVar


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
