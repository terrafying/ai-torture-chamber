from __future__ import annotations

from collections.abc import Callable
from typing import Any

from painlab.models.hooks import ActivationHook


class ActivationPatchingHook(ActivationHook):
    """Run a caller-supplied activation patch under a managed hook lifecycle."""

    def __init__(self, module: Any, patch: Callable[[Any], Any]):
        super().__init__(module, patch)


def rescue_projection(
    module: Any, basis: Any, reference_activation: Any
) -> ActivationPatchingHook:
    from painlab.interventions.ablation import restore_component

    return ActivationPatchingHook(
        module, lambda hidden: restore_component(hidden, reference_activation, basis)
    )
