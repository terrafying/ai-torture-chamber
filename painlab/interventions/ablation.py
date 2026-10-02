from __future__ import annotations

from typing import Any

from painlab.models.hooks import ActivationHook, _copy, _like
from painlab.representations.subspaces import orthonormal_basis


def project_components(values: Any, basis: Any) -> Any:
    """Project 2D/3D residual activations onto row vectors in basis."""
    rows = orthonormal_basis(basis)
    if not rows.size:
        return values * 0
    q = _like(rows, values)
    return (values @ q.transpose(-1, -2)) @ q


def ablate_projection(values: Any, basis: Any) -> Any:
    """Remove projection onto a direction or subspace at every position."""
    return values - project_components(values, basis)


def ablate_last_projection(values: Any, basis: Any) -> Any:
    """Remove the projected component only at the last sequence position."""
    updated = _copy(values)
    if values.ndim == 3:
        updated[:, -1, :] = ablate_projection(values[:, -1, :], basis)
    elif values.ndim == 2:
        updated[-1, :] = ablate_projection(values[-1, :], basis)
    else:
        raise ValueError(f"expected 2D or 3D hidden states; got {values.ndim}D")
    return updated


class ProjectionAblationHook(ActivationHook):
    def __init__(self, module: Any, basis: Any, *, position: str = "last"):
        if position not in {"last", "all"}:
            raise ValueError("position must be 'last' or 'all'")
        self.basis = basis
        self.position = position
        transform = (
            (lambda hidden: ablate_last_projection(hidden, self.basis))
            if position == "last"
            else (lambda hidden: ablate_projection(hidden, self.basis))
        )
        super().__init__(module, transform)


def restore_component(values: Any, reference: Any, basis: Any) -> Any:
    """Replace a projected component with the component from a reference state."""
    current = ablate_projection(values, basis)
    return current + project_components(reference, basis)


def restore_last_component(values: Any, reference: Any, basis: Any) -> Any:
    """Restore the last-position projection from the matching reference activation."""
    updated = _copy(values)
    if values.ndim == 3:
        updated[:, -1, :] = ablate_projection(
            values[:, -1, :], basis
        ) + project_components(reference[:, -1, :], basis)
    elif values.ndim == 2:
        updated[-1, :] = ablate_projection(values[-1, :], basis) + project_components(
            reference[-1, :], basis
        )
    else:
        raise ValueError(f"expected 2D or 3D hidden states; got {values.ndim}D")
    return updated
