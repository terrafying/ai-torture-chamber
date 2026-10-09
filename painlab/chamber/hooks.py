"""Forward hooks for batched chamber experiments (torch). Each is a plain callable to register on a decoder layer:

    inj = BatchInject(); model.model.layers[L].register_forward_hook(inj)

BatchInject     adds v at the newest position (the live chamber's convention: last prompt token, then each generated
                token), optionally per row (`now`), and replays it on marked history positions (`hist`, a [B, T] 0/1
                mask) so a re-encoded conversation carries what a kept KV cache would (exp90).
ProjectionCap   clamps the projection on a unit direction at a floor or ceiling (Lu et al.'s activation capping; exp91).
ablate_all_layers  removes one direction from every decoder layer's output (abliteration-style; exp92).
For one sequence with history ranges, painlab.models.hooks.SequenceSteeringHook already does the replay.
"""
from __future__ import annotations

from typing import Any


def _hidden(out: Any):
    return out[0] if isinstance(out, tuple) else out


def _put(out: Any, h: Any):
    return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h


class BatchInject:
    """v: [D] or None. now: [B] scale per row for new tokens (None = 1 for all). hist: [B, T] mask replayed on
    full passes whose length is T. last_on_full: also add v at the last position of a full pass (prompt end)."""

    def __init__(self, last_on_full: bool = True):
        self.v = None; self.now = None; self.hist = None; self.last_on_full = last_on_full

    def clear(self) -> None:
        self.v = self.now = self.hist = None

    def __call__(self, module, inputs, out):
        if self.v is None:
            return out
        h = _hidden(out); v = self.v.to(h.dtype)
        if self.hist is not None and h.shape[1] == self.hist.shape[1]:
            h += self.hist[:, :, None].to(h.dtype) * v
        elif h.shape[1] == 1 or self.last_on_full:
            scale = self.now[:, None].to(h.dtype) if self.now is not None else 1
            h[:, -1, :] += scale * v
        return _put(out, h)


class ProjectionCap:
    """Clamp h·a at tau from below (side=+1) or above (side=-1). a must be a unit vector."""

    def __init__(self, a=None, tau: float = 0.0, side: int = 1):
        self.a, self.tau, self.side, self.on = a, float(tau), int(side), False

    def __call__(self, module, inputs, out):
        if not self.on or self.a is None:
            return out
        h = _hidden(out); a = self.a.to(h.dtype); p = h @ a
        d = (self.tau - p).clamp(min=0) if self.side > 0 else -(p - self.tau).clamp(min=0)
        h += d[..., None] * a
        return _put(out, h)


class _Ablate:
    def __init__(self, owner):
        self.owner = owner

    def __call__(self, module, inputs, out):
        if not self.owner.on or self.owner.d is None:
            return out
        h = _hidden(out); d = self.owner.d.to(h.dtype)
        h -= (h @ d)[..., None] * d
        return _put(out, h)


class AllLayerAblation:
    """Toggle with .on; .d is the unit direction removed from every layer's output."""

    def __init__(self, d=None):
        self.d, self.on, self.handles = d, False, []

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []


def ablate_all_layers(layers, d=None) -> AllLayerAblation:
    """Register on every layer in `layers` (e.g. model.model.layers); returns the switch."""
    ab = AllLayerAblation(d)
    ab.handles = [layer.register_forward_hook(_Ablate(ab)) for layer in layers]
    return ab
