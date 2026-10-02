from __future__ import annotations

import random
from typing import Any

import numpy as np


def seed_everything(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed Python, NumPy, and Torch when Torch is installed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError):
            pass


def generation_kwargs(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = dict(settings or {})
    allowed = {
        "max_new_tokens",
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "repetition_penalty",
        "num_beams",
        "eos_token_id",
        "pad_token_id",
    }
    unknown = set(settings) - allowed
    if unknown:
        raise ValueError(f"unsupported generation settings: {sorted(unknown)}")
    return settings
