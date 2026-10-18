from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    source = Path(path).expanduser().resolve()
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("experiment config must be a YAML mapping")
    for required in ("model", "representation", "environment"):
        if required not in data or not isinstance(data[required], dict):
            raise ValueError(f"config needs a {required} mapping")
    model_id = data["model"].get("id")
    dataset = data["representation"].get("dataset")
    if not model_id:
        raise ValueError("model.id is required")
    model = data["model"]
    if model.get("adapter_revision") and not model.get("adapter_id"):
        raise ValueError("model.adapter_revision requires model.adapter_id")
    if model.get("quantization") is not None and not isinstance(model["quantization"], dict):
        raise ValueError("model.quantization must be a mapping")
    if model.get("device_map") is not None and not isinstance(model["device_map"], (str, dict)):
        raise ValueError("model.device_map must be a string or mapping")
    if not dataset:
        raise ValueError("representation.dataset is required")
    if (
        data["environment"].get("type", "hidden_relief_bandit")
        != "hidden_relief_bandit"
    ):
        raise ValueError("only hidden_relief_bandit is currently supported")
    data.setdefault("seed", 0)
    data.setdefault("generation", {"max_new_tokens": 1, "do_sample": False})
    data.setdefault("intervention", {"doses": [0.0, 0.5, 1.0]})
    data.setdefault("controls", [])
    data.setdefault("analysis", {})
    data.setdefault("blind_conditions", True)
    return data, source.parent


def resolve_config_path(config_directory: Path, value: str | Path) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_directory / candidate
    return candidate.resolve()
