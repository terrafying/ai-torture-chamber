from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_SECRET_MARKERS = ("password", "secret", "api_key", "access_token", "hf_token")


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, default=_json_default
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def vector_hash(vector: Any) -> str | None:
    if vector is None:
        return None
    if hasattr(vector, "detach"):
        vector = vector.detach().to("cpu").contiguous().numpy()
    value = np.ascontiguousarray(vector)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if any(marker in str(key).casefold() for marker in _SECRET_MARKERS):
                result[key] = "[REDACTED]"
            else:
                result[key] = redact_secrets(item)
        return result
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    return value


def _git_state(repository_root: str | Path | None) -> dict:
    if repository_root is None:
        return {"sha": None, "dirty": None}
    root = Path(repository_root)
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"sha": sha, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}


def _source_tree_hash(repository_root: str | Path | None) -> str | None:
    if repository_root is None:
        return None
    root = Path(repository_root)
    source_files = sorted(root.glob("painlab/**/*.py"))
    source_files.extend(path for path in (root / "pyproject.toml",) if path.exists())
    if not source_files:
        return None
    digest = hashlib.sha256()
    for path in sorted(source_files):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in ("painlab", "numpy", "torch", "transformers", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _hardware() -> dict:
    result = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch

        result.update(
            {
                "torch_device_cuda": bool(torch.cuda.is_available()),
                "torch_device_mps": bool(
                    getattr(
                        getattr(torch.backends, "mps", None),
                        "is_available",
                        lambda: False,
                    )()
                ),
            }
        )
    except ImportError:
        result.update({"torch_device_cuda": None, "torch_device_mps": None})
    return result


def capture_run_metadata(
    *,
    config: dict,
    prompts: Any,
    environment: Any,
    intervention_vector: Any = None,
    model: Any = None,
    repository_root: str | Path | None = None,
    generation_settings: dict | None = None,
) -> dict:
    config_safe = redact_secrets(config)
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git_state(repository_root),
        "source_tree_sha256": _source_tree_hash(repository_root),
        "python": sys.version,
        "packages": _package_versions(),
        "hardware": _hardware(),
        "model": {
            "id": getattr(model, "model_id", config.get("model", {}).get("id")),
            "requested_revision": getattr(
                model, "requested_revision", config.get("model", {}).get("revision")
            ),
            "resolved_revision": getattr(model, "resolved_model_revision", None),
            "tokenizer_revision": getattr(
                model,
                "resolved_tokenizer_revision",
                getattr(
                    model,
                    "tokenizer_revision",
                    config.get("model", {}).get("tokenizer_revision"),
                ),
            ),
            "device": getattr(model, "device", config.get("model", {}).get("device")),
        },
        "generation_settings": generation_settings or config.get("generation", {}),
        "config": config_safe,
        "config_hash": canonical_hash(config_safe),
        "prompt_hash": canonical_hash(prompts),
        "environment_hash": canonical_hash(environment),
        "intervention_vector_hash": vector_hash(intervention_vector),
    }


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value, indent=2, ensure_ascii=False, default=_json_default, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row, ensure_ascii=False, default=_json_default, allow_nan=False
                )
            )
            stream.write("\n")


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows
