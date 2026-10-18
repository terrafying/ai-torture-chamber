from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from painlab.provenance.run_metadata import read_jsonl, write_jsonl


def blind_condition_names(
    names: list[str], *, seed: int = 0
) -> tuple[dict[str, str], dict[str, str]]:
    rng = random.Random(seed)
    shuffled = list(names)
    rng.shuffle(shuffled)
    forward = {
        actual: f"condition_{index + 1:03d}" for index, actual in enumerate(shuffled)
    }
    reverse = {blind: actual for actual, blind in forward.items()}
    return forward, reverse


def unblind_run(run_directory: str | Path) -> dict:
    root = Path(run_directory)
    key_path = root / "condition_key.json"
    if not key_path.exists():
        raise FileNotFoundError("no blinded condition key exists for this run")
    key = json.loads(key_path.read_text(encoding="utf-8"))
    record = {
        "unblinded_at_utc": datetime.now(timezone.utc).isoformat(),
        "mapping": key["blinded_to_condition"],
    }
    observations_name = "observations.jsonl"
    artifacts_path = root / "artifacts.json"
    if artifacts_path.exists():
        artifacts = json.loads(artifacts_path.read_text(encoding="utf-8"))
        observations_name = str(artifacts.get("analysis_input", observations_name))
    relative_observations = Path(observations_name)
    if relative_observations.is_absolute() or ".." in relative_observations.parts:
        raise ValueError("analysis input must be a relative path within the run")
    observations_path = (root / relative_observations).resolve()
    if not observations_path.is_relative_to(root.resolve()):
        raise ValueError("analysis input must remain within the run directory")
    unblinded_path = root / "unblinded_observations.jsonl"
    if observations_path.exists():
        mapping = key["blinded_to_condition"]
        observations = read_jsonl(observations_path)
        for row in observations:
            row["unblinded_condition"] = mapping.get(
                str(row.get("condition_id", "")), "unknown"
            )
        write_jsonl(unblinded_path, observations)
        record["observations_file"] = unblinded_path.name
        record["source_observations_file"] = observations_name
    (root / "unblinded_mapping.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return record
