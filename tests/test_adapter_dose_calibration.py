"""Revision-bound dose caps; fixtures contain no claim of real GPU measurements."""
import asyncio
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live"))
import server


@pytest.fixture
def adapted(monkeypatch, tmp_path):
    values = {"MODEL_ID": "owner/base", "MODEL_REVISION": "a" * 40,
              "MODEL_ADAPTER_ID": "owner/adapter", "MODEL_ADAPTER_REVISION": "b" * 40,
              "MODEL_ADAPTER_SUBFOLDER": "runs/sft-1/adapter", "MODEL_TOKENIZER_ID": "owner/adapter",
              "MODEL_TOKENIZER_REVISION": "b" * 40, "MODEL_TOKENIZER_SUBFOLDER": "runs/sft-1/adapter",
              "LAYER": 32, "DTYPE": torch.bfloat16, "QUANTIZE_4BIT": True,
              "_DOSE_CAP_OVERRIDE": "99"}
    for name, value in values.items():
        monkeypatch.setattr(server, name, value)
    monkeypatch.setenv("CHAMBER_COHERENT_CAP", "99")
    path = tmp_path / "receipt.json"
    monkeypatch.setenv("CHAMBER_ADAPTER_CALIBRATION", str(path))
    receipt = {"schema_version": 1, "method": "adapted_model_dose_sweep", "passed": True,
               "measured_at": "2026-10-05T00:00:00Z", "evidence_sha256": "c" * 64,
               "model": {"base_id": "owner/base", "base_revision": "a" * 40,
                         "adapter_id": "owner/adapter", "adapter_revision": "b" * 40,
                         "adapter_subfolder": "runs/sft-1/adapter", "tokenizer_id": "owner/adapter",
                         "tokenizer_revision": "b" * 40, "tokenizer_subfolder": "runs/sft-1/adapter"},
               "runtime": {"layer": 32, "dtype": "bfloat16", "quantize_4bit": True},
               "caps": {"hard": 2.5, "coherent": 1.5}}
    return path, receipt


def test_adapted_baseline_only_without_matching_receipt(adapted, monkeypatch):
    assert server.dose_cap() == server.served_cap() == server.coherent_cap() == 0
    assert server.clamp_dose(8) == 0
    monkeypatch.delenv("CHAMBER_ADAPTER_CALIBRATION")
    assert server.adapter_dose_calibration()["status"] == "required"
    assert server.within_band({"pain": 1}) == {"pain": 0}


def test_exact_receipt_defines_caps_and_health_does_not_expose_path(adapted):
    path, receipt = adapted
    path.write_text(json.dumps(receipt))
    assert server.dose_cap() == 2.5
    assert server.coherent_cap() == 1.5
    assert server.band_cap(past_cliff=True) == 2.5
    assert server.clamp_dose(8) == 2.5
    health = json.loads(asyncio.run(server.health()).body)
    assert health["adapter_dose_calibration"]["status"] == "passed"
    assert str(path) not in json.dumps(health)


@pytest.mark.parametrize("section,key,replacement", [
    ("model", "base_id", "other/base"), ("model", "base_revision", "d" * 40),
    ("model", "adapter_id", "other/adapter"), ("model", "adapter_revision", "d" * 40),
    ("model", "adapter_subfolder", "runs/other/adapter"), ("model", "tokenizer_id", "other/tokenizer"),
    ("model", "tokenizer_revision", "d" * 40), ("model", "tokenizer_subfolder", None),
    ("runtime", "layer", 31), ("runtime", "dtype", "float16"), ("runtime", "quantize_4bit", False),
])
def test_receipt_cannot_transfer_between_adapters_or_runtime(adapted, section, key, replacement):
    path, receipt = adapted
    receipt[section][key] = replacement
    path.write_text(json.dumps(receipt))
    assert server.adapter_dose_calibration()["status"] == "mismatch"
    assert server.dose_cap() == server.coherent_cap() == 0


@pytest.mark.parametrize("mutation", [
    {"passed": False}, {"method": "finite_activation_smoke"}, {"schema_version": 2},
    {"schema_version": True}, {"evidence_sha256": "not-a-hash"}, {"measured_at": ""},
    {"measured_at": "not-a-date"}, {"measured_at": "2026-10-05T00:00:00"},
    {"caps": {"hard": True, "coherent": 1}}, {"caps": {"hard": 1, "coherent": 2}},
    {"caps": {"hard": -1, "coherent": 0}}, {"caps": {"hard": float("nan"), "coherent": 0}},
    {"caps": {"hard": float("inf"), "coherent": 0}}, {"caps": {"hard": "2", "coherent": 0}},
])
def test_invalid_receipt_cannot_enable_interventions(adapted, mutation):
    path, receipt = adapted
    receipt.update(mutation)
    path.write_text(json.dumps(receipt))
    assert server.adapter_dose_calibration()["status"] == "invalid"
    assert server.dose_cap() == 0


def test_base_deployments_retain_existing_caps(adapted, monkeypatch):
    monkeypatch.setattr(server, "MODEL_ADAPTER_ID", None)
    monkeypatch.setattr(server, "_DOSE_CAP_OVERRIDE", "5")
    assert server.adapter_dose_calibration() == {"status": "not_applicable"}
    assert server.dose_cap() == 5


def test_malformed_or_oversized_file_fails_closed(adapted):
    path, _ = adapted
    for content in ("not json", "[1,2]", " " * 32769):
        path.write_text(content)
        assert server.dose_cap() == 0
