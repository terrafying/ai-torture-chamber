"""Read-only broker relay and owner-only sealed dataset exports."""
import hashlib
import importlib
import io
import json
import zipfile

from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
import pytest

from observatory.app import create_app
from observatory.curation import build_snapshot
from observatory.store import Store

app_module = importlib.import_module("observatory.app")
NETWORK = "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp"
ASSET = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSERVATORY_SECRET_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("OBSERVATORY_ADMIN_TOKEN", "owner-test-token")
    store = Store(tmp_path / "state.sqlite3")
    yield store
    store.close()


def mock_broker(monkeypatch, catalog=None, funding=None, failure=False):
    class Broker:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get_catalog(self):
            if failure:
                raise RuntimeError("secret bearer and https://private-control.example")
            return catalog or {"models": []}

        async def get_funding(self):
            if failure:
                raise RuntimeError("secret bearer and https://private-control.example")
            return funding or {"status": "unknown"}
    monkeypatch.setattr(app_module, "_broker_client", Broker)


def model(identifier, **updates):
    return {"id": identifier, "name": "Research model", "categories": ["research"],
            "supportedprotocols": ["responses"], "capabilities": {"vision": True, "structured_actions": True, "structured_outputs": True, "tool_calling": True}, "capability_source": "gateway_advertised", "eligible": True, **updates}


def test_public_catalog_filters_capabilities_protocols_and_private_fields(db, monkeypatch):
    mock_broker(monkeypatch, catalog={"network": NETWORK, "asset": ASSET,
        "api_key": "must-never-be-public", "gateway_url": "https://private-control.example",
        "models": [model("openai/verified", pricing={"token": "private"}, endpoint="https://control.example"),
                   model("text-only", capabilities={"vision": False, "structured_actions": True}),
                   model("no-structure", capabilities={"vision": True}),
                   model("prefix-only", capability_source="unverified", eligible=False),
                   model("unknown-protocol", supportedprotocols=["chat"]),
                   model("https://private.example/model"),
                   model("anthropic/verified", protocols=["messages"], name="https://control.example/Bearer private")]})
    with TestClient(create_app(db, enable_runtime=False)) as client:
        response = client.get("/api/research-models")
    result = response.json()
    assert response.status_code == 200
    assert [item["id"] for item in result["models"]] == ["openai/verified", "anthropic/verified"]
    assert result["models"][1]["protocols"] == ["messages"]
    assert result["models"][1]["name"] == "anthropic/verified"
    assert result["models"][0]["capability_source"] == "gateway_advertised"
    assert result["models"][0]["compatibility_tested"] is False
    assert "https://" not in response.text and "must-never-be-public" not in response.text
    assert "pricing" not in response.text and "endpoint" not in response.text


def test_funding_relay_has_only_safe_amounts_addresses_and_receipts(db, monkeypatch):
    mock_broker(monkeypatch, funding={"status": "ready", "configured": True, "enabled": True,
        "network": NETWORK, "asset": ASSET, "wallet_address": "1" * 32,
        "balance_atomic": "120000000", "reserved_atomic": "2000000", "settled_atomic": "3000000",
        "daily_spent_and_reserved_atomic": "5000000", "daily_remaining_atomic": "95000000", "required_request_reserve_atomic": "3000000",
        "reason": "DAILY_LIMIT",
        "signing_key": "must-never-be-public", "url": "https://control.example",
        "receipts": [{"request_id": "request_1", "status": "settled", "amount_atomic": "3000000",
                      "transaction": "5" * 88, "network": NETWORK, "asset": ASSET,
                      "prompt": "private user prompt", "response": "private model response", "token": "private"},
                     {"request_id": "https://private.example", "status": "Bearer private", "amount_atomic": 1.5,
                      "transaction": "https://private.example"}]})
    with TestClient(create_app(db, enable_runtime=False)) as client:
        response = client.get("/api/funding")
    result = response.json()
    assert result["wallet_address"] == "1" * 32
    assert result["balance_atomic"] == "120000000"
    assert result["daily_remaining_atomic"] == "95000000" and result["reason"] == "DAILY_LIMIT"
    assert result["receipts"][1]["request_id"] is None
    assert result["receipts"][1]["transaction"] is None
    assert result["receipts"][1]["amount_atomic"] is None
    assert result["hf_jobs"]["status"] == "separate"
    for private in ("must-never-be-public", "https://", "prompt", "response", "Bearer"):
        assert private not in response.text


def test_disconnected_broker_has_no_fake_wallet_balance_or_models(db, monkeypatch):
    mock_broker(monkeypatch, failure=True)
    with TestClient(create_app(db, enable_runtime=False)) as client:
        funding = client.get("/api/funding").json()
        catalog = client.get("/api/research-models").json()
    assert funding["status"] == "unknown" and funding["configured"] is False
    assert funding["wallet_address"] is None and funding["balance_atomic"] is None
    assert funding["receipts"] == [] and catalog["models"] == []
    assert "secret" not in json.dumps(funding) + json.dumps(catalog)


def add_source(db, identifier, **updates):
    return db.put("sources", {"id": identifier, "canonical_url": "https://example.org/paper/" + identifier,
        "license": "CC-BY-4.0", "license_verified": True,
        "rights_evidence": "https://creativecommons.org/licenses/by/4.0/",
        "provenance": {"title": "Primary source " + identifier, "authors": ["Researcher"], "version": "v1"},
        "text": ("Evidence and uncertainty for " + identifier + ". ") * 100,
        "quality_review": {"status": "approved", "reviewed_by": "fixture-owner", "rationale": "Source fixture reviewed",
                           "topic_relevance": "relevant", "evidence_stance": "uncertain", "source_type": "empirical_paper"},
        "extraction_review_status": "approved", "extraction_review_evidence": "Fixture source text verified",
        **updates})


def test_export_requires_header_auth_and_contains_sealed_splits_and_hashes(db):
    db.save_settings({"hf_token": "hf-export-not-allowed"})
    add_source(db, "train")
    add_source(db, "holdout", held_out=True)
    add_source(db, "excluded", license="unknown")
    db.put("notes", {"id": "unapproved-note", "source_id": "train", "question": "Question", "answer": "Unapproved synthetic answer"})
    snapshot = build_snapshot(db)
    with TestClient(create_app(db, enable_runtime=False)) as client:
        route = "/api/admin/datasets/" + snapshot["id"] + "/export"
        assert client.get(route).status_code == 401
        assert client.get(route + "?token=owner-test-token").status_code == 401
        assert client.get(route, headers={"Authorization": "Bearer wrong"}).status_code == 401
        response = client.get(route, headers={"Authorization": "Bearer owner-test-token"})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {"cpt/train.jsonl", "cpt/validation.jsonl", "sft/train.jsonl", "sft/validation.jsonl",
                                            "manifest.json", "rights-provenance.json", "exclusions.json", "snapshot.json", "hashes.json"}
        hashes = json.loads(archive.read("hashes.json"))
        for name, digest in hashes.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
            assert b"hf-export-not-allowed" not in archive.read(name)
            assert b"owner-test-token" not in archive.read(name)
        assert json.loads(archive.read("manifest.json")) == snapshot["manifest"]
        heldout = [json.loads(line) for line in archive.read("cpt/validation.jsonl").splitlines()]
        assert any("holdout" in item["source_ids"] for item in heldout)
        train = [json.loads(line) for line in archive.read("cpt/train.jsonl").splitlines()]
        assert not any("holdout" in item["source_ids"] for item in train)
        assert archive.read("sft/train.jsonl") == b"" and archive.read("sft/validation.jsonl") == b""
        assert "excluded" in archive.read("exclusions.json").decode()
        assert "Unapproved synthetic answer" not in archive.read("manifest.json").decode()
        assert json.loads(archive.read("rights-provenance.json"))[0]["license"] == "CC-BY-4.0"


def test_export_refuses_tampered_snapshot_and_missing_id(db):
    add_source(db, "source")
    snapshot = build_snapshot(db)
    snapshot["original_text"][0]["text"] = "tampered"
    snapshot["id"] = "corrupted-copy"
    db.put("datasets", snapshot)
    with TestClient(create_app(db, enable_runtime=False)) as client:
        headers = {"Authorization": "Bearer owner-test-token"}
        assert client.get("/api/admin/datasets/missing/export", headers=headers).status_code == 404
        response = client.get("/api/admin/datasets/" + snapshot["id"] + "/export", headers=headers)
    assert response.status_code == 409
    assert "integrity" in response.json()["detail"]


def test_export_includes_only_approved_supported_sft_in_heldout_split(db):
    document = add_source(db, "sft-source", held_out=True)
    db.save_settings({"synthetic_training_approved": True, "provider_policy_reference": "Owner-reviewed teacher terms, 2026-10-05"})
    db.put("evidence", {"id": "evidence-1", "source_id": document["id"], "support_verified": True, "passage": document["text"][:60]})
    db.put("notes", {"id": "approved-note", "source_id": document["id"], "question": "What remains uncertain?", "answer": "The evidence is subject to alternative interpretations.",
                     "review_status": "approved", "support_verified": True, "evidence_ids": ["evidence-1"], "generated_by": "fixture-teacher"})
    snapshot = build_snapshot(db)
    with TestClient(create_app(db, enable_runtime=False)) as client:
        response = client.get("/api/admin/datasets/" + snapshot["id"] + "/export", headers={"Authorization": "Bearer owner-test-token"})
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        assert archive.read("sft/train.jsonl") == b""
        records = [json.loads(line) for line in archive.read("sft/validation.jsonl").splitlines()]
        assert len(records) == 1 and records[0]["synthetic"] is True
        assert records[0]["source_ids"] == [document["id"]]
        assert records[0]["evidence_ids"] == ["evidence-1"]
        assert records[0]["split"] == "validation"
        assert records[0]["messages"][1]["role"] == "assistant"


def test_health_reports_worker_task_liveness(db):
    class Task:
        def __init__(self, done):
            self.finished = done
        def done(self):
            return self.finished
    class Worker:
        def __init__(self, task):
            self.task = task
        async def close(self):
            pass
    with TestClient(create_app(db, enable_runtime=False)) as client:
        client.app.state.services = {"research": Worker(Task(False)), "training": Worker(Task(True))}
        response = client.get("/api/health").json()
    assert response["workers"]["research"]["running"] is True
    assert response["workers"]["training"]["running"] is False
    assert response["workers"]["training"]["state"] == "stopped"
