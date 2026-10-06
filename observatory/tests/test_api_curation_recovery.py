"""Owner-authorized lost-review recovery never calls a model or resumes work."""
from contextlib import contextmanager
from types import SimpleNamespace

from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
import pytest

from observatory.app import create_app
from observatory.store import Store

BASE = "/api/admin/curation/recovery"
AUTH = {"Authorization": "Bearer owner-curation-recovery-token"}
PRIVATE_QUOTE = "PRIVATE_REVIEW_SOURCE_QUOTATION"


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSERVATORY_SECRET_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("OBSERVATORY_ADMIN_TOKEN", "owner-curation-recovery-token")
    store = Store(tmp_path / "recovery.sqlite3")
    store.set_mission({"id": "mission-review", "status": "faulted"})
    store.put("curation_work", {"id": "automatic-curation", "status": "reviewing", "source_id": "paper", "review_id": "review-interrupted",
        "source_fingerprint": "f" * 64, "policy_hash": "a" * 64, "active_stage": "critic", "call_state": "awaiting_operator",
        "reviews": [{"stage": "primary", "verdict": {"decision": "accept", "quotes": [PRIVATE_QUOTE], "rationale": "Private completed verdict"}}]})
    yield store
    store.close()


@contextmanager
def client_with_idle_worker(db, *, busy=False):
    app = create_app(db, enable_runtime=False)
    async def close_stub():
        pass
    with TestClient(app) as client:
        # No factory, model, browser or provider client exists in this fixture.
        app.state.services["automatic_curation"] = SimpleNamespace(_lock=SimpleNamespace(locked=lambda: busy), close=close_stub)
        yield client


def payload():
    return {"reviewed": True, "review_id": "review-interrupted"}


def test_recovery_requires_owner_header_not_query_token(db):
    initial = db.get("curation_work", "automatic-curation")
    with client_with_idle_worker(db) as client:
        assert client.post(BASE, json=payload()).status_code == 401
        assert client.post(BASE + "?token=owner-curation-recovery-token", json=payload()).status_code == 401
        assert client.post(BASE, headers={"Authorization": "Bearer wrong-token"}, json=payload()).status_code == 401
    assert db.get("curation_work", "automatic-curation") == initial


@pytest.mark.parametrize("bad", [{}, {"reviewed": False, "review_id": "review-interrupted"},
    {"reviewed": 1, "review_id": "review-interrupted"}, {"reviewed": "true", "review_id": "review-interrupted"},
    {"reviewed": True}, {"reviewed": True, "review_id": 123}, {"reviewed": True, "review_id": None},
    {"reviewed": True, "review_id": "review-interrupted", "force": True}])
def test_recovery_requires_exact_inspection_payload(db, bad):
    initial = db.get("curation_work", "automatic-curation")
    with client_with_idle_worker(db) as client:
        response = client.post(BASE, headers=AUTH, json=bad)
    assert response.status_code == 422
    assert db.get("curation_work", "automatic-curation") == initial


@pytest.mark.parametrize("mission_status", ["paused", "faulted", "stopped"])
@pytest.mark.parametrize("call_state", ["in_flight", "awaiting_operator"])
def test_reviewed_idle_recovery_preserves_completed_verdicts_without_resuming_mission(db, mission_status, call_state):
    db.set_mission({"id": "mission-review", "status": mission_status})
    initial = db.get("curation_work", "automatic-curation")
    db.put("curation_work", {**initial, "call_state": call_state})
    with client_with_idle_worker(db) as client:
        response = client.post(BASE, headers=AUTH, json=payload())
    assert response.status_code == 200
    assert response.json() == {"acknowledged": True, "review_id": "review-interrupted", "resume_required": True}
    recovered = db.get("curation_work", "automatic-curation")
    assert recovered["call_state"] == "retry_authorized"
    for key in ("source_id", "review_id", "source_fingerprint", "policy_hash", "reviews", "active_stage", "status"):
        assert recovered[key] == initial[key]
    assert db.get_mission()["status"] == mission_status
    event = db.events_after(0)[-1]
    assert event["type"] == "curation.recovery_authorized"
    assert event["data"] == {"review_id": "review-interrupted", "source_id": "paper", "stage": "critic"}
    assert PRIVATE_QUOTE not in response.text + str(event)


@pytest.mark.parametrize("status", ["running", "pausing", "stopping", "funding_paused"])
def test_recovery_refuses_active_or_automatic_funding_pause(db, status):
    db.set_mission({"id": "mission-review", "status": status})
    initial = db.get("curation_work", "automatic-curation")
    with client_with_idle_worker(db) as client:
        response = client.post(BASE, headers=AUTH, json=payload())
    assert response.status_code == 409
    assert db.get("curation_work", "automatic-curation") == initial
    assert db.get_mission()["status"] == status


def test_busy_worker_cannot_authorize_another_stage(db):
    initial = db.get("curation_work", "automatic-curation")
    with client_with_idle_worker(db, busy=True) as client:
        assert client.post(BASE, headers=AUTH, json=payload()).status_code == 409
    assert db.get("curation_work", "automatic-curation") == initial


def test_pending_automatic_curation_payment_must_be_reconciled_first(db):
    request_id = db.reserve_research_request("automatic-curation", "a" * 64, "responses", {"input": "PRIVATE_PENDING_REVIEW", "model": "fixture"})
    initial = db.get("curation_work", "automatic-curation")
    with client_with_idle_worker(db) as client:
        response = client.post(BASE, headers=AUTH, json=payload())
    assert response.status_code == 409
    assert db.pending_research_requests("automatic-curation")[0]["request_id"] == request_id
    assert db.get("curation_work", "automatic-curation") == initial
    assert "PRIVATE_PENDING_REVIEW" not in response.text


def test_unrelated_agent_pending_payment_is_not_cleared_or_paid_by_review_recovery(db):
    request_id = db.reserve_research_request("mission-scholar", "b" * 64, "responses", {"input": "private unrelated request", "model": "fixture"})
    with client_with_idle_worker(db) as client:
        assert client.post(BASE, headers=AUTH, json=payload()).status_code == 200
    assert db.pending_research_requests("mission-scholar")[0]["request_id"] == request_id


def test_changed_review_id_cannot_recover_a_new_work_item(db):
    work = db.get("curation_work", "automatic-curation")
    db.put("curation_work", {**work, "review_id": "different-review"})
    with client_with_idle_worker(db) as client:
        assert client.post(BASE, headers=AUTH, json=payload()).status_code == 409
    assert db.get("curation_work", "automatic-curation")["call_state"] == "awaiting_operator"


@pytest.mark.parametrize("changes", [{"status": "completed"}, {"status": "idle"}, {"call_state": "completed"},
                                       {"call_state": "retry_authorized"}, {"call_state": "unpaid_retryable"}, {"call_state": "unknown"}])
def test_noninterrupted_or_completed_review_cannot_be_reset(db, changes):
    work = db.get("curation_work", "automatic-curation")
    initial = db.put("curation_work", {**work, **changes})
    with client_with_idle_worker(db) as client:
        assert client.post(BASE, headers=AUTH, json=payload()).status_code == 409
    assert db.get("curation_work", "automatic-curation") == initial


def test_public_curation_runtime_exposes_metadata_without_saved_verdicts_or_quotes(db):
    with client_with_idle_worker(db) as client:
        response = client.get("/api/state")
    runtime = response.json()["curation_runtime"]
    assert runtime == {"status": "reviewing", "call_state": "awaiting_operator", "active_stage": "critic", "review_id": "review-interrupted", "source_id": "paper"}
    assert PRIVATE_QUOTE not in response.text
    assert "Private completed verdict" not in response.text
    assert "source_fingerprint" not in runtime and "policy_hash" not in runtime and "reviews" not in runtime
