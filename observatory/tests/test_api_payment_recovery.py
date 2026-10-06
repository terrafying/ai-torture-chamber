"""Owner recovery inspects cache and seals unpaid IDs; it cannot initiate payment."""
import importlib
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
from uuid import uuid4

from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
import httpx
import pytest

from observatory.app import create_app
from observatory.store import Store

app_module = importlib.import_module("observatory.app")
AUTH = {"Authorization": "Bearer owner-recovery-token"}
BASE = "/api/admin/research-payments"
NETWORK = "solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp"
ASSET = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"


@pytest.fixture
def pending(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSERVATORY_SECRET_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("OBSERVATORY_ADMIN_TOKEN", "owner-recovery-token")
    db = Store(tmp_path / "state.sqlite3")
    db.set_mission({"id": "mission-review", "status": "faulted"})
    request_id = db.reserve_research_request("mission-review-scholar", "1" * 64, "responses",
        {"model": "openai/fixture", "input": "private request passage", "max_output_tokens": 40})
    yield db, request_id
    db.close()


def inspection(request_id, state="completed", *, in_flight=False, response=None):
    settled = state in {"completed", "settled_error", "reconciled"}
    status = "settled" if settled else state
    return {"found": state != "not_found", "request_id": request_id, "state": state, "status": status,
            "in_flight": in_flight, "response": response,
            "payment": None if state == "not_found" else {"request_id": request_id, "status": status,
                "network": NETWORK, "asset": ASSET, "amount_atomic": "0" if state in {"intent", "cancelled"} else "100000",
                "transaction": "5" * 88 if settled else None}}


def mock_inspection(monkeypatch, result, *, callback=None, failure=False, abandon_result=None):
    calls = []
    class Broker:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def get_request(self, request_id):
            calls.append(request_id)
            if callback:
                callback()
            if failure:
                raise RuntimeError("Bearer private-broker-token https://private-broker.example")
            return result
        async def abandon_request(self, request_id, protocol, body):
            calls.append(("abandon", request_id, protocol, body))
            return abandon_result if abandon_result is not None else {"abandoned": True, "request_id": request_id, "state": "cancelled", "in_flight": False}
    # Deliberately no request/sign/settle method: these routes cannot buy work.
    monkeypatch.setattr(app_module, "_broker_client", Broker)
    return calls


def test_recovery_routes_require_owner_header_and_pending_list_omits_body(pending, monkeypatch):
    db, request_id = pending
    calls = mock_inspection(monkeypatch, inspection(request_id))
    with TestClient(create_app(db, enable_runtime=False)) as client:
        for method, route, payload in (("GET", BASE, None), ("GET", BASE + "/" + request_id, None),
                                       ("POST", BASE + "/" + request_id + "/acknowledge", {"reviewed": True})):
            assert client.request(method, route, json=payload).status_code == 401
            assert client.request(method, route + "?token=owner-recovery-token", json=payload).status_code == 401
        response = client.get(BASE, headers=AUTH)
        public = client.get("/api/state")
    assert calls == []
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["pending"] == [{"request_id": request_id, "scope": "mission-review-scholar",
                                           "body_hash": "1" * 64, "protocol": "responses",
                                           "created_at": db.pending_research_requests()[0]["created_at"]}]
    assert "private request passage" not in response.text + public.text
    assert '"body"' not in response.text
    assert request_id not in public.text


def test_cached_response_is_owner_only_and_request_and_signing_data_are_removed(pending, monkeypatch):
    db, request_id = pending
    result = inspection(request_id, response={"output": [{"content": [{"type": "output_text", "text": "cached research output"}]}],
        "content": [{"type": "tool_use", "name": "save_note", "input": {"evidence_id": "evidence-1"}}],
        "input": "private request passage", "request": {"body": "private request passage"},
        "payment_signature": "must-not-relay", "signed_transaction": "must-not-relay",
        "signature": "must-not-relay", "api_key": "must-not-relay", "usage": {"output_tokens": 4}})
    result.update(body={"input": "private request passage"}, payment_signature="must-not-relay")
    result["payment"].update(prompt="private request passage", payment_signature="must-not-relay")
    calls = mock_inspection(monkeypatch, result)
    with TestClient(create_app(db, enable_runtime=False)) as client:
        assert client.get(BASE + "/" + request_id).status_code == 401
        response = client.get(BASE + "/" + request_id, headers=AUTH)
    assert calls == [request_id] and response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "cached research output" in response.text
    assert response.json()["response"]["usage"]["output_tokens"] == 4
    assert response.json()["response"]["content"][0]["input"] == {"evidence_id": "evidence-1"}
    assert "private request passage" not in response.text and "must-not-relay" not in response.text
    assert '"body"' not in response.text and '"signature"' not in response.text
    assert len(db.pending_research_requests()) == 1


@pytest.mark.parametrize("status", ["running", "pausing", "stopping", "funding_paused"])
def test_active_or_automatic_funding_pause_cannot_acknowledge(pending, monkeypatch, status):
    db, request_id = pending
    db.set_mission({"id": "mission-review", "status": status})
    calls = mock_inspection(monkeypatch, inspection(request_id))
    with TestClient(create_app(db, enable_runtime=False)) as client:
        response = client.post(BASE + "/" + request_id + "/acknowledge", headers=AUTH, json={"reviewed": True})
    assert response.status_code == 409 and calls == []
    assert len(db.pending_research_requests()) == 1


@pytest.mark.parametrize("state,in_flight", [("unknown", False), ("authorized", False), ("reserved", False),
                                            ("signing", False), ("completed", True), ("intent", True), ("not_found", True)])
def test_unresolved_or_in_flight_outcomes_retain_pending_intent(pending, monkeypatch, state, in_flight):
    db, request_id = pending
    calls = mock_inspection(monkeypatch, inspection(request_id, state, in_flight=in_flight))
    with TestClient(create_app(db, enable_runtime=False)) as client:
        response = client.post(BASE + "/" + request_id + "/acknowledge", headers=AUTH, json={"reviewed": True})
    assert response.status_code == 409 and calls == [request_id]
    assert len(db.pending_research_requests()) == 1


@pytest.mark.parametrize("state,status", [("completed", "paused"), ("settled_error", "faulted"),
                                         ("reconciled", "stopped"), ("intent", "paused"), ("not_found", "faulted"), ("cancelled", "stopped")])
def test_reviewed_settled_or_definitively_unpaid_intent_can_be_cleared_without_payment(pending, monkeypatch, state, status):
    db, request_id = pending
    db.set_mission({"id": "mission-review", "status": status})
    calls = mock_inspection(monkeypatch, inspection(request_id, state))
    with TestClient(create_app(db, enable_runtime=False)) as client:
        response = client.post(BASE + "/" + request_id + "/acknowledge", headers=AUTH, json={"reviewed": True})
    assert response.status_code == 200 and response.json()["acknowledged"] is True
    if state in {"intent", "not_found", "cancelled"}:
        assert calls == [request_id, ("abandon", request_id, "responses", {"model": "openai/fixture", "input": "private request passage", "max_output_tokens": 40})]
        final_state = "cancelled"
    else:
        assert calls == [request_id]
        final_state = state
    assert db.pending_research_requests() == []
    assert response.json()["state"] == final_state
    assert db.get_mission()["status"] == status
    event = db.events_after(0)[-1]
    assert event["type"] == "research.payment_reviewed"
    assert event["data"] == {"request_id": request_id, "state": final_state}
    assert "private request passage" not in json.dumps(event)


@pytest.mark.parametrize("payload", [{}, {"reviewed": False}, {"reviewed": 1}, {"reviewed": "true"}, {"reviewed": True, "force": True}])
def test_explicit_boolean_review_is_required(pending, monkeypatch, payload):
    db, request_id = pending
    calls = mock_inspection(monkeypatch, inspection(request_id))
    with TestClient(create_app(db, enable_runtime=False)) as client:
        response = client.post(BASE + "/" + request_id + "/acknowledge", headers=AUTH, json=payload)
    assert response.status_code == 422 and calls == []
    assert len(db.pending_research_requests()) == 1


def test_mission_resume_during_inspection_retains_intent(pending, monkeypatch):
    db, request_id = pending
    mock_inspection(monkeypatch, inspection(request_id), callback=lambda: db.set_mission({"id": "mission-review", "status": "running"}))
    with TestClient(create_app(db, enable_runtime=False)) as client:
        response = client.post(BASE + "/" + request_id + "/acknowledge", headers=AUTH, json={"reviewed": True})
    assert response.status_code == 409 and len(db.pending_research_requests()) == 1


def test_broker_failure_is_safe_and_cannot_clear_pending(pending, monkeypatch):
    db, request_id = pending
    mock_inspection(monkeypatch, {}, failure=True)
    with TestClient(create_app(db, enable_runtime=False)) as client:
        response = client.post(BASE + "/" + request_id + "/acknowledge", headers=AUTH, json={"reviewed": True})
    assert response.status_code == 503 and "private-broker" not in response.text
    assert len(db.pending_research_requests()) == 1


@pytest.mark.parametrize("mutation", ["missing_in_flight", "wrong_request", "wrong_receipt", "nonzero_idle_intent"])
def test_invalid_or_ambiguous_broker_inspection_fails_closed(pending, monkeypatch, mutation):
    db, request_id = pending
    result = inspection(request_id, "intent" if mutation == "nonzero_idle_intent" else "completed")
    if mutation == "missing_in_flight":
        result.pop("in_flight")
    elif mutation == "wrong_request":
        result["request_id"] = str(uuid4())
    elif mutation == "wrong_receipt":
        result["payment"]["request_id"] = str(uuid4())
    else:
        result["payment"]["amount_atomic"] = "1"
    mock_inspection(monkeypatch, result)
    with TestClient(create_app(db, enable_runtime=False)) as client:
        response = client.post(BASE + "/" + request_id + "/acknowledge", headers=AUTH, json={"reviewed": True})
    assert response.status_code in {409, 502} and len(db.pending_research_requests()) == 1


def test_invalid_uuid_and_missing_pending_never_contact_broker(pending, monkeypatch):
    db, request_id = pending
    calls = mock_inspection(monkeypatch, inspection(request_id))
    with TestClient(create_app(db, enable_runtime=False)) as client:
        assert client.get(BASE + "/invalid-uuid", headers=AUTH).status_code == 422
        response = client.post(BASE + "/" + str(uuid4()) + "/acknowledge", headers=AUTH, json={"reviewed": True})
    assert response.status_code == 404 and calls == []


@pytest.mark.parametrize("abandon_result", [{"abandoned": False, "in_flight": False, "state": "cancelled"},
    {"abandoned": True, "in_flight": True, "state": "cancelled"},
    {"abandoned": True, "in_flight": False, "state": "intent"},
    {"abandoned": True, "in_flight": False, "state": "cancelled", "request_id": "wrong"}])
def test_unpaid_acknowledgement_requires_exact_idle_tombstone(pending, monkeypatch, abandon_result):
    db, request_id = pending
    result = {"request_id": request_id, **abandon_result}
    calls = mock_inspection(monkeypatch, inspection(request_id, "not_found"), abandon_result=result)
    with TestClient(create_app(db, enable_runtime=False)) as client:
        response = client.post(BASE + "/" + request_id + "/acknowledge", headers=AUTH, json={"reviewed": True})
    assert response.status_code == 409 and len(db.pending_research_requests()) == 1
    assert len(calls) == 2 and calls[1][0] == "abandon"


@pytest.mark.parametrize("settled", [True, False])
def test_real_internal_broker_contract_recovers_cache_or_seals_unpaid_uuid_offline(pending, monkeypatch, tmp_path, settled):
    """Exercise both ASGI apps and the actual internal client without networking."""
    from observatory.x402_broker import BrokerConfig, BrokerError, PaymentBroker, canonical, create_app as create_broker_app
    from observatory.x402_client import X402BrokerClient

    db, request_id = pending
    stored = db.pending_research_requests()[0]
    original = {"request_id": request_id, "protocol": stored["protocol"], "body": stored["body"]}
    monkeypatch.setenv("OBSERVATORY_PAYMENT_BROKER_TOKEN", "internal-recovery-token")
    monkeypatch.setenv("OBSERVATORY_PAYMENT_BROKER_URL", "http://127.0.0.1:8062")

    class NoExternalOperations:
        async def close(self):
            pass
        def __getattr__(self, name):
            raise AssertionError("Recovery invoked forbidden external operation: " + name)

    broker = PaymentBroker(BrokerConfig(db_path=str(tmp_path / "payments.sqlite3"), token="internal-recovery-token",
                                        encryption_key=Fernet.generate_key().decode()), transport=NoExternalOperations())
    if settled:
        broker.ledger.intent(request_id, hashlib.sha256(canonical(original)).hexdigest(), stored["protocol"], stored["body"]["model"])
        broker.ledger.update(request_id, status="completed", amount=100000, tx="5" * 88,
                             response={"output": [{"content": [{"type": "output_text", "text": "actual broker cached output"}]}]})
    internal_app = create_broker_app(broker)

    @asynccontextmanager
    async def client_factory():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=internal_app), base_url="http://127.0.0.1:8062") as internal_http:
            async with X402BrokerClient(client=internal_http) as client:
                yield client
    monkeypatch.setattr(app_module, "_broker_client", client_factory)

    try:
        with TestClient(create_app(db, enable_runtime=False)) as client:
            view = client.get(BASE + "/" + request_id, headers=AUTH)
            assert view.status_code == 200 and view.json()["in_flight"] is False
            if settled:
                assert "actual broker cached output" in view.text
            else:
                assert view.json()["state"] == "not_found"
            acknowledged = client.post(BASE + "/" + request_id + "/acknowledge", headers=AUTH, json={"reviewed": True})
            assert acknowledged.status_code == 200
        assert db.pending_research_requests() == []
        assert broker.ledger.get(request_id)["status"] == ("completed" if settled else "cancelled")
        if not settled:
            with pytest.raises(BrokerError) as error:
                asyncio.run(broker.request(original))
            assert error.value.code == "REQUEST_ABANDONED"
    finally:
        asyncio.run(broker.close())
