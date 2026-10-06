"""Offline payment-policy and official-SDK checks; no wallet funding or transfers."""
from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
import json
from types import SimpleNamespace
from uuid import uuid4

from cryptography.fernet import Fernet
import httpx
import pytest

from observatory.x402_broker import (
    ASSET, GATEWAY, MEMO_PROGRAM, NETWORK, TOKEN_PROGRAM, BrokerConfig,
    BrokerError, GatewayTransport, OfficialSvmSigner, PaymentBroker,
    PaymentLedger, SignedPayment, capability_registry, create_app, usdc_atomic,
)

WALLET = "So11111111111111111111111111111111111111112"
PAY_TO = "Vote111111111111111111111111111111111111111"
FEE_PAYER = "Stake11111111111111111111111111111111111111"
TX = "3" * 88
MODEL = "openai/test-model"


def payload(protocol="responses", request_id=None):
    return {"request_id": request_id or str(uuid4()), "protocol": protocol,
        "body": {"model": MODEL if protocol == "responses" else "anthropic/test-model",
                 "max_output_tokens" if protocol == "responses" else "max_tokens": 200,
                 "input" if protocol == "responses" else "messages": "PRIVATE_PROMPT_FIXTURE" if protocol == "responses" else [{"role": "user", "content": "PRIVATE_PROMPT_FIXTURE"}]}}


def challenge(protocol="responses", **changes):
    quote = {"scheme": "exact", "network": NETWORK, "asset": ASSET, "amount": "100000",
             "payTo": PAY_TO, "maxTimeoutSeconds": 300, "extra": {"feePayer": FEE_PAYER}}
    quote.update(changes)
    return {"x402Version": 2, "resource": {"url": GATEWAY + "/" + protocol}, "accepts": [quote]}


def encoded(value):
    return base64.b64encode(json.dumps(value).encode()).decode()


class FakeSigner:
    address = WALLET
    calls = 0
    fail = False
    cancelled = False

    async def sign(self, required, quote):
        self.calls += 1
        if self.cancelled:
            raise asyncio.CancelledError
        if self.fail:
            raise RuntimeError("PRIVATE_SIGNER_FAILURE_FIXTURE")
        return SignedPayment("PRIVATE_SIGNED_PAYLOAD_FIXTURE", "fixture-unique-memo")


class FakeTransport:
    def __init__(self):
        self.balance_value = 1_000_000
        self.catalog_fail = False
        self.catalog_count = 0
        self.balance_fail = False
        self.probe_fail = False
        self.paid_fail = False
        self.paid_cancel = False
        self.confirm = True
        self.paid_count = 0
        self.probe_count = 0
        self.quote_changes = {}
        self.required_changes = {}
        self.headers = {}
        self.response_status = 200
        self.vendor = {"output": [{"type": "message", "content": [{"text": "PRIVATE_VENDOR_RESPONSE_FIXTURE"}]}]}
        self.paid_raw = None
        self.closed = False

    async def close(self):
        self.closed = True

    async def catalog(self):
        self.catalog_count += 1
        if self.catalog_fail:
            raise RuntimeError("PRIVATE_CATALOG_FAILURE")
        return {"data": [{"id": MODEL, "name": "Untrusted provider name", "categories": ["chat", "vision", "reasoning"], "pricing": {"input": "10", "output": "20", "api_key": "secret"},
            "supportedprotocols": ["responses"], "capabilities": {"vision": True, "structured_actions": True, "structured_outputs": True}},
            {"id": "anthropic/test-model", "categories": ["vision", "chat"], "supportedprotocols": ["messages"],
             "capabilities": {"vision": True, "structured_actions": True, "tool_calling": True}},
            {"id": "google/test-model", "categories": ["vision", "chat"]}]}

    async def balance(self, address):
        assert address == WALLET
        if self.balance_fail:
            raise RuntimeError("PRIVATE_RPC_FAILURE")
        return self.balance_value

    async def post(self, protocol, body, payment_header=None, request_id=""):
        if not payment_header:
            self.probe_count += 1
            if self.probe_fail:
                raise httpx.ConnectError("PRIVATE_UPSTREAM_FAILURE")
            required = challenge(protocol, **self.quote_changes)
            required.update(self.required_changes)
            return httpx.Response(402, headers={"PAYMENT-REQUIRED": encoded(required)})
        assert payment_header == "PRIVATE_SIGNED_PAYLOAD_FIXTURE"
        self.paid_count += 1
        if self.paid_cancel:
            raise asyncio.CancelledError
        if self.paid_fail:
            raise httpx.ReadTimeout("PRIVATE_PAID_TIMEOUT")
        headers = {"PAYMENT-RESPONSE": encoded({"success": True, "transaction": TX, "network": NETWORK, "payer": FEE_PAYER}), **self.headers}
        return httpx.Response(self.response_status, headers=headers, content=self.paid_raw) if self.paid_raw else httpx.Response(self.response_status, headers=headers, json=self.vendor)

    async def verify_receipt(self, transaction, address, pay_to, amount, memo):
        assert (transaction, address, pay_to, amount, memo) == (TX, WALLET, PAY_TO, 100000, "fixture-unique-memo")
        return self.confirm


@pytest.fixture
def broker(tmp_path):
    config = BrokerConfig(db_path=str(tmp_path / "payments.sqlite3"), token="INTERNAL_TOKEN_FIXTURE",
        private_key="NON_WALLET_SIGNER_FIXTURE", encryption_key=Fernet.generate_key().decode(), enabled=True,
        max_request_atomic=200000, daily_limit_atomic=600000, min_reserve_atomic=25000,
        allowed_pay_to=frozenset({PAY_TO}))
    instance = PaymentBroker(config, transport=FakeTransport(), signer=FakeSigner())
    yield instance
    instance.ledger.close()


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-1", "0", "0.0000001", "100000000000000000000"])
def test_caps_are_exact_finite_decimal_micro_usdc(value):
    with pytest.raises(ValueError):
        usdc_atomic(value)
    assert usdc_atomic("0.100001") == 100001
    assert usdc_atomic("0", zero=True) == 0


def test_default_config_cannot_authorize_spending():
    config = BrokerConfig.from_env({})
    assert not config.enabled
    assert "explicit_spending_limits_missing" in config.spending_errors()
    assert config.private_key == ""


@pytest.mark.asyncio
async def test_funding_does_not_fake_balance_or_reveal_secrets(broker):
    broker.signer = None
    state = await broker.funding()
    assert state["status"] == "unknown"
    assert state["wallet_address"] is None
    assert state["balance_atomic"] is None
    assert "PRIVATE" not in json.dumps(state)
    assert state["reason"] == "SIGNER_NOT_CONFIGURED"


@pytest.mark.asyncio
async def test_metadata_catalog_only_exposes_approved_safe_shape(broker):
    catalog = await broker.catalog()
    selected = catalog["models"][0]
    assert selected["supportedprotocols"] == ["responses"]
    assert selected["capabilities"]["structured_outputs"] is True
    assert catalog["models"][1]["supportedprotocols"] == ["messages"]
    assert catalog["models"][1]["capabilities"]["tool_calling"] is True
    assert catalog["models"][2]["supportedprotocols"] == []
    assert "api_key" not in json.dumps(catalog)
    assert selected["availability"] == "advertised"
    assert selected["capability_source"] == "gateway_advertised"
    assert selected["eligible"] is True
    assert catalog["models"][2]["capability_source"] == "unverified"
    assert catalog["models"][2]["eligible"] is False


@pytest.mark.asyncio
async def test_names_and_categories_never_infer_native_model_compatibility(broker):
    async def raw_catalog():
        return {"data": [{"id": MODEL, "categories": ["vision", "chat"]},
                         {"id": "anthropic/test-model", "categories": ["vision", "chat"]}]}
    broker.transport.catalog = raw_catalog
    catalog = await broker.catalog()
    assert all(model["capability_source"] == "unverified" and model["supportedprotocols"] == [] and not model["eligible"] for model in catalog["models"])
    with pytest.raises(BrokerError) as error:
        await broker.request(payload())
    assert error.value.code == "UNSUPPORTED_MODEL"
    assert broker.transport.probe_count == 0
    assert broker.signer.calls == 0


@pytest.mark.asyncio
async def test_owner_registry_is_exact_explicit_and_never_a_test_certificate(broker):
    registry = {MODEL: {"protocols": ["responses"], "vision": True, "structured_actions": True, "structured_outputs": True}}
    broker.config = replace(broker.config, model_capabilities=capability_registry(json.dumps(registry)))
    catalog = await broker.catalog()
    model = catalog["models"][0]
    assert model["capability_source"] == "owner_declared"
    assert model["eligible"] is True
    assert model["capabilities"]["tool_calling"] is False
    assert catalog["models"][1]["capability_source"] == "gateway_advertised"
    assert BrokerConfig.from_env({"X402_RESEARCH_MODEL_CAPABILITIES": json.dumps(registry)}).model_capabilities == broker.config.model_capabilities


@pytest.mark.parametrize("entry", [{"protocols": ["chat"]}, {"protocols": ["responses"], "vision": "true"}, {"protocols": ["responses"], "endpoint": "https://unapproved.invalid"}, {"protocols": []}])
def test_capability_registry_rejects_invalid_or_implicit_assertions(entry):
    with pytest.raises(ValueError):
        capability_registry(json.dumps({MODEL: entry}))


@pytest.mark.asyncio
async def test_declared_protocol_alone_cannot_authorize_visual_research(broker):
    broker.config = replace(broker.config, model_capabilities={MODEL: {"protocols": ["responses"]}})
    with pytest.raises(BrokerError) as error:
        await broker.request(payload())
    assert error.value.code == "UNSUPPORTED_MODEL"
    assert broker.signer.calls == 0


@pytest.mark.asyncio
async def test_catalog_refreshes_after_finite_ttl_and_never_falls_back_to_expired_data(broker, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("observatory.x402_broker.monotonic", lambda: clock[0])
    first = await broker.catalog()
    clock[0] = 399.0
    assert await broker.catalog() is first
    assert broker.transport.catalog_count == 1
    clock[0] = 400.0
    assert await broker.catalog() is not first
    assert broker.transport.catalog_count == 2
    clock[0] = 700.0
    broker.transport.catalog_fail = True
    with pytest.raises(BrokerError) as error:
        await broker.catalog()
    assert error.value.code == "CATALOG_UNAVAILABLE"
    assert broker.transport.paid_count == 0


@pytest.mark.asyncio
async def test_asgi_auth_and_validation_never_echo_private_input(broker):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(broker)), base_url="http://broker") as client:
        health = (await client.get("/health")).json()
        assert health["status"] == "ok"
        response = await client.post("/v1/request", json=payload())
        assert response.status_code == 401
        bad = payload()
        bad["body"]["stream"] = True
        response = await client.post("/v1/request", json=bad, headers={"Authorization": "Bearer INTERNAL_TOKEN_FIXTURE"})
        assert response.status_code == 422
        assert "PRIVATE_PROMPT" not in response.text
        response = await client.post("/v1/request", content="PRIVATE_BAD_JSON", headers={"Authorization": "Bearer INTERNAL_TOKEN_FIXTURE"})
        assert response.status_code == 422
        assert "PRIVATE_BAD_JSON" not in response.text
    assert broker.transport.paid_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("state,has_response", [("completed", True), ("unknown", True), ("unknown", False)])
async def test_authenticated_receipt_lookup_is_local_read_only_and_recovers_encrypted_response(broker, state, has_response):
    request = payload()
    identifier = request["request_id"]
    broker.ledger.intent(identifier, "PRIVATE_BODY_HASH_FIXTURE", "responses", MODEL)
    broker.ledger.update(identifier, status=state, amount=100000, memo="PRIVATE_MEMO_FIXTURE",
                         response=broker.transport.vendor if has_response else None)
    before = broker.ledger.get(identifier)
    async def unexpected(*args, **kwargs):
        pytest.fail("Receipt lookup must never call signer, provider or RPC")
    broker.transport.post = broker.transport.catalog = broker.transport.balance = broker.transport.verify_receipt = unexpected
    broker.signer.sign = unexpected
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(broker)), base_url="http://broker") as client:
        response = await client.get("/v1/requests/" + identifier, headers={"Authorization": "Bearer INTERNAL_TOKEN_FIXTURE"})
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    result = response.json()
    assert result["found"] is True
    assert result["in_flight"] is False
    assert result["state"] == state
    assert result["status"] == ("settled" if state == "completed" else "unknown")
    assert result["response"] == (broker.transport.vendor if has_response else None)
    assert result["payment"] == broker.ledger.receipt(before)
    assert all(secret not in response.text for secret in ("PRIVATE_BODY_HASH", "PRIVATE_MEMO", "PRIVATE_SIGNED", "PRIVATE_PROMPT", "INTERNAL_TOKEN"))
    assert broker.ledger.get(identifier) == before
    assert broker.transport.paid_count == broker.transport.probe_count == broker.signer.calls == 0


@pytest.mark.asyncio
async def test_receipt_lookup_not_found_and_authentication_never_expose_private_cache(broker):
    request = payload()
    identifier = request["request_id"]
    broker.ledger.intent(identifier, "hash", "responses", MODEL)
    broker.ledger.update(identifier, status="completed", response=broker.transport.vendor)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(broker)), base_url="http://broker") as client:
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            response = await client.get("/v1/requests/" + identifier, headers=headers)
            assert response.status_code == 401
            assert "PRIVATE_VENDOR" not in response.text
        absent = str(uuid4())
        response = await client.get("/v1/requests/" + absent, headers={"Authorization": "Bearer INTERNAL_TOKEN_FIXTURE"})
        assert response.status_code == 200
        assert response.json() == {"found": False, "request_id": absent, "status": "not_found", "state": "not_found", "response": None, "payment": None, "in_flight": False}
        response = await client.get("/v1/requests/PRIVATE_INVALID_ID", headers={"Authorization": "Bearer INTERNAL_TOKEN_FIXTURE"})
        assert response.status_code == 422
        assert "PRIVATE_INVALID_ID" not in response.text
    assert broker.transport.paid_count == broker.transport.probe_count == broker.signer.calls == 0


@pytest.mark.asyncio
async def test_inspection_tracks_every_queued_request_until_finally(broker):
    request = payload()
    await broker.lock.acquire()
    first, second = (asyncio.create_task(broker.request(request)) for _ in range(2))
    try:
        await asyncio.sleep(0)
        state = broker.inspect_request(request["request_id"])
        assert state["found"] is False and state["in_flight"] is True
        with pytest.raises(BrokerError) as active:
            await broker.abandon_request(request["request_id"], {"protocol": request["protocol"], "body": request["body"]})
        assert active.value.code == "REQUEST_IN_FLIGHT"
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert broker.inspect_request(request["request_id"])["in_flight"] is True
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        assert broker.inspect_request(request["request_id"])["in_flight"] is False
        assert broker.ledger.get(request["request_id"]) is None
        assert broker.transport.probe_count == broker.transport.paid_count == broker.signer.calls == 0
    finally:
        broker.lock.release()
        for task in (first, second):
            if not task.done():
                task.cancel()
        await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.asyncio
async def test_unpaid_probe_is_active_even_with_idle_looking_intent(broker):
    request = payload()
    entered = asyncio.Event()
    async def blocked_probe(*args, **kwargs):
        entered.set()
        await asyncio.Event().wait()
    broker.transport.post = blocked_probe
    task = asyncio.create_task(broker.request(request))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        state = broker.inspect_request(request["request_id"])
        assert state["found"] is True and state["state"] == "intent" and state["in_flight"] is True
        with pytest.raises(BrokerError) as active:
            await broker.abandon_request(request["request_id"], {"protocol": request["protocol"], "body": request["body"]})
        assert active.value.code == "REQUEST_IN_FLIGHT"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert broker.inspect_request(request["request_id"])["in_flight"] is False
    assert (await broker.abandon_request(request["request_id"], {"protocol": request["protocol"], "body": request["body"]}))["abandoned"] is True
    assert broker.signer.calls == broker.transport.paid_count == 0


@pytest.mark.asyncio
async def test_abandonment_durably_blocks_late_post_and_is_idempotent_without_network(broker):
    request = payload()
    identifier = request["request_id"]
    original = {"protocol": request["protocol"], "body": request["body"]}
    async def unexpected(*args, **kwargs):
        pytest.fail("Abandonment and a tombstoned late POST must never call signer/provider/RPC")
    broker.transport.post = broker.transport.catalog = broker.transport.balance = broker.transport.verify_receipt = unexpected
    broker.signer.sign = unexpected
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(broker)), base_url="http://broker") as client:
        headers = {"Authorization": "Bearer INTERNAL_TOKEN_FIXTURE"}
        response = await client.post("/v1/requests/" + identifier + "/abandon", headers=headers, json=original)
        assert response.status_code == 200
        expected = {"abandoned": True, "request_id": identifier, "state": "cancelled", "in_flight": False}
        assert response.json() == expected
        assert response.headers["Cache-Control"] == "no-store"
        before = broker.ledger.get(identifier)
        assert (await client.post("/v1/requests/" + identifier + "/abandon", headers=headers, json=original)).json() == expected
        assert broker.ledger.get(identifier) == before
        late = await client.post("/v1/request", headers=headers, json=request)
        assert late.status_code == 409 and late.json()["error"]["code"] == "REQUEST_ABANDONED"
        assert "PRIVATE_PROMPT" not in late.text
        inspected = (await client.get("/v1/requests/" + identifier, headers=headers)).json()
        assert inspected["state"] == inspected["status"] == "cancelled"
        assert inspected["in_flight"] is False and inspected["payment"]["amount_atomic"] == "0"
    broker.ledger.close()
    broker.ledger = PaymentLedger(broker.config.db_path, broker.config.encryption_key)
    with pytest.raises(BrokerError) as late_after_restart:
        await broker.request(request)
    assert late_after_restart.value.code == "REQUEST_ABANDONED"
    assert broker.ledger.totals()["reserved"] == broker.ledger.totals()["settled"] == 0
    assert broker.transport.probe_count == broker.transport.paid_count == broker.signer.calls == 0


@pytest.mark.asyncio
async def test_abandon_auth_exact_body_binding_and_validation(broker):
    request = payload()
    identifier = request["request_id"]
    original = {"protocol": request["protocol"], "body": request["body"]}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(broker)), base_url="http://broker") as client:
        url = "/v1/requests/" + identifier + "/abandon"
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            response = await client.post(url, headers=headers, json=original)
            assert response.status_code == 401 and "PRIVATE_PROMPT" not in response.text
        assert broker.ledger.get(identifier) is None
        headers = {"Authorization": "Bearer INTERNAL_TOKEN_FIXTURE"}
        invalid = await client.post(url, headers=headers, json={**original, "extra": "PRIVATE_EXTRA"})
        assert invalid.status_code == 422 and "PRIVATE_EXTRA" not in invalid.text
        assert (await client.post(url, headers=headers, json=original)).status_code == 200
        changed = {**original, "body": {**original["body"], "input": "PRIVATE_CHANGED_BODY"}}
        conflict = await client.post(url, headers=headers, json=changed)
        assert conflict.status_code == 409 and conflict.json()["error"]["code"] == "REQUEST_CONFLICT"
        assert "PRIVATE_CHANGED_BODY" not in conflict.text
        assert broker.ledger.get(identifier)["status"] == "cancelled"
    assert broker.transport.probe_count == broker.transport.paid_count == broker.signer.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("protected", ["reserved", "signing", "authorized", "unknown", "completed", "settled_error", "reconciled"])
async def test_abandon_never_clears_reserved_authorized_uncertain_or_settled_payment(broker, protected):
    request = payload()
    broker.transport.probe_fail = True
    with pytest.raises(BrokerError):
        await broker.request(request)
    broker.ledger.update(request["request_id"], status=protected, amount=100000)
    before = broker.ledger.get(request["request_id"])
    with pytest.raises(BrokerError) as denied:
        await broker.abandon_request(request["request_id"], {"protocol": request["protocol"], "body": request["body"]})
    assert denied.value.code == "REQUEST_NOT_ABANDONABLE"
    assert broker.ledger.get(request["request_id"]) == before
    assert broker.transport.probe_count == 1 and broker.transport.paid_count == broker.signer.calls == 0


@pytest.mark.parametrize("change", [{"stream": []}, {"stream": True}, {"max_output_tokens": None}, {"max_output_tokens": True}, {"max_output_tokens": 0}, {"model": None}, {"api_key": "secret"}])
def test_invalid_requests_cannot_get_to_signer(change):
    request = payload()
    request["body"].update(change)
    with pytest.raises(BrokerError, match=".*") as error:
        PaymentBroker.validate_request(request)
    assert error.value.code == "INVALID_REQUEST"


@pytest.mark.parametrize("protocol,content", [([], "ok"), ("responses", []), ("responses", None), ("messages", "incorrect text-only shape"), ("messages", ["incorrect item"])])
def test_native_vendor_input_is_validated_before_challenge(protocol, content):
    request = payload()
    request["protocol"] = protocol
    request["body"]["input" if protocol != "messages" else "messages"] = content
    with pytest.raises(BrokerError) as error:
        PaymentBroker.validate_request(request)
    assert error.value.code == "INVALID_REQUEST"


@pytest.mark.asyncio
async def test_paid_result_is_memo_verified_encrypted_and_idempotent_across_restart(broker):
    request = payload()
    first = await broker.request(request)
    again = await broker.request(request)
    assert first == again
    assert broker.transport.paid_count == 1
    assert broker.signer.calls == 1
    assert first["payment"]["status"] == "settled"
    assert first["payment"]["amount_atomic"] == "100000"
    row = broker.ledger.get(request["request_id"])
    assert b"PRIVATE_VENDOR_RESPONSE" not in row["response"]
    assert "PRIVATE_PROMPT" not in str(row)
    assert "PRIVATE_SIGNED_PAYLOAD" not in str(row)
    config = broker.config
    broker.ledger.close()
    broker.ledger = PaymentLedger(config.db_path, config.encryption_key)
    assert await broker.request(request) == first
    assert broker.transport.paid_count == 1
    with pytest.raises(BrokerError) as conflict:
        changed = {**request, "body": {**request["body"], "input": "changed"}}
        await broker.request(changed)
    assert conflict.value.code == "REQUEST_CONFLICT"
    assert "PRIVATE_VENDOR_RESPONSE" not in json.dumps(await broker.funding())


@pytest.mark.asyncio
async def test_native_messages_can_be_paid_without_protocol_translation(broker):
    result = await broker.request(payload("messages"))
    assert result["payment"]["protocol"] == "messages"


@pytest.mark.parametrize("changes", [{"network": "eip155:8453"}, {"asset": "different-mint"}, {"payTo": WALLET}, {"scheme": "upto"}, {"amount": "NaN"}, {"amount": "-100"}, {"maxTimeoutSeconds": 9999}, {"extra": {"feePayer": WALLET}}, {"extra": {"feePayer": FEE_PAYER, "paymentFlow": "upfront"}}])
@pytest.mark.asyncio
async def test_unapproved_quotes_never_create_payment(broker, changes):
    broker.transport.quote_changes = changes
    with pytest.raises(BrokerError) as error:
        await broker.request(payload())
    assert error.value.code == "UNAPPROVED_QUOTE"
    assert broker.transport.paid_count == broker.signer.calls == 0
    assert broker.ledger.totals()["reserved"] == 0


@pytest.mark.parametrize("memo", ["merchant-static-memo", "a" * 32, False, 42])
@pytest.mark.asyncio
async def test_seller_memo_is_rejected_before_reserving_signing_or_paid_post(broker, memo):
    broker.transport.quote_changes = {"extra": {"feePayer": FEE_PAYER, "memo": memo}}
    request = payload()
    with pytest.raises(BrokerError) as denied:
        await broker.request(request)
    assert denied.value.code == "UNAPPROVED_QUOTE"
    assert denied.value.retryable is False
    assert broker.ledger.get(request["request_id"])["status"] == "intent"
    assert broker.ledger.totals()["reserved"] == broker.ledger.totals()["settled"] == 0
    assert broker.transport.paid_count == broker.signer.calls == 0


@pytest.mark.parametrize("memo", [None, ""])
@pytest.mark.asyncio
async def test_empty_seller_memo_preserves_sdk_random_nonce_quote_path(broker, memo):
    broker.transport.quote_changes = {"extra": {"feePayer": FEE_PAYER, "memo": memo}}
    assert (await broker.request(payload()))["payment"]["status"] == "settled"
    assert broker.transport.paid_count == broker.signer.calls == 1


@pytest.mark.asyncio
async def test_changed_resource_or_legacy_challenge_rejected(broker):
    broker.transport.required_changes = {"resource": {"url": "https://unapproved.invalid/pay"}}
    with pytest.raises(BrokerError) as error:
        await broker.request(payload())
    assert error.value.code == "UNAPPROVED_QUOTE"
    broker.transport.required_changes = {"x402Version": 1}
    with pytest.raises(BrokerError):
        await broker.request(payload())
    assert broker.signer.calls == 0


@pytest.mark.asyncio
async def test_caps_balance_and_daily_limit_fail_before_signing(broker):
    broker.transport.quote_changes = {"amount": "200001"}
    with pytest.raises(BrokerError) as cap:
        await broker.request(payload())
    assert cap.value.code == "CAP_EXCEEDED"
    broker.transport.quote_changes = {}
    broker.transport.balance_value = 124999
    with pytest.raises(BrokerError) as unfunded:
        await broker.request(payload())
    assert unfunded.value.code == "UNFUNDED"
    assert (await broker.funding())["status"] == "unfunded"
    broker.transport.balance_value = 1000000
    broker.config = replace(broker.config, daily_limit_atomic=99999)
    with pytest.raises(BrokerError) as daily:
        await broker.request(payload())
    assert daily.value.code == "DAILY_LIMIT"
    assert (await broker.funding())["reason"] == "DAILY_LIMIT"
    assert broker.signer.calls == 0


@pytest.mark.asyncio
async def test_concurrent_identical_requests_only_pay_once(broker):
    request = payload()
    results = await asyncio.gather(*(broker.request(request) for _ in range(8)))
    assert all(result == results[0] for result in results)
    assert broker.transport.paid_count == 1


@pytest.mark.asyncio
async def test_normal_authorized_payment_does_not_fault_other_agent_funding_preflight(broker):
    original_post = broker.transport.post
    entered, release = asyncio.Event(), asyncio.Event()
    async def blocked_post(protocol, body, payment_header=None, request_id=""):
        if payment_header:
            entered.set()
            await release.wait()
        return await original_post(protocol, body, payment_header, request_id)
    broker.transport.post = blocked_post
    request = payload()
    task = asyncio.create_task(broker.request(request))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert broker.ledger.get(request["request_id"])["status"] == "authorized"
        state = await broker.funding()
        assert state["status"] == "ready"
        assert state["reason"] is None
        assert state["reserved_atomic"] == "100000"
        assert state["daily_spent_and_reserved_atomic"] == "100000"
        assert state["receipts"][0]["status"] == "authorized"
        assert broker.ledger.totals()["unknown"] == 0
    finally:
        release.set()
        await task
    assert broker.ledger.totals()["reserved"] == 0


@pytest.mark.asyncio
async def test_logical_deadline_includes_payment_queue_and_never_authorizes_timed_out_waiter(broker):
    broker.config = replace(broker.config, request_deadline_seconds=0.02)
    await broker.lock.acquire()
    try:
        request = payload()
        with pytest.raises(BrokerError) as error:
            await broker.request(request)
        assert error.value.code == "UPSTREAM_UNAVAILABLE"
        assert error.value.retryable is True
        assert broker.ledger.get(request["request_id"]) is None
        assert broker.signer.calls == 0
    finally:
        broker.lock.release()


@pytest.mark.asyncio
async def test_logical_deadline_after_authorization_keeps_uncertain_payment_reserved(broker):
    broker.config = replace(broker.config, request_deadline_seconds=0.02)
    original_post = broker.transport.post
    async def blocked_post(protocol, body, payment_header=None, request_id=""):
        if payment_header:
            await asyncio.Event().wait()
        return await original_post(protocol, body, payment_header, request_id)
    broker.transport.post = blocked_post
    request = payload()
    with pytest.raises(BrokerError) as error:
        await broker.request(request)
    assert error.value.code == "SETTLEMENT_UNKNOWN"
    assert broker.ledger.get(request["request_id"])["status"] == "unknown"
    assert broker.ledger.totals()["reserved"] == 100000
    assert (await broker.funding())["reason"] == "SETTLEMENT_UNKNOWN"


@pytest.mark.asyncio
async def test_pre_sign_failures_release_reservation_and_keep_retry_identity(broker):
    request = payload()
    broker.signer.fail = True
    with pytest.raises(BrokerError) as error:
        await broker.request(request)
    assert error.value.code == "SIGNING_FAILED"
    assert broker.ledger.totals()["reserved"] == 0
    assert broker.transport.paid_count == 0
    broker.signer.fail = False
    assert (await broker.request(request))["payment"]["status"] == "settled"


@pytest.mark.asyncio
async def test_unpaid_probe_failure_does_not_turn_into_unknown_payment(broker):
    request = payload()
    broker.transport.probe_fail = True
    with pytest.raises(BrokerError) as error:
        await broker.request(request)
    assert error.value.code == "UPSTREAM_UNAVAILABLE"
    assert broker.ledger.get(request["request_id"])["status"] == "intent"
    assert broker.signer.calls == 0


@pytest.mark.asyncio
async def test_paid_timeout_is_durable_unknown_and_blocks_all_new_payments(broker):
    request = payload()
    broker.transport.paid_fail = True
    with pytest.raises(BrokerError) as error:
        await broker.request(request)
    assert error.value.code == "SETTLEMENT_UNKNOWN"
    assert broker.ledger.totals()["reserved"] == 100000
    for retry in (request, payload()):
        with pytest.raises(BrokerError) as blocked:
            await broker.request(retry)
        assert blocked.value.code == "SETTLEMENT_UNKNOWN"
    assert broker.transport.paid_count == 1
    assert (await broker.funding())["reason"] == "SETTLEMENT_UNKNOWN"
    broker.ledger.close()
    broker.ledger = PaymentLedger(broker.config.db_path, broker.config.encryption_key)
    assert broker.ledger.get(request["request_id"])["status"] == "unknown"


@pytest.mark.asyncio
async def test_late_confirmation_preserves_response_and_reconcile_never_pays_again(broker):
    request = payload()
    broker.transport.confirm = False
    with pytest.raises(BrokerError):
        await broker.request(request)
    cached = await broker.request(request)
    assert cached["payment"]["status"] == "unknown"
    assert cached["response"] == broker.transport.vendor
    with pytest.raises(BrokerError):
        await broker.reconcile(request["request_id"])
    broker.transport.confirm = True
    receipt = await broker.reconcile(request["request_id"])
    assert receipt["response_recovered"] is True
    assert (await broker.request(request))["payment"]["status"] == "settled"
    assert broker.transport.paid_count == 1
    assert broker.ledger.totals()["reserved"] == 0


@pytest.mark.asyncio
async def test_a_prior_receipt_cannot_settle_another_same_amount_response_after_restart(broker):
    first_request = payload()
    first = await broker.request(first_request)
    before = broker.ledger.get(first_request["request_id"])
    broker.ledger.close()
    broker.ledger = PaymentLedger(broker.config.db_path, broker.config.encryption_key)
    # The test signer intentionally repeats a memo, and the fake gateway returns
    # the old transaction: the ledger must independently reject attribution.
    later_request = payload()
    with pytest.raises(BrokerError) as replayed:
        await broker.request(later_request)
    assert replayed.value.code == "SETTLEMENT_UNKNOWN"
    later = broker.ledger.get(later_request["request_id"])
    assert later["status"] == "unknown" and later["tx"] is None
    assert broker.ledger.get(first_request["request_id"]) == before
    assert broker.ledger.totals()["reserved"] == broker.ledger.totals()["settled"] == 100000
    assert await broker.request(first_request) == first
    with pytest.raises(BrokerError) as blocked:
        await broker.request(payload())
    assert blocked.value.code == "SETTLEMENT_UNKNOWN"
    assert broker.transport.paid_count == broker.signer.calls == 2


@pytest.mark.asyncio
async def test_reconcile_cannot_assign_an_existing_receipt_to_another_same_amount_intent(broker):
    first_request = payload()
    await broker.request(first_request)
    before = broker.ledger.get(first_request["request_id"])
    later_request = payload()
    broker.ledger.intent(later_request["request_id"], "authored-fixture-hash", "responses", MODEL)
    broker.ledger.update(later_request["request_id"], status="unknown", amount=100000,
                         pay_to=PAY_TO, memo="fixture-unique-memo")
    later_before = broker.ledger.get(later_request["request_id"])
    with pytest.raises(BrokerError) as denied:
        await broker.reconcile(later_request["request_id"], TX)
    assert denied.value.code == "INVALID_RECONCILIATION"
    assert broker.ledger.get(later_request["request_id"]) == later_before
    assert broker.ledger.get(first_request["request_id"]) == before
    assert broker.ledger.totals()["reserved"] == broker.ledger.totals()["settled"] == 100000
    assert broker.transport.paid_count == broker.signer.calls == 1


@pytest.mark.asyncio
async def test_missing_receipt_is_unknown_even_when_vendor_json_arrived(broker):
    broker.transport.headers = {"PAYMENT-RESPONSE": "", "X-Payment-Settled": "false"}
    request = payload()
    with pytest.raises(BrokerError) as error:
        await broker.request(request)
    assert error.value.code == "SETTLEMENT_UNKNOWN"
    assert (await broker.request(request))["payment"]["status"] == "unknown"
    assert broker.ledger.totals()["settled"] == 0


@pytest.mark.parametrize("headers,raw", [({"X-Fallback-Used": "true"}, None), ({"X-Context-Truncated": "true"}, None), ({}, b"not valid JSON")])
@pytest.mark.asyncio
async def test_known_paid_bad_response_keeps_cost_and_cannot_repay(broker, headers, raw):
    request = payload()
    broker.transport.headers = headers
    broker.transport.paid_raw = raw
    with pytest.raises(BrokerError) as error:
        await broker.request(request)
    assert error.value.code == "PAID_UPSTREAM_REJECTED"
    assert broker.ledger.get(request["request_id"])["status"] == "settled_error"
    assert broker.ledger.totals()["settled"] == 100000
    with pytest.raises(BrokerError) as retry:
        await broker.request(request)
    assert retry.value.code == "PAID_RESPONSE_UNAVAILABLE"
    assert broker.transport.paid_count == 1


@pytest.mark.asyncio
async def test_cancellation_before_vs_after_paid_send_has_different_reservation_state(broker):
    first = payload()
    broker.signer.cancelled = True
    with pytest.raises(asyncio.CancelledError):
        await broker.request(first)
    assert broker.ledger.get(first["request_id"])["status"] == "intent"
    broker.signer.cancelled = False
    broker.transport.paid_cancel = True
    second = payload()
    with pytest.raises(asyncio.CancelledError):
        await broker.request(second)
    assert broker.ledger.get(second["request_id"])["status"] == "unknown"


def test_another_writer_cannot_reset_live_payment_reservations(broker):
    with pytest.raises(RuntimeError, match="one payment broker"):
        PaymentLedger(broker.config.db_path, broker.config.encryption_key)


@pytest.mark.asyncio
async def test_restart_recovers_authorized_as_unknown_and_unsigned_reservation_as_intent(broker):
    first, second = payload(), payload()
    for request in (first, second):
        broker.ledger.intent(request["request_id"], "hash", "responses", MODEL)
    broker.ledger.update(first["request_id"], status="authorized", amount=100000)
    broker.ledger.update(second["request_id"], status="reserved", amount=100000)
    broker.ledger.close()
    broker.ledger = PaymentLedger(broker.config.db_path, broker.config.encryption_key)
    assert broker.ledger.get(first["request_id"])["status"] == "unknown"
    assert broker.ledger.get(second["request_id"])["status"] == "intent"
    assert broker.ledger.totals()["reserved"] == 100000


@pytest.mark.asyncio
async def test_daily_headroom_resets_but_unresolved_payments_remain_reserved(broker, monkeypatch):
    monkeypatch.setattr("observatory.x402_broker.utc_now", lambda: "2026-10-05T12:00:00Z")
    request = payload()
    await broker.request(request)
    assert broker.ledger.totals()["daily"] == 100000
    monkeypatch.setattr("observatory.x402_broker.utc_now", lambda: "2026-10-06T12:00:00Z")
    assert broker.ledger.totals()["daily"] == 0
    broker.ledger.update(request["request_id"], status="unknown")
    assert broker.ledger.totals()["daily"] == 100000
    assert (await broker.funding())["status"] == "unknown"


@pytest.mark.asyncio
async def test_official_sdk_signs_only_selected_exact_offer_without_rpc_or_transfer(broker, monkeypatch):
    pytest.importorskip("x402")
    from solders.hash import Hash
    from solders.keypair import Keypair
    from solders.pubkey import Pubkey
    from solders.transaction import VersionedTransaction
    from x402.http.utils import decode_payment_signature_header
    # Public deterministic test fixture, never funded or submitted to a network.
    fixture = Keypair.from_seed(bytes(range(32)))
    config = replace(broker.config, private_key=str(fixture))
    signer = OfficialSvmSigner(config)
    monkeypatch.setattr("x402.mechanisms.svm.exact.client.SolanaClient", lambda _url: object())
    monkeypatch.setattr("x402.mechanisms.svm.exact.client.get_cached_mint_metadata", lambda *args: SimpleNamespace(token_program=Pubkey.from_string(TOKEN_PROGRAM), decimals=6))
    monkeypatch.setattr("x402.mechanisms.svm.exact.client.resolve_blockhash", lambda *args: Hash.default())
    required = challenge()
    signed = await signer.sign(required, required["accepts"][0])
    decoded = decode_payment_signature_header(signed.header)
    assert decoded.accepted.amount == "100000"
    assert decoded.accepted.asset == ASSET
    assert decoded.accepted.pay_to == PAY_TO
    assert decoded.accepted.network == NETWORK
    transaction = VersionedTransaction.from_bytes(base64.b64decode(decoded.payload["transaction"]))
    assert transaction.signatures[1].verify(fixture.pubkey(), b"\x80" + bytes(transaction.message))
    token_instruction = next(item for item in transaction.message.instructions if str(transaction.message.account_keys[item.program_id_index]) == TOKEN_PROGRAM)
    assert bytes(token_instruction.data) == b"\x0c" + (100000).to_bytes(8, "little") + b"\x06"
    assert signed.memo and len(signed.memo) == 32 and all(character in "0123456789abcdef" for character in signed.memo)
    # The pinned SDK uses a new random nonce for absent, null and empty seller
    # memos, even when amount/recipient/blockhash remain identical.
    signed_nonces = {signed.memo}
    for merchant_memo in (None, ""):
        fresh_required = challenge(extra={"feePayer": FEE_PAYER, "memo": merchant_memo})
        fresh = await signer.sign(fresh_required, fresh_required["accepts"][0])
        assert fresh.memo and len(fresh.memo) == 32
        assert fresh.memo not in signed_nonces
        signed_nonces.add(fresh.memo)
        fresh_payload = decode_payment_signature_header(fresh.header)
        fresh_transaction = VersionedTransaction.from_bytes(base64.b64decode(fresh_payload.payload["transaction"]))
        assert fresh_transaction.signatures[1].verify(fixture.pubkey(), b"\x80" + bytes(fresh_transaction.message))
    # A historical transfer absent from this ledger still must not settle a
    # newer same-amount authorization. Verify against the actual fresh SDK
    # nonce, not a gateway success flag or a timestamp heuristic.
    from x402.mechanisms.svm.utils import derive_ata
    transport = GatewayTransport(config)
    old_chain_receipt = {"meta": {"err": None, "innerInstructions": []}, "transaction": {"message": {"instructions": [
        {"programId": TOKEN_PROGRAM, "parsed": {"type": "transferChecked", "info": {
            "authority": signer.address, "mint": ASSET, "source": derive_ata(signer.address, ASSET),
            "destination": derive_ata(PAY_TO, ASSET), "tokenAmount": {"amount": "100000", "decimals": 6}}}},
        {"programId": MEMO_PROGRAM, "parsed": signed.memo}]}}}
    async def historical_rpc(method, params):
        assert method == "getTransaction"
        return old_chain_receipt
    transport.rpc = historical_rpc
    try:
        assert broker.ledger.db.execute("SELECT 1 FROM intents WHERE tx=?", (TX,)).fetchone() is None
        assert await transport.verify_receipt(TX, signer.address, PAY_TO, 100000, signed.memo)
        assert not await transport.verify_receipt(TX, signer.address, PAY_TO, 100000, fresh.memo)
    finally:
        await transport.close()
    fixed_required = challenge(extra={"feePayer": FEE_PAYER, "memo": "a" * 32})
    with pytest.raises(ValueError, match="fresh client nonce"):
        await signer.sign(fixed_required, fixed_required["accepts"][0])
    assert "PRIVATE" not in repr(signed)


@pytest.mark.asyncio
async def test_rpc_receipt_checks_exact_transfer_and_memo_not_just_success_flag(broker):
    pytest.importorskip("x402")
    from x402.mechanisms.svm.utils import derive_ata
    transport = GatewayTransport(broker.config)
    transfer = {"programId": TOKEN_PROGRAM, "parsed": {"type": "transferChecked", "info": {
        "authority": WALLET, "mint": ASSET, "source": derive_ata(WALLET, ASSET), "destination": derive_ata(PAY_TO, ASSET),
        "tokenAmount": {"amount": "100000", "decimals": 6}}}}
    memo = {"programId": MEMO_PROGRAM, "parsed": "fixture-unique-memo"}
    chain_result = {"meta": {"err": None, "innerInstructions": []}, "transaction": {"message": {"instructions": [transfer, memo]}}}
    async def rpc(method, params):
        assert method == "getTransaction"
        return chain_result
    transport.rpc = rpc
    try:
        assert await transport.verify_receipt(TX, WALLET, PAY_TO, 100000, "fixture-unique-memo")
        assert not await transport.verify_receipt(TX, WALLET, PAY_TO, 100001, "fixture-unique-memo")
        assert not await transport.verify_receipt(TX, WALLET, PAY_TO, 100000, "unrelated-memo")
        transfer["parsed"]["info"]["mint"] = "unrelated-mint"
        assert not await transport.verify_receipt(TX, WALLET, PAY_TO, 100000, "fixture-unique-memo")
    finally:
        await transport.close()


@pytest.mark.asyncio
async def test_gateway_transport_uses_only_fixed_native_urls_and_no_bearer_wallet_key(broker):
    transport = GatewayTransport(broker.config)
    seen = []
    def handler(request):
        seen.append(request)
        return httpx.Response(402)
    await transport.http.aclose()
    transport.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await transport.post("messages", payload("messages")["body"], "fixture-signature", str(uuid4()))
        assert str(seen[0].url) == GATEWAY + "/messages"
        assert seen[0].headers["anthropic-version"] == "2023-06-01"
        assert seen[0].headers["payment-signature"] == "fixture-signature"
        assert "authorization" not in seen[0].headers
        assert "NON_WALLET_SIGNER" not in str(seen[0].headers)
    finally:
        await transport.close()
