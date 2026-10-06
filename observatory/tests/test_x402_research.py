"""Research payment transport/control tests; no browser, vendor or chain calls."""
import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel

from observatory.research import ResearchSupervisor
from observatory.research_llm import (MonitoredResearchModel, X402MessagesResearchModel,
                                     X402ResponsesResearchModel, validate_research_settings)
from observatory.store import Store
from observatory.x402_client import (ResearchFundingError, ResearchPermanentError,
                                    ResearchTransientError, X402BrokerClient, BROKER_POST_TIMEOUT_SECONDS,
                                    BROKER_REQUEST_DEADLINE_SECONDS, X402_AGENT_LLM_TIMEOUT_SECONDS)


MODEL = "openai/gpt-6-astra"
CLAUDE = "anthropic/claude-fable-5-1"


@pytest.fixture
def store(tmp_path):
    db = Store(tmp_path / "state.sqlite3")
    yield db
    db.close()


@pytest.fixture(autouse=True)
def broker_environment(monkeypatch):
    monkeypatch.setenv("OBSERVATORY_PAYMENT_BROKER_URL", "https://broker.test")
    monkeypatch.setenv("OBSERVATORY_PAYMENT_BROKER_TOKEN", "internal-fixture-secret")


def catalog():
    return {"models": [
        {"id": MODEL, "supportedprotocols": ["responses"],
         "capabilities": {"vision": True, "structured_actions": True, "structured_outputs": True}},
        {"id": CLAUDE, "supportedprotocols": ["messages"],
         "capabilities": {"vision": True, "structured_actions": True, "tool_calling": True}},
    ]}


def handler(calls, *, models=None, funding=None, post=None):
    def respond(request):
        calls.append(request)
        assert request.headers["Authorization"] == "Bearer internal-fixture-secret"
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/catalog":
            return httpx.Response(200, json=models if models is not None else catalog())
        if request.url.path == "/funding":
            return httpx.Response(200, json=funding or {"status": "ready", "configured": True, "enabled": True})
        return post(request) if post else httpx.Response(200, json={"response": {"ok": True}, "payment": {"status": "settled"}})
    return respond


@pytest.mark.asyncio
async def test_broker_preflight_pins_catalog_model_and_uses_only_internal_auth():
    calls = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler(calls))) as http:
        async with X402BrokerClient(client=http) as broker:
            selected = await broker.preflight(MODEL, "responses")
    assert selected["id"] == MODEL
    assert [item.url.path for item in calls] == ["/health", "/catalog", "/funding"]
    assert all(item.method == "GET" for item in calls)


@pytest.mark.parametrize("model,protocol,models,code", [
    ("unknown", "responses", catalog(), "MODEL_NOT_IN_CATALOG"),
    (MODEL, "messages", catalog(), "MODEL_PROTOCOL_UNSUPPORTED"),
    (MODEL, "responses", {"models": [{"id": MODEL, "supportedprotocols": ["responses"], "capabilities": {"vision": False}}]}, "MODEL_CAPABILITIES_UNSUPPORTED"),
    (MODEL, "responses", {"models": [{"id": MODEL, "supportedprotocols": ["responses"], "capabilities": {"vision": True, "structured_actions": True}}]}, "MODEL_SCHEMA_UNSUPPORTED"),
])
@pytest.mark.asyncio
async def test_catalog_mismatch_fails_without_payment(model, protocol, models, code):
    calls = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler(calls, models=models))) as http:
        with pytest.raises(ResearchPermanentError) as error:
            await X402BrokerClient(client=http).preflight(model, protocol)
    assert error.value.code == code
    assert not any(item.method == "POST" for item in calls)


@pytest.mark.parametrize("funding,exception,code", [
    ({"status": "unfunded"}, ResearchFundingError, "FUNDING_NOT_READY"),
    ({"status": "unknown"}, ResearchFundingError, "FUNDING_NOT_READY"),
    ({"status": "unknown", "reason": "SETTLEMENT_UNKNOWN"}, ResearchPermanentError, "SETTLEMENT_UNKNOWN"),
    ({"status": "configured", "reason": "SIGNER_NOT_CONFIGURED"}, ResearchPermanentError, "SIGNER_NOT_CONFIGURED"),
    ({"status": "configured", "reason": "SIGNER_INVALID"}, ResearchPermanentError, "SIGNER_INVALID"),
    ({"status": "configured", "reason": "NOT_CONFIGURED"}, ResearchPermanentError, "NOT_CONFIGURED"),
])
@pytest.mark.asyncio
async def test_funding_and_unresolved_settlement_are_distinct(funding, exception, code):
    calls = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler(calls, funding=funding))) as http:
        with pytest.raises(exception) as error:
            await X402BrokerClient(client=http).preflight(MODEL, "responses")
    assert error.value.code == code
    assert not any(item.method == "POST" for item in calls)


@pytest.mark.parametrize("status,code,exception", [
    (401, "AUTH_FAILED", ResearchPermanentError),
    (402, "UNFUNDED", ResearchFundingError),
    (402, "SETTLEMENT_UNKNOWN", ResearchPermanentError),
    (503, "FUNDING_UNKNOWN", ResearchFundingError),
    (409, "SETTLEMENT_UNKNOWN", ResearchPermanentError),
    (422, "CAP_EXCEEDED", ResearchPermanentError),
    (429, "DAILY_LIMIT", ResearchFundingError),
    (502, "PAID_UPSTREAM_REJECTED", ResearchPermanentError),
])
@pytest.mark.asyncio
async def test_broker_errors_are_actionable_and_never_expose_upstream_bodies(status, code, exception):
    def respond(request):
        return httpx.Response(status, json={"error": {"code": code, "message": "private prompt sk-do-not-publish"}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        with pytest.raises(exception) as error:
            await X402BrokerClient(client=http).request("responses", {"model": MODEL})
    assert "private prompt" not in str(error.value)
    assert "sk-do-not-publish" not in str(error.value)


@pytest.mark.asyncio
async def test_transient_retry_reuses_same_payment_request_id():
    calls = []
    def respond(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(503, json={"error": {"code": "UPSTREAM_UNAVAILABLE"}})
        return httpx.Response(200, json={"response": {"ok": True}, "payment": {"status": "settled"}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        broker = X402BrokerClient(client=http)
        assert await broker.request("responses", {"model": MODEL}) == {"ok": True}
    assert calls[0]["request_id"] == calls[1]["request_id"]


@pytest.mark.asyncio
async def test_unknown_payment_with_vendor_response_still_requires_reconciliation():
    def respond(request):
        return httpx.Response(200, json={"response": {"ok": True}, "payment": {"status": "unknown"}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        with pytest.raises(ResearchPermanentError, match="settlement"):
            await X402BrokerClient(client=http).request("responses", {"model": MODEL})


@pytest.mark.parametrize("status", [None, "", "authorized", "pending", "arbitrary"])
@pytest.mark.asyncio
async def test_only_settled_payment_success_is_accepted(status):
    def respond(request):
        payment = {} if status is None else {"status": status}
        return httpx.Response(200, json={"response": {"ok": True}, "payment": payment})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        with pytest.raises(ResearchPermanentError) as error:
            await X402BrokerClient(client=http).request("responses", {"model": MODEL})
    assert error.value.code == "SETTLEMENT_UNKNOWN"


@pytest.mark.parametrize("url,allowed", [
    ("http://127.0.0.1:8062", True), ("http://payments:8062", True),
    ("https://broker.example", True), ("http://broker.example", False),
    ("http://192.168.1.5:8062", False),
])
def test_remote_broker_requires_https_and_local_addresses_are_explicit(monkeypatch, url, allowed):
    monkeypatch.setenv("OBSERVATORY_PAYMENT_BROKER_URL", url)
    transport = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={})))
    if allowed:
        assert X402BrokerClient(client=transport).base == url
    else:
        with pytest.raises(ResearchPermanentError) as error:
            X402BrokerClient(client=transport)
        assert error.value.code == "BROKER_HTTPS_REQUIRED"


@pytest.mark.asyncio
async def test_post_transport_timeout_never_automatically_retries():
    calls = []
    def respond(request):
        calls.append(request)
        raise httpx.ReadTimeout("private connection detail", request=request)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        with pytest.raises(ResearchPermanentError) as error:
            await X402BrokerClient(client=http).request("responses", {"model": MODEL})
    assert error.value.code == "BROKER_REQUEST_UNKNOWN"
    assert len(calls) == 1
    assert "private connection" not in str(error.value)


@pytest.mark.asyncio
async def test_post_read_timeout_fits_inside_logical_and_agent_deadlines():
    calls = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler(calls))) as http:
        await X402BrokerClient(client=http).request("responses", {"model": MODEL})
    assert calls[0].extensions["timeout"]["read"] == BROKER_POST_TIMEOUT_SECONDS
    assert calls[0].extensions["timeout"]["connect"] == 10
    assert BROKER_POST_TIMEOUT_SECONDS < BROKER_REQUEST_DEADLINE_SECONDS < X402_AGENT_LLM_TIMEOUT_SECONDS


class Action(BaseModel):
    next_goal: str


def responses_result():
    return {"id": "resp-fixture", "created_at": 0, "model": "gpt-6-astra", "object": "response",
            "status": "completed", "parallel_tool_calls": False, "tool_choice": "auto", "tools": [],
            "output": [{"id": "msg-fixture", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": '{"next_goal":"Read competing evidence"}', "annotations": []}]}],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5,
                      "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}}}


@pytest.mark.asyncio
async def test_responses_broker_transport_preserves_vision_schema_and_exact_model():
    from browser_use.llm.messages import UserMessage, ContentPartImageParam, ContentPartTextParam, ImageURL
    calls = []
    def post(request):
        return httpx.Response(200, json={"response": responses_result(), "payment": {"status": "settled"}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler(calls, post=post))) as http:
        model = X402ResponsesResearchModel(MODEL, X402BrokerClient(client=http), max_output_tokens=2048)
        result = await model.ainvoke([UserMessage(content=[ContentPartTextParam(text="Read"), ContentPartImageParam(image_url=ImageURL(url="data:image/png;base64,Zg=="))])], Action)
    payload = json.loads(calls[-1].content)
    assert payload["protocol"] == "responses" and payload["body"]["model"] == MODEL
    assert payload["body"]["text"]["format"]["strict"] is True
    assert payload["body"]["max_output_tokens"] == 2048
    assert payload["body"]["input"][0]["content"][1]["type"] == "input_image"
    assert result.completion.next_goal == "Read competing evidence"
    assert result.usage.total_tokens == 5


@pytest.mark.asyncio
async def test_messages_broker_transport_reuses_anthropic_image_and_tool_schema():
    from browser_use.llm.messages import UserMessage, ContentPartImageParam, ContentPartTextParam, ImageURL
    calls = []
    def post(request):
        response = {"id": "msg-fixture", "type": "message", "role": "assistant", "model": "claude-fable-5-1",
                    "stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "tool-fixture", "name": "Action",
                                                            "input": {"next_goal": "Inspect primary paper"}}],
                    "usage": {"input_tokens": 3, "output_tokens": 2}}
        return httpx.Response(200, json={"response": response, "payment": {"status": "settled"}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler(calls, post=post))) as http:
        model = X402MessagesResearchModel(CLAUDE, X402BrokerClient(client=http))
        result = await model.ainvoke([UserMessage(content=[ContentPartTextParam(text="Read"), ContentPartImageParam(image_url=ImageURL(url="data:image/png;base64,Zg=="))])], Action)
    payload = json.loads(calls[-1].content)
    assert payload["protocol"] == "messages" and payload["body"]["model"] == CLAUDE
    assert payload["body"]["messages"][0]["content"][1]["type"] == "image"
    assert payload["body"]["tools"][0]["input_schema"]["type"] == "object"
    assert "thinking" not in payload["body"]
    assert result.completion.next_goal == "Inspect primary paper"


@pytest.mark.asyncio
async def test_messages_sdk_wrap_does_not_hide_funding_failure_from_monitor():
    failures = []
    def post(request):
        return httpx.Response(402, json={"error": {"code": "UNFUNDED"}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(post)) as http:
        model = X402MessagesResearchModel(CLAUDE, X402BrokerClient(client=http))
        monitor = MonitoredResearchModel(model, failures.append)
        with pytest.raises(ResearchFundingError):
            await monitor.ainvoke([], Action)
    assert len(failures) == 1 and failures[0].code == "UNFUNDED"


@pytest.mark.asyncio
async def test_nonreasoning_responses_model_does_not_receive_reasoning_control():
    calls = []
    def post(request):
        return httpx.Response(200, json={"response": responses_result(), "payment": {"status": "settled"}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler(calls, post=post))) as http:
        model = X402ResponsesResearchModel("openai/gpt-4.1", X402BrokerClient(client=http))
        await model.ainvoke([], Action)
    assert "reasoning" not in json.loads(calls[-1].content)["body"]


@pytest.mark.parametrize("updates", [
    {"agent_count": "six"}, {"agent_count": 1.5}, {"agent_count": True},
    {"research_protocol": "guess"}, {"steps_per_pass": 100}, {"reasoning_effort": "unbounded"},
    {"research_provider": []}, {"research_protocol": {}}, {"browser_provider": []}, {"reasoning_effort": {}},
])
def test_invalid_settings_rejected_before_supervisor_casts(updates):
    with pytest.raises(ResearchPermanentError):
        validate_research_settings(updates)


@pytest.mark.asyncio
async def test_funding_preflight_fails_before_any_browser_provisioning(store, monkeypatch):
    store.save_settings({"research_provider": "x402", "research_model": MODEL, "browser_provider": "local"})
    store.set_mission({"id": "mission", "status": "running", "objective": "Research"})
    calls = []
    class LLM:
        name = MODEL
        client = None
        async def preflight(self):
            raise ResearchFundingError("UNFUNDED", "Waiting for funds.")
    @asynccontextmanager
    async def browser(*args):
        calls.append("browser")
        yield "never"
    injected = []
    def factory(settings, **kwargs):
        injected.append(kwargs)
        return LLM()
    monkeypatch.setattr("observatory.research.researcher_model", factory)
    monkeypatch.setattr("observatory.research.browser_endpoint", browser)
    supervisor = ResearchSupervisor(store)
    supervisor.mission_id = "mission"
    await supervisor._researcher("mission-scholar", "Primary research")
    assert store.get_mission()["status"] == "funding_paused"
    assert calls == []
    assert injected == [{"intent_store": store, "scope": "mission-scholar"}]


@pytest.mark.asyncio
async def test_monitored_model_funding_failure_survives_sdk_error_swallow_and_releases_worker(store, monkeypatch):
    store.set_mission({"id": "mission", "status": "running", "objective": "Research"})
    supervisor = ResearchSupervisor(store)
    supervisor.mission_id = "mission"
    cleaned = asyncio.Event()
    class LLM:
        async def ainvoke(self, *args, **kwargs):
            raise ResearchFundingError("UNFUNDED", "Waiting for funds.")
    monitor = MonitoredResearchModel(LLM(), lambda error: supervisor._fail_mission(error, mission_id="mission"))
    async def worker():
        try:
            try:
                await monitor.ainvoke([])
            except Exception:
                pass  # Browser Use can consume a failed step.
            await asyncio.Event().wait()
        finally:
            cleaned.set()
    supervisor.workers["mission-scholar"] = asyncio.create_task(worker())
    for _ in range(10):
        if store.get_mission()["status"] == "funding_paused":
            break
        await asyncio.sleep(0)
    async def no_probe(mission):
        pass
    monkeypatch.setattr(supervisor, "_check_funding", no_probe)
    await supervisor._tick()
    assert cleaned.is_set() and not supervisor.workers
    assert store.get_mission()["status"] == "funding_paused"


@pytest.mark.parametrize("operator_status", ["paused", "pausing", "stopping", "stopped"])
@pytest.mark.asyncio
async def test_late_funding_failure_cannot_override_operator_control(store, operator_status):
    store.set_mission({"id": "mission", "status": operator_status})
    supervisor = ResearchSupervisor(store)
    supervisor._fail_mission(ResearchFundingError("UNFUNDED", "Waiting for funds."), mission_id="mission")
    assert store.get_mission()["status"] == operator_status


@pytest.mark.asyncio
async def test_late_funding_failure_releases_browser_worker_while_preserving_manual_pause(store):
    store.set_mission({"id": "mission", "status": "paused"})
    supervisor = ResearchSupervisor(store)
    supervisor.mission_id = "mission"
    cleaned = asyncio.Event()
    async def worker():
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()
    supervisor.workers["mission-scholar"] = asyncio.create_task(worker())
    await asyncio.sleep(0)
    supervisor._fail_mission(ResearchFundingError("UNFUNDED", "Waiting."), mission_id="mission")
    await supervisor._tick()
    assert cleaned.is_set() and store.get_mission()["status"] == "paused"


@pytest.mark.parametrize("operator_status", [None, "pausing", "stopping"])
@pytest.mark.asyncio
async def test_funding_autoresume_only_changes_unchanged_funding_pause(store, monkeypatch, operator_status):
    store.save_settings({"research_provider": "x402", "research_model": MODEL})
    mission = store.set_mission({"id": "mission", "status": "funding_paused"})
    class Broker:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def preflight(self, model, protocol):
            if operator_status:
                store.set_mission({**store.get_mission(), "status": operator_status})
            return {"id": model}
    monkeypatch.setattr("observatory.research.X402BrokerClient", Broker)
    supervisor = ResearchSupervisor(store)
    await supervisor._check_funding(mission)
    assert store.get_mission()["status"] == (operator_status or "running")


@pytest.mark.asyncio
async def test_permanent_provider_error_faults_once_and_transient_passes_have_bounded_backoff(store, monkeypatch):
    async def no_sleep(delay):
        delays.append(delay)
    delays = []
    monkeypatch.setattr("observatory.research.asyncio.sleep", no_sleep)
    for error, expected_calls in [(ResearchPermanentError("AUTH_FAILED", "Configure auth."), 1),
                                  (ResearchTransientError("TEMPORARY", "Temporary."), 3)]:
        store.set_mission({"id": "mission", "status": "running", "objective": "Research"})
        supervisor = ResearchSupervisor(store)
        supervisor.mission_id = "mission"
        calls = []
        async def research(*args):
            calls.append("attempt")
            raise error
        monkeypatch.setattr(supervisor, "_pass", research)
        await supervisor._researcher("mission-scholar", "Primary research")
        assert len(calls) == expected_calls
        assert store.get_mission()["status"] == "faulted"
    assert delays == [2, 4]


@pytest.mark.asyncio
async def test_malformed_count_faults_but_does_not_kill_supervisor(store, monkeypatch):
    # A legacy encrypted database may predate input validation.
    monkeypatch.setattr(store, "get_settings", lambda private=False: {"agent_count": "six"})
    store.set_mission({"id": "mission", "status": "running"})
    supervisor = ResearchSupervisor(store)
    await supervisor.start()
    for _ in range(20):
        if store.get_mission()["status"] == "faulted":
            break
        await asyncio.sleep(0.01)
    assert store.get_mission()["status"] == "faulted"
    assert not supervisor.task.done()
    await supervisor.close()


@pytest.mark.asyncio
async def test_unknown_request_replays_same_uuid_after_actual_store_restart(tmp_path):
    path = tmp_path / "caller.sqlite3"
    body = {"model": MODEL, "input": [{"role": "user", "content": "PRIVATE_RESEARCH_INTENT_FIXTURE"}]}
    calls = []
    charges = []
    def respond(request):
        payload = json.loads(request.content)
        calls.append(payload)
        if len(calls) == 1:
            charges.append(payload["request_id"])
            raise httpx.ReadTimeout("response lost after a hypothetical payment", request=request)
        assert payload["request_id"] == charges[0]
        return httpx.Response(200, json={"response": {"cached": True}, "payment": {"status": "settled"}})
    first = Store(path)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        client = X402BrokerClient(client=http, intent_store=first, scope="mission-scholar")
        with pytest.raises(ResearchPermanentError) as error:
            await client.request("responses", body)
    assert error.value.code == "BROKER_REQUEST_UNKNOWN"
    assert first.pending_research_requests("mission-scholar")[0]["body"] == body
    ciphertext = first._db.execute("SELECT payload FROM research_payment_intents").fetchone()[0]
    assert b"PRIVATE_RESEARCH_INTENT_FIXTURE" not in ciphertext
    assert "PRIVATE_RESEARCH_INTENT_FIXTURE" not in json.dumps(first.state())
    first.close()
    restarted = Store(path)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            client = X402BrokerClient(client=http, intent_store=restarted, scope="mission-scholar")
            assert await client.request("responses", body) == {"cached": True}
        assert calls[0]["request_id"] == calls[1]["request_id"]
        assert len(charges) == 1
        assert restarted.pending_research_requests("mission-scholar") == []
    finally:
        restarted.close()


@pytest.mark.asyncio
async def test_changed_body_after_restart_requires_recovery_before_http(tmp_path):
    path = tmp_path / "caller.sqlite3"
    first = Store(path)
    calls = []
    def respond(request):
        calls.append(request)
        return httpx.Response(409, json={"error": {"code": "SETTLEMENT_UNKNOWN"}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        with pytest.raises(ResearchPermanentError):
            await X402BrokerClient(client=http, intent_store=first, scope="mission-scholar").request(
                "responses", {"model": MODEL, "input": "Original browser context"})
    first.close()
    restarted = Store(path)
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            with pytest.raises(ResearchPermanentError) as error:
                await X402BrokerClient(client=http, intent_store=restarted, scope="mission-scholar").request(
                    "responses", {"model": MODEL, "input": "Changed browser context"})
        assert error.value.code == "CALLER_RECOVERY_REQUIRED"
        assert len(calls) == 1 and len(restarted.pending_research_requests("mission-scholar")) == 1
    finally:
        restarted.close()


@pytest.mark.parametrize("status,code", [
    (402, "UNFUNDED"), (503, "FUNDING_UNKNOWN"), (429, "DAILY_LIMIT"),
    (422, "INVALID_REQUEST"), (422, "UNAPPROVED_QUOTE"), (422, "UNSUPPORTED_MODEL"),
    (422, "CAP_EXCEEDED"), (503, "SIGNING_FAILED"), (503, "UPSTREAM_UNAVAILABLE"),
    (503, "CATALOG_UNAVAILABLE"),
])
@pytest.mark.asyncio
async def test_only_explicit_definitely_unpaid_codes_resolve_durable_pending(store, monkeypatch, status, code):
    async def no_sleep(delay):
        pass
    monkeypatch.setattr("observatory.x402_client.asyncio.sleep", no_sleep)
    def respond(request):
        return httpx.Response(status, json={"error": {"code": code}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        with pytest.raises((ResearchFundingError, ResearchPermanentError)):
            await X402BrokerClient(client=http, intent_store=store, scope="mission-scholar").request(
                "responses", {"model": MODEL})
    assert store.pending_research_requests("mission-scholar") == []


@pytest.mark.parametrize("status,code", [
    (402, None), (409, "SETTLEMENT_UNKNOWN"), (502, "PAID_UPSTREAM_REJECTED"),
])
@pytest.mark.asyncio
async def test_unknown_or_paid_error_never_resolves_durable_pending(store, status, code):
    def respond(request):
        return httpx.Response(status, json={"error": {"code": code}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        with pytest.raises((ResearchFundingError, ResearchPermanentError)):
            await X402BrokerClient(client=http, intent_store=store, scope="mission-scholar").request(
                "responses", {"model": MODEL})
    assert len(store.pending_research_requests("mission-scholar")) == 1


@pytest.mark.asyncio
async def test_settled_raw_vendor_response_completes_pending_before_vendor_schema_validation(store):
    def respond(request):
        return httpx.Response(200, json={"response": {"invalid_native_schema": True}, "payment": {"status": "settled"}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        model = X402ResponsesResearchModel(MODEL, X402BrokerClient(client=http, intent_store=store, scope="mission-scholar"))
        with pytest.raises(ResearchPermanentError) as error:
            await model.ainvoke([], Action)
    assert error.value.code == "VENDOR_SCHEMA_INVALID"
    assert store.pending_research_requests("mission-scholar") == []


@pytest.mark.asyncio
async def test_known_unpaid_funding_pause_resumes_new_context_after_store_restart(tmp_path, monkeypatch):
    path = tmp_path / "caller.sqlite3"
    first = Store(path)
    first.save_settings({"research_provider": "x402", "research_model": MODEL})
    first.set_mission({"id": "mission", "status": "running", "objective": "Research"})
    requests = []
    def unfunded(request):
        requests.append(json.loads(request.content))
        return httpx.Response(402, json={"error": {"code": "UNFUNDED"}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(unfunded)) as http:
        try:
            await X402BrokerClient(client=http, intent_store=first, scope="mission-scholar").request(
                "responses", {"model": MODEL, "input": "Old browser context"})
        except ResearchFundingError as error:
            ResearchSupervisor(first)._fail_mission(error, mission_id="mission")
    assert first.get_mission()["status"] == "funding_paused"
    assert first.pending_research_requests("mission-scholar") == []
    first.close()
    restarted = Store(path)
    try:
        calls = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler(calls))) as http:
            monkeypatch.setattr("observatory.research.X402BrokerClient", lambda: X402BrokerClient(client=http))
            supervisor = ResearchSupervisor(restarted)
            await supervisor._check_funding(restarted.get_mission())
            assert restarted.get_mission()["status"] == "running"
            await X402BrokerClient(client=http, intent_store=restarted, scope="mission-scholar").request(
                "responses", {"model": MODEL, "input": "Fresh browser context"})
        paid = next(json.loads(request.content) for request in calls if request.method == "POST")
        assert paid["request_id"] != requests[0]["request_id"]
        assert restarted.pending_research_requests("mission-scholar") == []
    finally:
        restarted.close()


@pytest.mark.parametrize("request_id", ["../secret", "not-uuid", "550E8400-E29B-41D4-A716-446655440000"])
@pytest.mark.asyncio
async def test_owner_request_lookup_validates_uuid_before_http(request_id):
    calls = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler(calls))) as http:
        with pytest.raises(ResearchPermanentError) as error:
            await X402BrokerClient(client=http).get_request(request_id)
    assert error.value.code == "INVALID_REQUEST_ID" and calls == []


@pytest.mark.asyncio
async def test_owner_abandon_helper_targets_only_local_ledger_endpoint():
    calls = []
    identifier = "550e8400-e29b-41d4-a716-446655440000"
    body = {"model": MODEL, "input": "Private exact request"}
    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={"request_id": identifier, "state": "cancelled", "status": "cancelled",
                                        "abandoned": True, "in_flight": False})
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
        result = await X402BrokerClient(client=http).abandon_request(identifier, "responses", body)
    assert result["abandoned"] is True and result["state"] == "cancelled"
    assert len(calls) == 1
    assert calls[0].url.path == "/v1/requests/" + identifier + "/abandon"
    assert json.loads(calls[0].content) == {"protocol": "responses", "body": body}
    assert calls[0].headers["Authorization"] == "Bearer internal-fixture-secret"


@pytest.mark.asyncio
async def test_durable_pending_preflight_faults_before_browser_or_broker_calls(store, monkeypatch):
    import hashlib
    body = {"model": MODEL, "input": "Previous page action"}
    digest = hashlib.sha256(json.dumps({"protocol": "responses", "body": body}, sort_keys=True).encode()).hexdigest()
    store.reserve_research_request("mission-scholar", digest, "responses", body)
    store.save_settings({"research_provider": "x402", "research_model": MODEL})
    store.set_mission({"id": "mission", "status": "running", "objective": "Research"})
    calls = []
    @asynccontextmanager
    async def browser(*args):
        calls.append("browser")
        yield "never"
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler(calls))) as http:
        broker = X402BrokerClient(client=http, intent_store=store, scope="mission-scholar")
        with pytest.raises(ResearchPermanentError) as error:
            await broker.preflight(MODEL, "responses")
        assert error.value.code == "CALLER_RECOVERY_REQUIRED"
        class LLM:
            name = MODEL
            client = broker
            async def preflight(self):
                return await broker.preflight(MODEL, "responses")
        monkeypatch.setattr("observatory.research.researcher_model", lambda settings, **kwargs: LLM())
        monkeypatch.setattr("observatory.research.browser_endpoint", browser)
        supervisor = ResearchSupervisor(store)
        supervisor.mission_id = "mission"
        await supervisor._researcher("mission-scholar", "Primary evidence")
    assert calls == []
    assert store.get_mission()["status"] == "faulted"
    assert store.get_mission()["error_code"] == "CALLER_RECOVERY_REQUIRED"
