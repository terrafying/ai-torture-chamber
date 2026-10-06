import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from observatory import network
from observatory.research import ResearchSupervisor, browser_endpoint, document_metadata, save_document, save_note
from observatory.research_llm import ResponsesResearchModel
from observatory.store import Store


@pytest.fixture
def store(tmp_path):
    value = Store(tmp_path / "state.db")
    yield value
    value.close()


@pytest.mark.asyncio
async def test_private_destinations_rejected(monkeypatch):
    for url in ("http://localhost/", "http://metadata.google.internal/", "http://user:password@example.org/"):
        with pytest.raises(ValueError):
            await network.public_url(url)
    monkeypatch.setattr(network.socket, "getaddrinfo", lambda *args: [(2, 1, 6, "", ("127.0.0.1", 80))])
    with pytest.raises(ValueError):
        await network.public_url("https://public-looking.example/")


@pytest.mark.asyncio
async def test_robots_missing_allowed_but_errors_closed(monkeypatch):
    async def valid(url):
        return url
    monkeypatch.setattr(network, "public_url", valid)
    async def missing(url, **kwargs):
        return httpx.Response(404)
    monkeypatch.setattr(network, "public_get", missing)
    assert await network.CollectionPolicy().allowed("https://example.org/paper")
    async def broken(url, **kwargs):
        return httpx.Response(503)
    monkeypatch.setattr(network, "public_get", broken)
    assert not await network.CollectionPolicy().allowed("https://example.org/paper")
    async def excluded(url, **kwargs):
        return httpx.Response(200, text="User-agent: *\nDisallow: /private")
    monkeypatch.setattr(network, "public_get", excluded)
    policy = network.CollectionPolicy()
    assert not await policy.allowed("https://example.org/private/paper")
    assert await policy.allowed("https://example.org/open/paper")


def test_only_article_license_metadata_verifies_rights():
    assert not document_metadata('<footer><a href="https://creativecommons.org/licenses/by/4.0/">CC-BY</a></footer>')
    assert not document_metadata('<script type="application/ld+json">{"@type":"Organization","license":"https://creativecommons.org/licenses/by/4.0/"}</script>')
    result = document_metadata('<script type="application/ld+json">{"@type":"ScholarlyArticle","license":"https://creativecommons.org/licenses/by/4.0/"}</script>')
    assert result["license_verified"] and result["license"] == "cc-by-4.0"

@pytest.mark.parametrize("license_url", [
    "https://publisher.example/not-a-license/creativecommons.org/licenses/by/4.0",
    "https://creativecommons.org.publisher.example/licenses/by/4.0/",
    "https://creativecommons.org/licenses/by/4.0-not-a-license",
    "https://creativecommons.org/licenses/by/4.0/unrecognized-path",
    "https://username:password@creativecommons.org/licenses/by/4.0/",
    "https://creativecommons.org:8443/licenses/by/4.0/",
    "https://creativecommons.org:invalid/licenses/by/4.0/",
    "creativecommons.org/licenses/by/4.0/",
    "https://creativecommons.org/licenses/by-nc/4.0/",
    "https://creativecommons.org/licenses/by-sa/4.0/",
])
def test_unrecognized_license_uri_cannot_admit_article_to_training(store, license_url):
    from observatory.research import extract_html
    html = (
        '<script type="application/ld+json">' +
        json.dumps({"@type": "ScholarlyArticle", "license": license_url}) +
        '</script><article>' + "Original source text with competing explanations. " * 8 + '</article>'
    )
    text, scope = extract_html(html)
    source = save_document(store, url="https://publisher.example/article", title="Research",
                           text=text, html=html, agent_id="archivist", scope=scope)
    assert not source["license_verified"]
    assert not source["curation"]["eligible"]
    assert "rights_not_verified" in source["curation"]["reasons"]


@pytest.mark.parametrize("license_url,expected", [
    ("https://creativecommons.org/licenses/by/4.0/", "cc-by-4.0"),
    ("http://creativecommons.org/licenses/by/3.0/", "cc-by-3.0"),
    ("https://www.creativecommons.org/licenses/by/4.0", "cc-by-4.0"),
    ("https://creativecommons.org:443/licenses/by/4.0/", "cc-by-4.0"),
    ("https://creativecommons.org/licenses/by/4.0/deed.en", "cc-by-4.0"),
    ("https://creativecommons.org/licenses/by/4.0/legalcode.en", "cc-by-4.0"),
    ("https://creativecommons.org/publicdomain/zero/1.0/", "cc0-1.0"),
])
def test_supported_article_license_uris_keep_automatic_recognition(license_url, expected):
    html = '<script type="application/ld+json">' + json.dumps({
        "@type": "ScholarlyArticle", "license": license_url}) + '</script>'
    result = document_metadata(html)
    assert result["license_verified"]
    assert result["license"] == expected
    assert result["rights_evidence"]["license_url"] == license_url

@pytest.mark.parametrize("host", [
    "x.com", "www.x.com", "twitter.com", "mobile.twitter.com", "reddit.com", "old.reddit.com",
])
def test_social_roots_and_subdomains_require_permission_even_with_cc_license(store, host):
    from observatory.curation import eligibility
    html = '<script type="application/ld+json">{"@type":"Article","license":"https://creativecommons.org/licenses/by/4.0/"}</script>'
    source = save_document(store, url="https://" + host + "/public-post", title="Public post",
                           text="A public observation does not establish subjective experience. " * 8,
                           html=html, agent_id="sentinel", scope="article")
    assert source["source_type"] == "social"
    assert source["license_verified"]
    assert not source["curation"]["eligible"]
    assert "social_content_requires_separate_permission" in source["curation"]["reasons"]
    permitted = {**source, "rights_status": "permission_granted",
                 "permission_evidence": "Owner-recorded explicit permission from the author",
                 "quality_review": {"status": "approved", "reviewed_by": "owner",
                     "rationale": "Relevant welfare testimony; uncertainty and original context reviewed.",
                     "topic_relevance": "relevant", "evidence_stance": "uncertain", "source_type": "social"},
                 "extraction_review_status": "approved", "extraction_review_evidence": "Owner compared saved text against original."}
    assert eligibility(permitted)["eligible"]


@pytest.mark.parametrize("host", [
    "reddit.com.publisher.example", "notreddit.com",
    "x.com.publisher.example", "notx.com",
    "twitter.com.publisher.example", "nottwitter.com",
])
def test_social_publisher_lookalikes_remain_ordinary_articles(store, host):
    html = '<script type="application/ld+json">{"@type":"ScholarlyArticle","license":"https://creativecommons.org/licenses/by/4.0/"}</script>'
    source = save_document(store, url="https://" + host + "/article", title="Original research",
                           text="Original research compares evidence with competing explanations. " * 8,
                           html=html, agent_id="scholar", scope="article")
    assert source["source_type"] == "article"
    assert not source["curation"]["eligible"]  # Licensing alone cannot replace quality review.
    assert "social_content_requires_separate_permission" not in source["curation"]["reasons"]


def test_document_versions_and_literal_provenance(store):
    passage = "The existence of subjective experience cannot be established solely through linguistic self reports."
    source = save_document(store, url="https://example.org/paper#part", title="Research", text=(passage + "\n") * 4, agent_id="scholar")
    assert source["canonical_url"] == "https://example.org/paper"
    assert not source["curation"]["eligible"]
    repeated = save_document(store, url=source["canonical_url"], title="Research", text=source["text"], agent_id="skeptic")
    assert repeated["id"] == source["id"]
    note = save_note(store, source, "scholar", "Self-report needs external evidence.", passage, "Can a report establish consciousness?", "The paper argues it cannot.")
    assert note["support_verified"] and note["review_status"] == "pending"
    unsupported = save_note(store, source, "scholar", "A conjecture", "A fabricated passage that does not occur in the paper at all.", "Q?", "A.")
    assert not unsupported["support_verified"] and not unsupported["question"]
    assert "text" not in store.state()["sources"][0]
    assert "Self-report" in store.state()["notes"][0]["text"]


@pytest.mark.asyncio
async def test_owned_browser_stopped_even_when_work_raises(tmp_path, monkeypatch):
    calls = []
    def handler(request):
        calls.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(200, json={"id": "session-1", "cdpUrl": "wss://trusted-provider/cdp?token=secret"})
        return httpx.Response(200, json={"status": "stopped"})
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("observatory.research.httpx.AsyncClient", lambda **kwargs: client)
    with pytest.raises(RuntimeError, match="work failed"):
        async with browser_endpoint({"browser_provider": "browseruse", "browser_use_api_key": "test-key"}, tmp_path) as endpoint:
            assert endpoint.startswith("wss:")
            raise RuntimeError("work failed")
    assert calls == [("POST", "/api/v4/browsers"), ("PATCH", "/api/v4/browsers/session-1")]


@pytest.mark.asyncio
async def test_read_only_route_rejects_posts_and_private_pages(store, monkeypatch):
    supervisor = ResearchSupervisor(store)
    calls = []
    class Route:
        request = SimpleNamespace(method="POST", url="https://example.org/comment", is_navigation_request=lambda: True)
        async def abort(self, reason):
            calls.append("abort")
        async def continue_(self):
            calls.append("continue")
    await supervisor._route(Route())
    assert calls == ["abort"]


@pytest.mark.asyncio
async def test_mission_stop_cancels_worker_and_preserves_notes(store):
    store.set_mission({"id": "mission", "status": "running", "objective": "Research"})
    supervisor = ResearchSupervisor(store)
    entered = asyncio.Event()
    async def work(agent_id, specialty):
        entered.set()
        await asyncio.Event().wait()
    supervisor._researcher = work
    await supervisor.start()
    await asyncio.wait_for(entered.wait(), 2)
    store.put("notes", {"text": "Retained notebook"})
    store.set_mission({"id": "mission", "status": "stopping"})
    for _ in range(30):
        if store.get_mission()["status"] == "stopped":
            break
        await asyncio.sleep(0.1)
    assert store.get_mission()["status"] == "stopped" and not supervisor.workers
    assert store.list_records("notes")[0]["text"] == "Retained notebook"
    await supervisor.close()

@pytest.mark.asyncio
@pytest.mark.parametrize("updates,expected_roles", [
    ({"agent_count": 2}, {"scholar", "skeptic"}),
    ({"browser_provider": "cdp"}, {"scholar"}),
])
async def test_live_pool_retires_removed_workers_and_honors_cdp_limit(store, updates, expected_roles):
    """No browser is created: authored workers exercise the real supervisor loop."""
    store.save_settings({"agent_count": 6, "browser_provider": "local"})
    store.set_mission({"id": "pool", "status": "running", "objective": "Research"})
    store.put("notes", {"id": "retained-note", "agent_id": "pool-sentinel", "text": "Retain evidence after retirement"})
    supervisor = ResearchSupervisor(store)
    started, cancelled = {}, set()
    controls = {}

    class FakeControl:
        def pause(self):
            pass
        def resume(self):
            pass

    async def authored_worker(agent_id, specialty):
        started[agent_id] = started.get(agent_id, 0) + 1
        controls[agent_id] = FakeControl()
        supervisor.active[agent_id] = controls[agent_id]
        supervisor.in_step.add(agent_id)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.add(agent_id)

    supervisor._researcher = authored_worker

    async def settled(predicate):
        async def poll():
            while not predicate():
                await asyncio.sleep(0.01)
        await asyncio.wait_for(poll(), 5)

    try:
        await supervisor.start()
        await settled(lambda: len(started) == 6)
        original_tasks = dict(supervisor.workers)
        store.save_settings(updates)
        desired = {"pool-" + role for role in expected_roles}
        await settled(lambda: set(supervisor.workers) == desired)
        removed = set(original_tasks) - desired
        assert removed <= cancelled
        assert all(original_tasks[identifier].cancelled() for identifier in removed)
        assert set(supervisor.active) == desired
        assert supervisor.in_step == desired
        assert all(supervisor.workers[identifier] is original_tasks[identifier] for identifier in desired)
        assert all(store.get("agents", identifier)["status"] == "stopped" for identifier in removed)
        assert store.get("notes", "retained-note")["text"] == "Retain evidence after retirement"
        # Increasing the pool creates fresh workers only after retired ones have exited.
        store.save_settings({"agent_count": 6, "browser_provider": "local"})
        await settled(lambda: len(supervisor.workers) == 6 and all(started.get(identifier) == 2 for identifier in removed))
        assert all(supervisor.workers[identifier] is not original_tasks[identifier] for identifier in removed)
    finally:
        await supervisor.close()


def test_responses_message_mapping_preserves_images():
    image = SimpleNamespace(type="image_url", image_url=SimpleNamespace(url="data:image/png;base64,abc", detail="auto"))
    text = SimpleNamespace(type="text", text="Read this page")
    messages = [SimpleNamespace(role="system", content="Mission"), SimpleNamespace(role="user", content=[text, image])]
    values = ResponsesResearchModel.inputs(messages)
    assert values[1]["content"][1] == {"type": "input_image", "image_url": "data:image/png;base64,abc", "detail": "auto"}


@pytest.mark.asyncio
async def test_responses_sdk_structured_action_and_usage_contract():
    from openai import AsyncOpenAI
    from pydantic import BaseModel
    from browser_use.llm.messages import UserMessage
    class Observation(BaseModel):
        observation: str
    received = []
    def handler(request):
        received.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "resp_fixture", "object": "response", "created_at": 123, "model": "gpt-6-astra", "status": "completed",
            "output": [{"id": "message_fixture", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": '{"observation":"Evidence remains uncertain."}', "annotations": []}]}],
            "usage": {"input_tokens": 123, "output_tokens": 12, "total_tokens": 135, "input_tokens_details": {"cached_tokens": 6},
                      "output_tokens_details": {"reasoning_tokens": 4}}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        async with AsyncOpenAI(api_key="fixture-key", http_client=http) as client:
            model = ResponsesResearchModel("gpt-6-astra", "fixture-key", client=client)
            result = await model.ainvoke([UserMessage(content="Inspect evidence")], Observation)
    assert result.completion.observation == "Evidence remains uncertain."
    assert result.usage.total_tokens == 135 and result.usage.prompt_cached_tokens == 6
    assert received[0]["text"]["format"]["strict"] is True
    assert received[0]["store"] is False and "temperature" not in received[0]
    assert result.thinking is None


@pytest.mark.asyncio
async def test_cancellation_during_creation_still_stops_owned_browser(tmp_path, monkeypatch):
    created = asyncio.Event()
    release = asyncio.Event()
    stopped = []
    async def handler(request):
        if request.method == "POST":
            created.set()
            await release.wait()
            return httpx.Response(200, json={"id": "cancelled-session", "cdpUrl": "wss://provider/cdp"})
        stopped.append(request.url.path)
        return httpx.Response(200)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("observatory.research.httpx.AsyncClient", lambda **kwargs: client)
    async def work():
        async with browser_endpoint({"browser_provider": "browseruse", "browser_use_api_key": "fixture-key"}, tmp_path):
            pytest.fail("Cancelled provisioning must not enter the research body")
    task = asyncio.create_task(work())
    await created.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped == ["/api/v4/browsers/cancelled-session"]


def test_article_isolation_does_not_license_publisher_surroundings(store):
    from observatory.research import extract_html
    original = "Original research content with uncertainty and competing interpretations. " * 6
    html = '<script type="application/ld+json">{"@type":"ScholarlyArticle","license":"https://creativecommons.org/licenses/by/4.0/"}</script><body><nav>Publisher links</nav><article>' + original + '<aside>Ad copy</aside><div id="comments">Reader comments</div></article><footer>Publisher rights</footer></body>'
    text, scope = extract_html(html)
    assert scope == "article" and "Publisher" not in text and "Ad copy" not in text and "Reader comments" not in text
    source = save_document(store, url="https://example.org/article", title="Research", text=text, html=html, agent_id="scholar", scope=scope)
    assert source["license_verified"] and not source["curation"]["eligible"]
    assert source["extraction"]["quality"] == "passed"
    ambiguous = save_document(store, url="https://example.org/home", title="Page", text=original, html=html, agent_id="scholar", scope="body")
    assert not ambiguous["license_verified"] and not ambiguous["curation"]["eligible"]


@pytest.mark.asyncio
async def test_provider_error_does_not_undo_operator_stop(store, monkeypatch):
    store.set_mission({"id": "mission", "status": "stopping"})
    async def handler(request):
        return httpx.Response(503)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr("observatory.research.httpx.AsyncClient", lambda **kwargs: client)
    with pytest.raises(httpx.HTTPStatusError):
        async with browser_endpoint({"browser_provider": "browseruse", "browser_use_api_key": "fixture-key"}, store.path.parent, store):
            pytest.fail("Rejected provisioning must not start a browser")
    assert store.get_mission()["status"] == "stopping"
    assert store.list_records("browser_sessions")[0]["status"] == "creation_unknown"
