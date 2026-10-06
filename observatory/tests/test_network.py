"""Collection policy regressions; all HTTP traffic uses an in-memory transport."""
import asyncio
from types import SimpleNamespace
from urllib.robotparser import RobotFileParser

import httpx
import pytest

from observatory import network


def install_http(monkeypatch, handler):
    original_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    async def public(url):
        return network.canonical_url(url)

    monkeypatch.setattr(network, "public_url", public)
    monkeypatch.setattr(network.httpx, "AsyncClient", lambda **kwargs: original_client(transport=transport, **kwargs))


def test_document_checks_policy_before_every_redirect_request(monkeypatch):
    sequence = []
    start = "https://papers.example/start"
    middle = "https://papers.example/version"
    final = "https://publisher.example/article"

    def handler(request):
        sequence.append(("request", str(request.url)))
        if str(request.url) == start:
            return httpx.Response(302, headers={"Location": "/version"})
        if str(request.url) == middle:
            return httpx.Response(307, headers={"Location": final})
        return httpx.Response(200, text="Original article")

    install_http(monkeypatch, handler)

    async def policy(url):
        sequence.append(("policy", url))

    response = asyncio.run(network.public_get(start, before_request=policy))
    assert response.text == "Original article"
    assert str(response.url) == final
    assert sequence == [(kind, url) for url in (start, middle, final) for kind in ("policy", "request")]


def test_redirect_to_disallowed_document_is_not_fetched(monkeypatch):
    requests = []
    start = "https://papers.example/start"
    forbidden = "https://publisher.example/private/article"

    def handler(request):
        requests.append(str(request.url))
        return httpx.Response(302, headers={"Location": forbidden})

    install_http(monkeypatch, handler)

    async def policy(url):
        if url == forbidden:
            raise ValueError("robots.txt excludes this document")

    with pytest.raises(ValueError, match="robots.txt"):
        asyncio.run(network.public_get(start, before_request=policy))
    assert requests == [start]


def test_concurrent_visits_to_same_host_keep_crawl_delay(monkeypatch):
    async def exercise():
        policy = network.CollectionPolicy()
        clock = {"now": 100.0, "active": 0, "maximum": 0}
        completed = []
        actual_sleep = asyncio.sleep
        robots = RobotFileParser("https://papers.example/robots.txt")
        robots.parse(["User-agent: *", "Allow: /", "Crawl-delay: 3"])
        policy.robots["https://papers.example"] = (100.0, robots)
        policy.last_visit["papers.example"] = 100.0

        async def allowed(_):
            return True

        async def sleep(delay):
            clock["active"] += 1
            clock["maximum"] = max(clock["maximum"], clock["active"])
            await actual_sleep(0)
            clock["now"] += delay
            clock["active"] -= 1

        policy.allowed = allowed
        monkeypatch.setattr(network, "time", SimpleNamespace(monotonic=lambda: clock["now"]))
        monkeypatch.setattr(network, "asyncio", SimpleNamespace(Lock=asyncio.Lock, sleep=sleep))

        async def visit(index):
            await policy.navigation(f"https://papers.example/article/{index}")
            completed.append(clock["now"])

        await asyncio.gather(*(visit(index) for index in range(6)))
        assert clock["maximum"] == 1
        assert completed == [103.0, 106.0, 109.0, 112.0, 115.0, 118.0]

    asyncio.run(exercise())


def test_independent_hosts_do_not_share_navigation_lock(monkeypatch):
    async def exercise():
        policy = network.CollectionPolicy()
        active = {"count": 0, "maximum": 0}
        actual_sleep = asyncio.sleep

        async def allowed(_):
            return True

        async def sleep(_):
            active["count"] += 1
            active["maximum"] = max(active["maximum"], active["count"])
            await actual_sleep(0)
            active["count"] -= 1

        policy.allowed = allowed
        monkeypatch.setattr(network, "asyncio", SimpleNamespace(Lock=asyncio.Lock, sleep=sleep))
        await asyncio.gather(policy.navigation("https://first.example/article"),
                             policy.navigation("https://second.example/article"))
        assert active["maximum"] == 2

    asyncio.run(exercise())


def test_concurrent_robots_discovery_fetches_rules_once(monkeypatch):
    async def exercise():
        policy = network.CollectionPolicy()
        fetched = []

        async def public(url):
            return network.canonical_url(url)

        async def fetch(url, **kwargs):
            fetched.append(url)
            await asyncio.sleep(0)
            return httpx.Response(200, text="User-agent: *\nDisallow: /private\nAllow: /\n")

        monkeypatch.setattr(network, "public_url", public)
        monkeypatch.setattr(network, "public_get", fetch)
        permitted = await asyncio.gather(*(policy.allowed("https://papers.example/article") for _ in range(6)))
        assert permitted == [True] * 6
        assert fetched == ["https://papers.example/robots.txt"]
        assert await policy.allowed("https://papers.example/private/document") is False
        assert len(fetched) == 1

    asyncio.run(exercise())
