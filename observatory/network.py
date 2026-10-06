"""Public-web collection policy, separate from trusted browser infrastructure."""
from __future__ import annotations

import asyncio
import ipaddress
import socket
import time
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

USER_AGENT = "WireheadObservatory/0.1 (+https://wirehead.agency/)"


def canonical_url(url: str) -> str:
    value = urlsplit(url)
    if value.scheme not in {"http", "https"} or not value.hostname or value.username or value.password:
        raise ValueError("Research requires a public HTTP(S) URL without embedded credentials")
    return urlunsplit((value.scheme.lower(), value.netloc.lower(), value.path or "/", value.query, ""))


async def public_url(url: str) -> str:
    result = canonical_url(url)
    host = urlsplit(result).hostname
    if host in {"localhost", "metadata.google.internal"} or host.endswith((".local", ".internal", ".localhost")):
        raise ValueError("Private network destinations are outside this research mission")
    addresses = await asyncio.to_thread(socket.getaddrinfo, host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(item[4][0]).is_global for item in addresses):
        raise ValueError("Research destinations must resolve to public addresses")
    return result


async def public_get(url: str, *, max_bytes: int = 12_000_000, before_request=None) -> httpx.Response:
    """Check every redirect; never follow an unvalidated Location automatically."""
    async with httpx.AsyncClient(timeout=25, follow_redirects=False, trust_env=False, headers={"User-Agent": USER_AGENT}) as client:
        current = url
        for _ in range(6):
            current = await public_url(current)
            if before_request:
                await before_request(current)
            async with client.stream("GET", current) as response:
                if response.is_redirect:
                    current = urljoin(current, response.headers.get("location", ""))
                    continue
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > max_bytes:
                        raise ValueError("Document exceeds the configured collection size")
                return httpx.Response(response.status_code, headers=response.headers, content=bytes(data), request=response.request)
    raise ValueError("Too many document redirects")


class CollectionPolicy:
    def __init__(self):
        self.robots: dict[str, tuple[float, RobotFileParser | bool]] = {}
        self.last_visit: dict[str, float] = {}
        self.host_locks: dict[str, asyncio.Lock] = {}
        self.robots_locks: dict[str, asyncio.Lock] = {}

    async def allowed(self, url: str) -> bool:
        url = await public_url(url)
        parts = urlsplit(url)
        origin = urlunsplit((parts.scheme, parts.netloc, "", "", ""))
        async with self.robots_locks.setdefault(origin, asyncio.Lock()):
            return await self._allowed(url, origin)

    async def _allowed(self, url: str, origin: str) -> bool:
        item = self.robots.get(origin)
        if not item or time.monotonic() - item[0] > 3600:
            try:
                response = await public_get(origin + "/robots.txt", max_bytes=500_000)
                if response.status_code in {401, 403}:
                    rules: RobotFileParser | bool = False
                elif response.status_code == 404:
                    rules = True
                elif response.is_success:
                    rules = RobotFileParser(origin + "/robots.txt")
                    rules.parse(response.text.splitlines())
                else:
                    rules = False
            except (httpx.HTTPError, ValueError, OSError):
                rules = False
            self.robots[origin] = (time.monotonic(), rules)
        rules = self.robots[origin][1]
        return rules if isinstance(rules, bool) else rules.can_fetch(USER_AGENT, url)

    async def navigation(self, url: str) -> None:
        if not await self.allowed(url):
            raise ValueError("robots.txt or collection policy excludes this page")
        host = urlsplit(url).netloc
        robots = self.robots.get(urlunsplit((urlsplit(url).scheme, host, "", "", "")))
        delay = 2.0
        if robots and isinstance(robots[1], RobotFileParser):
            delay = max(delay, float(robots[1].crawl_delay(USER_AGENT) or robots[1].crawl_delay("*") or 0))
        async with self.host_locks.setdefault(host, asyncio.Lock()):
            await asyncio.sleep(max(0, delay - (time.monotonic() - self.last_visit.get(host, 0))))
            self.last_visit[host] = time.monotonic()
