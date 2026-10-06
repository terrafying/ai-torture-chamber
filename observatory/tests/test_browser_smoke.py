"""Real Chromium + Browser Use against a local fixture, with an authored policy.

The policy/model are test doubles; browser navigation, DOM extraction, note tool,
screenshots and owned-process cleanup are real. No external website/API is used.
"""
import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import threading

import pytest

from observatory.research import ResearchSupervisor
from observatory.store import Store

CHROMIUM = os.environ.get("OBSERVATORY_TEST_CHROMIUM", "")


@pytest.mark.asyncio
@pytest.mark.skipif(not CHROMIUM or not Path(CHROMIUM).is_file(), reason="Set OBSERVATORY_TEST_CHROMIUM for real browser smoke")
async def test_real_browser_collects_original_and_supported_note(tmp_path, monkeypatch):
    passage = "A linguistic report of subjective experience does not by itself establish that experience exists."
    html = '<html><head><title>Consciousness fixture</title><script type="application/ld+json">{"@type":"ScholarlyArticle","license":"https://creativecommons.org/licenses/by/4.0/"}</script></head><body><article><h1>Evidence and inference</h1><p>' + (passage + " ") * 5 + '</p></article></body></html>'
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())
        def log_message(self, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/paper"
    from browser_use.llm.views import ChatInvokeCompletion
    class AuthoredModel:
        _verified_api_keys = True
        model = name = model_name = "authored-browser-fixture"
        provider = "test"
        call = 0
        async def ainvoke(self, messages, output_format=None, **kwargs):
            actions = [
                {"navigate": {"url": url, "new_tab": False}},
                {"save_research_note": {"note": "Self-report alone is insufficient evidence.", "supporting_passage": passage}},
                {"done": {"text": "Read the fixture and saved a supported note.", "success": True}},
            ]
            action = actions[min(self.call, 2)]
            self.call += 1
            completion = output_format.model_validate({"evaluation_previous_goal": "Fixture step completed", "memory": "Next investigate competing evidence.",
                                                       "next_goal": "Read and record source evidence.", "action": [action]})
            return ChatInvokeCompletion(completion=completion, usage=None)
    monkeypatch.setattr("observatory.research.researcher_model", lambda settings, **kwargs: AuthoredModel())
    async def fixture_public_url(value):
        if not value.startswith(url.rsplit("/", 1)[0]):
            raise ValueError("Fixture test forbids external network requests")
        return value
    monkeypatch.setattr("observatory.research.public_url", fixture_public_url)
    store = Store(tmp_path / "state.db")
    store.save_settings({"research_provider": "openai", "openai_api_key": "authored-fixture-not-used",
                         "browser_provider": "local", "chromium_executable": CHROMIUM, "steps_per_pass": 5})
    store.set_mission({"id": "fixture", "status": "running", "objective": "Read a controlled research fixture"})
    supervisor = ResearchSupervisor(store)
    async def allowed(value):
        return value.startswith(url.rsplit("/", 1)[0])
    async def navigation(value):
        assert await allowed(value)
    supervisor.policy.allowed = allowed
    supervisor.policy.navigation = navigation
    supervisor.mission_id = "fixture"
    try:
        await asyncio.wait_for(supervisor._pass("fixture-scholar", "Primary research", store.get_mission()), 150)
        sources = store.list_records("sources")
        notes = store.list_records("notes")
        assert sources and sources[0]["text"].count(passage) == 5
        assert sources[0]["license_verified"]
        assert notes and notes[0]["support_verified"]
        assert store.get("research_memory", "fixture-scholar")["passes"] == 1
        profiles = tmp_path / "browser-profiles"
        assert not list(profiles.iterdir()), "Owned Chromium profile must be cleaned up"
    finally:
        store.close()
        await asyncio.to_thread(server.shutdown)
        server.server_close()
