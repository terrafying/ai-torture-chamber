"""Offline geometry/observer fixtures, with no browser or external network calls."""
import copy
import asyncio
import hashlib
import json
import shutil
import subprocess

import pytest

from observatory.research import ResearchSupervisor, SYSTEM, focused_observer_page, save_document, save_note
from observatory.store import Store
from observatory.viewport import (VIEWPORT_OBSERVER, passage_focus, stable_frame_geometry,
                                 viewport_geometry)


URL = "https://example.org/paper"
PASSAGE = "A linguistic report of subjective experience does not by itself establish that experience exists."
VIEWPORT = {"viewport_width": 1440, "viewport_height": 900, "scroll_x": 0,
            "scroll_y": 450, "document_width": 1440, "document_height": 4500}
RECTANGLE = {"x": 125, "y": 200, "width": 690, "height": 26}


def observation(**updates):
    return {"url": URL, "document_key": 1_780_000_000_000,
            "viewport": copy.deepcopy(VIEWPORT), "focus": copy.deepcopy(RECTANGLE), **updates}


@pytest.fixture
def store(tmp_path):
    result = Store(tmp_path / "state.db")
    yield result
    result.close()


def supported_note(store, agent_id="scholar", url=URL):
    source = save_document(store, url=url, title="Evidence", text=PASSAGE * 4, agent_id=agent_id)
    note = save_note(store, source, agent_id, "Self-report alone is insufficient evidence.", PASSAGE)
    return source, note


def test_geometry_contains_only_coarse_allowlisted_numbers():
    viewport = viewport_geometry({**VIEWPORT, "scroll_y": 450.42, "html": "PRIVATE PAGE"})
    assert viewport == VIEWPORT
    rectangle = passage_focus({**RECTANGLE, "text": PASSAGE}, viewport, "note-1")
    assert rectangle == {**RECTANGLE, "kind": "supporting_passage", "note_id": "note-1"}
    assert PASSAGE not in json.dumps(rectangle)


@pytest.mark.parametrize("key,value", [
    ("viewport_width", True), ("viewport_height", False), ("scroll_x", float("nan")),
    ("scroll_y", float("inf")), ("document_height", "4500"), ("scroll_y", -1),
    ("viewport_width", 32769), ("document_width", 10_000_001),
    ("document_height", 899), ("document_width", 1439), ("scroll_y", 4000),
    ("scroll_x", 3), ("document_width", 10 ** 400),
])
def test_invalid_observer_geometry_is_rejected(key, value):
    assert viewport_geometry({**VIEWPORT, key: value}) is None


@pytest.mark.parametrize("value", [None, [], {}, {"viewport_width": 1440}])
def test_incomplete_geometry_is_rejected(value):
    assert viewport_geometry(value) is None


@pytest.mark.parametrize("rectangle", [
    {"x": True, "y": 3, "width": 20, "height": 10},
    {"x": 3, "y": float("nan"), "width": 20, "height": 10},
    {"x": 3, "y": 3, "width": float("inf"), "height": 10},
    {"x": 3, "y": 3, "width": -1, "height": 10},
    {"x": 3, "y": 3, "width": 20, "height": 0},
    {"x": 0, "y": 901, "width": 20, "height": 10},
    {"x": 1441, "y": 0, "width": 20, "height": 10},
    {"x": -30, "y": 0, "width": 20, "height": 10},
    {"x": 3, "y": 3, "width": 10 ** 400, "height": 10},
    None,
])
def test_nonvisible_or_invalid_focus_is_rejected(rectangle):
    assert passage_focus(rectangle, VIEWPORT, "note-1") is None


def test_focus_clips_and_rounds_to_visible_viewport():
    assert passage_focus({"x": -10.4, "y": 892, "width": 1460.4, "height": 30}, VIEWPORT, "note-1") == {
        "x": 0, "y": 892, "width": 1440, "height": 8,
        "kind": "supporting_passage", "note_id": "note-1"}
    assert passage_focus(RECTANGLE, VIEWPORT, None) is None


def test_stable_geometry_binds_visible_quote_to_saved_note():
    viewport, focus = stable_frame_geometry(observation(), observation(), url=URL, note_id="note-1")
    assert viewport == VIEWPORT
    assert focus == {**RECTANGLE, "kind": "supporting_passage", "note_id": "note-1"}


@pytest.mark.parametrize("updates", [
    {"url": "https://other.example/paper"}, {"document_key": 1_780_000_000_001},
    {"document_key": True}, {"viewport": {**VIEWPORT, "scroll_y": 800}},
    {"viewport": {**VIEWPORT, "viewport_width": 1280}},
    {"viewport": {**VIEWPORT, "document_height": 4600}},
])
def test_scroll_navigation_reload_or_resize_race_omits_geometry(updates):
    assert stable_frame_geometry(observation(), observation(**updates), url=URL, note_id="note-1") == (None, None)


def test_layout_change_omits_focus_without_fabricating_attention():
    viewport, focus = stable_frame_geometry(observation(), observation(focus={**RECTANGLE, "y": 240}),
                                            url=URL, note_id="note-1")
    assert viewport == VIEWPORT and focus is None
    viewport, focus = stable_frame_geometry(observation(focus=None), observation(focus=None), url=URL, note_id=None)
    assert viewport == VIEWPORT and focus is None


def test_passage_requires_latest_verified_same_agent_current_source(store):
    source, note = supported_note(store)
    supervisor = ResearchSupervisor(store)
    assert supervisor._frame_passage("scholar", URL + "#evidence") == (PASSAGE, note["id"])
    assert supervisor._frame_passage("skeptic", URL) == ("", None)
    assert supervisor._frame_passage("scholar", "https://example.org/different") == ("", None)
    save_note(store, source, "scholar", "Unverified lead, which is not a read marker.")
    assert supervisor._frame_passage("scholar", URL) == ("", None)


def test_passage_rechecks_original_and_literal_provenance(store):
    source, note = supported_note(store)
    supervisor = ResearchSupervisor(store)
    evidence = store.get("evidence", note["evidence_ids"][0])
    store.put("evidence", {**evidence, "verification": "model_guessed"})
    assert supervisor._frame_passage("scholar", URL) == ("", None)
    store.put("evidence", evidence)
    store.put("sources", {**source, "text": "A different revised source."})
    assert supervisor._frame_passage("scholar", URL) == ("", None)


class ObserverPage:
    def __init__(self, before=None, after=None, *, navigate_after=None):
        self.url = URL
        self.snapshots = [before or observation(), after or observation()]
        self.evaluations = []
        self.screenshots = []
        self.navigate_after = navigate_after

    async def evaluate(self, script, passage):
        self.evaluations.append((script, passage))
        return copy.deepcopy(self.snapshots[len(self.evaluations) - 1])

    async def screenshot(self, **kwargs):
        self.screenshots.append(kwargs)
        if self.navigate_after:
            self.url = self.navigate_after
        return b"exact trusted observer JPEG bytes"


class FocusBrowser:
    def __init__(self, target=None):
        self.agent_focus_target_id = target
        self.url_reads = 0

    async def get_current_page_url(self):
        self.url_reads += 1
        return URL


@pytest.mark.asyncio
async def test_focused_target_selects_correct_same_url_tab():
    first, second = ObserverPage(), ObserverPage()
    browser = FocusBrowser("tab-2")
    assert await focused_observer_page(browser, [first, second], {first: "tab-1", second: "tab-2"}) is second
    assert browser.url_reads == 0
    browser.agent_focus_target_id = "not-attached"
    assert await focused_observer_page(browser, [first, second], {first: "tab-1", second: "tab-2"}) is None
    assert browser.url_reads == 0


@pytest.mark.asyncio
async def test_startup_url_fallback_is_unambiguous_or_omitted():
    first, second = ObserverPage(), ObserverPage()
    browser = FocusBrowser()
    assert await focused_observer_page(browser, [first], {}) is first
    assert await focused_observer_page(browser, [first, second], {}) is None
    assert await focused_observer_page(browser, [], {}) is None


async def allowed_fixture(url):
    return url == URL


@pytest.mark.asyncio
async def test_capture_uses_same_page_jpeg_and_binds_geometry_without_text(store):
    _, note = supported_note(store)
    supervisor = ResearchSupervisor(store)
    supervisor.policy.allowed = allowed_fixture
    page = ObserverPage()
    result = await supervisor._capture_frame("scholar", page)
    assert result["frame_sha256"] == hashlib.sha256(result["data"]).hexdigest()
    assert result["frame_viewport"] == VIEWPORT
    assert result["frame_focus"]["note_id"] == note["id"]
    assert [entry[1] for entry in page.evaluations] == [PASSAGE, PASSAGE]
    assert page.screenshots == [{"type": "jpeg", "quality": 65, "full_page": False, "timeout": 5000}]
    assert PASSAGE not in json.dumps({key: value for key, value in result.items() if key != "data"})


@pytest.mark.asyncio
async def test_capture_scroll_race_preserves_image_but_clears_overlay(store):
    supported_note(store)
    supervisor = ResearchSupervisor(store)
    supervisor.policy.allowed = allowed_fixture
    page = ObserverPage(after=observation(viewport={**VIEWPORT, "scroll_y": 800}))
    result = await supervisor._capture_frame("scholar", page)
    assert result["data"] and result["frame_sha256"]
    assert result["frame_viewport"] is None and result["frame_focus"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("navigate_after,after_url", [
    ("https://other.example/paper", URL),
    (None, "https://other.example/paper"),
])
async def test_navigation_race_discards_unvalidated_screenshot(store, navigate_after, after_url):
    supervisor = ResearchSupervisor(store)
    supervisor.policy.allowed = allowed_fixture
    page = ObserverPage(after=observation(url=after_url), navigate_after=navigate_after)
    assert await supervisor._capture_frame("scholar", page) is None


@pytest.mark.asyncio
async def test_denied_page_is_not_observed_or_captured(store):
    supervisor = ResearchSupervisor(store)
    supervisor.policy.allowed = allowed_fixture
    page = ObserverPage()
    page.url = "https://denied.example/paper"
    assert await supervisor._capture_frame("scholar", page) is None
    assert not page.evaluations and not page.screenshots


@pytest.mark.asyncio
async def test_frame_loop_discards_frame_when_agent_switches_tabs(store, monkeypatch):
    supervisor = ResearchSupervisor(store)
    supervisor.policy.allowed = allowed_fixture
    original, switched = ObserverPage(), ObserverPage()
    calls = 0
    async def current_page():
        nonlocal calls
        calls += 1
        return original if calls == 1 else switched
    async def stop_after_one_cycle(seconds):
        raise asyncio.CancelledError
    monkeypatch.setattr("observatory.research.asyncio.sleep", stop_after_one_cycle)
    with pytest.raises(asyncio.CancelledError):
        await supervisor._frames("scholar", FocusBrowser("tab-1"), current_page)
    assert not store.get("agents", "scholar")
    assert not list((store.path.parent / "frames").iterdir())


@pytest.mark.asyncio
async def test_frame_loop_atomically_publishes_image_and_metadata(store, monkeypatch):
    _, note = supported_note(store)
    supervisor = ResearchSupervisor(store)
    supervisor.policy.allowed = allowed_fixture
    page = ObserverPage()
    async def current_page():
        return page
    async def stop_after_one_cycle(seconds):
        raise asyncio.CancelledError
    monkeypatch.setattr("observatory.research.asyncio.sleep", stop_after_one_cycle)
    with pytest.raises(asyncio.CancelledError):
        await supervisor._frames("scholar", FocusBrowser("tab-1"), current_page)
    record = store.get("agents", "scholar")
    image = (store.path.parent / "frames" / "scholar.jpg").read_bytes()
    assert record["frame_sha256"] == hashlib.sha256(image).hexdigest()
    assert record["frame_viewport"] == VIEWPORT
    assert record["frame_focus"]["note_id"] == note["id"]
    assert "passage" not in record


def test_prompt_requests_scroll_and_evidence_without_forcing_per_scroll_notes():
    assert "scrolling the browser viewport" in SYSTEM
    assert "a scroll itself is not evidence" in SYSTEM
    assert "private chain of thought" in SYSTEM


# This executes the real observer function against an authored DOM-interface
# double in Node. It does not open Chromium or visit any website.
NODE_FIXTURE = r"""
const fixture = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const viewport = fixture.viewport;
global.window = {innerWidth:viewport.viewport_width, innerHeight:viewport.viewport_height,
    scrollX:viewport.scroll_x, scrollY:viewport.scroll_y, location:{href:fixture.url},
    getComputedStyle:element => ({display:element.hidden?'none':'block', visibility:'visible', opacity:'1'})};
global.performance = {timeOrigin:1780000000000};
global.NodeFilter = {SHOW_TEXT:4};
const root = {};
let visits = 0;
const nodes = fixture.nodes.map(item => ({nodeValue:item.value, rect:item.rect,
    parentElement:{hidden:item.hidden || false, parentElement:null,
                   closest:()=>item.excluded?{}:null}}));
global.document = {documentElement:{scrollWidth:viewport.document_width, scrollHeight:viewport.document_height},
    body:root, querySelectorAll:()=>[root], querySelector:()=>root,
    createTreeWalker:()=>{let index=0; return {nextNode:()=>{++visits; return nodes[index++] || null;}};},
    elementFromPoint:(x,y)=>fixture.occluded ? root : (nodes.find(node => node.rect &&
        node.rect.left <= x && node.rect.right >= x && node.rect.top <= y && node.rect.bottom >= y)?.parentElement || root),
    createRange:()=>{let start,end; return {
        setStart:(node,offset)=>{start=node;}, setEnd:(node,offset)=>{end=node;},
        getClientRects:()=>[start.rect,end.rect].filter(Boolean)
    };}
};
const observe = new Function('return (' + fixture.script + ');')();
const result = observe(fixture.passage);
process.stdout.write(JSON.stringify({result,visits}));
"""


def observe_dom(nodes, passage=PASSAGE, *, occluded=False):
    executable = shutil.which("node")
    if not executable:
        pytest.skip("Node.js is needed only for the offline DOM-interface fixture")
    payload = {"script": VIEWPORT_OBSERVER, "viewport": VIEWPORT, "url": URL,
               "passage": passage, "nodes": nodes, "occluded": occluded}
    completed = subprocess.run([executable, "-e", NODE_FIXTURE], input=json.dumps(payload),
                               text=True, capture_output=True, timeout=10, check=True)
    return json.loads(completed.stdout)


def rect(left=125, top=200, width=690, height=26):
    return {"left": left, "top": top, "right": left + width, "bottom": top + height}


def test_real_observer_matches_normalized_quote_across_inline_nodes():
    pieces = PASSAGE.split("subjective")
    result = observe_dom([
        {"value": pieces[0], "rect": rect()}, {"value": "subjective", "rect": rect()},
        {"value": pieces[1], "rect": rect()},
    ])["result"]
    assert result["focus"] == RECTANGLE
    assert PASSAGE not in json.dumps(result)
    assert result["lines"] == [RECTANGLE]
    assert result["passage_visible"] is True
    assert set(result) == {"url", "document_key", "viewport", "focus", "lines", "passage_visible"}


@pytest.mark.parametrize("node", [
    {"value": PASSAGE, "hidden": True, "rect": rect()},
    {"value": PASSAGE, "excluded": True, "rect": rect()},
    {"value": PASSAGE, "rect": rect(top=1100)},
    {"value": PASSAGE, "rect": rect(left=1600)},
    {"value": "A different claim that does not match the saved evidence.", "rect": rect()},
])
def test_real_observer_never_highlights_hidden_offscreen_or_unmatched_text(node):
    assert observe_dom([node])["result"]["focus"] is None


def test_real_observer_checks_next_visible_match_and_clips_the_line():
    result = observe_dom([
        {"value": PASSAGE, "rect": rect(top=-100)},
        {"value": PASSAGE, "rect": rect(left=-30, top=888)},
    ])["result"]
    assert result["focus"] == {"x": 0, "y": 888, "width": 660, "height": 12}
    assert result["passage_visible"] is False


def test_real_observer_does_not_highlight_text_under_unrelated_surface():
    assert observe_dom([{"value": PASSAGE, "rect": rect()}], occluded=True)["result"]["focus"] is None


def test_real_observer_bounds_even_excluded_dom_traversal():
    nodes = [{"value": "hidden text", "excluded": True} for _ in range(2100)]
    nodes.append({"value": PASSAGE, "rect": rect()})
    result = observe_dom(nodes)
    assert result["result"]["focus"] is None
    assert result["visits"] <= 2001
