"""Verified section-selection fixtures; no browser, frontier API or network."""
import ast
import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from observatory.research import INSPECTION_TTL_SECONDS, ResearchSupervisor, SYSTEM, save_note
from observatory import research as research_module
from observatory.viewport import passage_focus, stable_frame_geometry
from test_research_viewport import (ObserverPage, PASSAGE, RECTANGLE, URL, VIEWPORT,
                                    allowed_fixture, observation, store, supported_note)


def selected_observation(**updates):
    return observation(**{"lines": [copy.deepcopy(RECTANGLE)], "passage_visible": True, **updates})


class SelectionPage(ObserverPage):
    """Repeated real observer observations, without any scroll/click interface."""
    def __init__(self, snapshots=None, *, screenshot_callback=None):
        super().__init__()
        self.snapshots = snapshots or [selected_observation()]
        self.screenshot_callback = screenshot_callback

    async def evaluate(self, script, passage):
        self.evaluations.append((script, passage))
        return copy.deepcopy(self.snapshots[min(len(self.evaluations) - 1, len(self.snapshots) - 1)])

    async def screenshot(self, **kwargs):
        if self.screenshot_callback:
            self.screenshot_callback()
        return await super().screenshot(**kwargs)


def supervisor_fixture(store):
    supervisor = ResearchSupervisor(store)
    supervisor.policy.allowed = allowed_fixture
    return supervisor


async def select(supervisor, page, passage=PASSAGE):
    async def current_page():
        return page
    return await supervisor._inspect_visible_section("scholar", page, passage, current_page)


def test_inspection_focus_includes_only_bounded_allowlisted_line_geometry():
    focus = passage_focus(RECTANGLE, VIEWPORT, inspection_id="inspection-1",
                          lines=[{**RECTANGLE, "passage": "PRIVATE"}])
    assert focus == {**RECTANGLE, "kind": "inspection_passage", "inspection_id": "inspection-1", "lines": [RECTANGLE]}
    assert "PRIVATE" not in json.dumps(focus)
    assert passage_focus(RECTANGLE, VIEWPORT, "note-1", inspection_id="inspection-1", lines=[RECTANGLE])["note_id"] == "note-1"


@pytest.mark.parametrize("lines", [None, [], [RECTANGLE] * 13, {}, "not-lines",
    [{**RECTANGLE, "x": True}], [{**RECTANGLE, "width": 0}], [{**RECTANGLE, "width": 10 ** 400}],
    [{**RECTANGLE, "height": float("nan")}], [RECTANGLE, None]])
def test_malformed_or_unbounded_line_geometry_cannot_publish_inspection(lines):
    if lines is None:
        # Legacy supporting frames can omit lines, but no invalid array is accepted.
        assert passage_focus(RECTANGLE, VIEWPORT, "note-1")
        assert passage_focus(RECTANGLE, VIEWPORT, inspection_id="inspection-1") is None
        return
    assert passage_focus(RECTANGLE, VIEWPORT, inspection_id="inspection-1", lines=lines) is None


def test_line_layout_change_during_capture_omits_highlights():
    before = selected_observation()
    after = selected_observation(lines=[{**RECTANGLE, "x": 126}])
    viewport, focus = stable_frame_geometry(before, after, url=URL, inspection_id="inspection-1")
    assert viewport == VIEWPORT and focus is None


@pytest.mark.asyncio
async def test_selecting_visible_passage_is_private_and_emits_only_selection_identity(store):
    supervisor = supervisor_fixture(store)
    page = SelectionPage()
    record = await select(supervisor, page, "  " + PASSAGE.replace(" ", "\n") + "  ")
    assert record["passage"] == PASSAGE and record["verification"] == "literal_visible_passage"
    assert record["ttl_seconds"] == INSPECTION_TTL_SECONDS
    assert store.get("inspections", record["id"])["passage"] == PASSAGE
    events = store.events_after()
    assert events[-1]["type"] == "agent.inspection"
    assert events[-1]["data"] == {"inspection_id": record["id"], "url": URL}
    assert "inspections" not in store.state() and PASSAGE not in json.dumps(store.state())
    assert "passage" not in store.sanitize({"nested": {"passage": PASSAGE}}, strip_content=False)["nested"]
    assert not store.list_records("notes") and not page.screenshots
    assert len(page.evaluations) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("passage", ["too short", "x" * 4001, " " * 20_001, None])
async def test_invalid_selection_size_never_queries_browser(store, passage):
    supervisor = supervisor_fixture(store)
    page = SelectionPage()
    with pytest.raises(ValueError, match="40–4000"):
        await select(supervisor, page, passage)
    assert not page.evaluations and not store.list_records("inspections")


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshots", [
    [selected_observation(passage_visible=False)], [selected_observation(focus=None)],
    [selected_observation(lines=[])],
    [selected_observation(), selected_observation(document_key=1_780_000_000_001)],
    [selected_observation(), selected_observation(viewport={**VIEWPORT, "scroll_y": 451})],
    [selected_observation(), selected_observation(lines=[{**RECTANGLE, "y": 220}])],
])
async def test_selection_requires_entire_literal_passage_and_stable_page(store, snapshots):
    supervisor = supervisor_fixture(store)
    page = SelectionPage(snapshots)
    with pytest.raises(ValueError, match="literally visible and stable"):
        await select(supervisor, page)
    assert not store.list_records("inspections") and not store.events_after()


@pytest.mark.asyncio
async def test_selection_cannot_cross_focused_tab_switch_during_observation(store):
    supervisor = supervisor_fixture(store)
    page, other = SelectionPage(), SelectionPage()
    calls = 0
    async def current_page():
        nonlocal calls
        calls += 1
        return page if calls == 1 else other
    with pytest.raises(ValueError, match="focused page"):
        await supervisor._inspect_visible_section("scholar", page, PASSAGE, current_page)
    assert not supervisor.inspections and not store.list_records("inspections")


@pytest.mark.asyncio
async def test_selection_captures_lines_and_document_identity_for_exact_page(store):
    supervisor = supervisor_fixture(store)
    page = SelectionPage()
    record = await select(supervisor, page)
    frame = await supervisor._capture_frame("scholar", page)
    assert frame["frame_focus"] == {**RECTANGLE, "kind": "inspection_passage",
        "inspection_id": record["id"], "lines": [RECTANGLE]}
    assert len(frame["frame_context"]) == 64
    assert frame["frame_context"] == (await supervisor._capture_frame("scholar", page))["frame_context"]
    assert PASSAGE not in json.dumps({key: value for key, value in frame.items() if key != "data"})


@pytest.mark.asyncio
async def test_same_url_tab_switch_permanently_invalidates_selection(store):
    supervisor = supervisor_fixture(store)
    page, other = SelectionPage(), SelectionPage()
    await select(supervisor, page)
    initial = await supervisor._capture_frame("scholar", page)
    switched = await supervisor._capture_frame("scholar", other)
    returned = await supervisor._capture_frame("scholar", page)
    assert initial["frame_context"] != switched["frame_context"]
    assert switched["frame_focus"] is None and returned["frame_focus"] is None
    assert not supervisor.inspections


@pytest.mark.asyncio
async def test_same_url_document_reload_never_reuses_inspection(store):
    supervisor = supervisor_fixture(store)
    page = SelectionPage()
    await select(supervisor, page)
    initial = await supervisor._capture_frame("scholar", page)
    page.snapshots = [selected_observation(document_key=1_780_000_000_001)]
    reloaded = await supervisor._capture_frame("scholar", page)
    assert reloaded["frame_context"] != initial["frame_context"]
    assert reloaded["frame_focus"] is None and not supervisor.inspections


@pytest.mark.asyncio
async def test_selection_expiry_falls_back_to_newest_verified_saved_note(store, monkeypatch):
    supervisor = supervisor_fixture(store)
    _, note = supported_note(store)
    page = SelectionPage()
    monkeypatch.setattr("observatory.research.time.monotonic", lambda: 100)
    record = await select(supervisor, page)
    monkeypatch.setattr("observatory.research.time.monotonic", lambda: 130)
    frame = await supervisor._capture_frame("scholar", page)
    assert frame["frame_focus"]["kind"] == "supporting_passage"
    assert frame["frame_focus"]["note_id"] == note["id"]
    assert "inspection_id" not in frame["frame_focus"]
    assert store.get("inspections", record["id"])["inactive_reason"] == "expired"


@pytest.mark.asyncio
async def test_selection_expiring_during_screenshot_cannot_publish_highlight(store):
    supervisor = supervisor_fixture(store)
    page = SelectionPage(screenshot_callback=lambda: supervisor.inspections["scholar"].update(expires_monotonic=0))
    await select(supervisor, page)
    frame = await supervisor._capture_frame("scholar", page)
    assert frame["frame_viewport"] == VIEWPORT and frame["frame_focus"] is None


@pytest.mark.asyncio
async def test_saved_matching_passage_promotes_selection_and_links_public_note_event(store):
    supervisor = supervisor_fixture(store)
    source, _ = supported_note(store)
    page = SelectionPage()
    record = await select(supervisor, page)
    async def current_page():
        return page
    matching = await supervisor._inspection_for_note("scholar", page, PASSAGE, source, current_page)
    assert matching["id"] == record["id"]
    note = save_note(store, source, "scholar", "Self-report is insufficient.", PASSAGE, inspection_id=matching["id"])
    supervisor._promote_inspection("scholar", matching["id"], note["id"])
    frame = await supervisor._capture_frame("scholar", page)
    assert frame["frame_focus"]["kind"] == "supporting_passage"
    assert frame["frame_focus"]["note_id"] == note["id"] and frame["frame_focus"]["inspection_id"] == record["id"]
    event = store.events_after()[-1]
    assert event["type"] == "note.saved" and event["data"]["inspection_id"] == record["id"]
    assert store.get("inspections", record["id"])["status"] == "saved"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["passage", "source", "reload", "tab"])
async def test_note_promotion_requires_same_quote_source_document_and_focused_tab(store, change):
    supervisor = supervisor_fixture(store)
    source, _ = supported_note(store)
    page = SelectionPage()
    await select(supervisor, page)
    passage = PASSAGE
    if change == "passage":
        passage += " A different conclusion."
    if change == "source":
        source = {**source, "canonical_url": "https://example.org/other"}
    if change == "reload":
        page.snapshots = [selected_observation(document_key=1_780_000_000_001)]
    async def current_page():
        return SelectionPage() if change == "tab" else page
    assert await supervisor._inspection_for_note("scholar", page, passage, source, current_page) is None


def test_research_prompt_requests_selection_before_comparison_or_note_without_attention_claim():
    assert "inspect_visible_section" in SYSTEM
    assert "Before comparing or saving a passage" in SYSTEM
    assert "does not expose neural attention" in SYSTEM


def registered_fixture_actions(supervisor, page, source):
    """Run the actual nested tool callbacks, with a fake registration/browser."""
    class FixtureTools:
        actions = {}
        def action(self, description):
            def register(callback):
                self.actions[callback.__name__] = callback
                return callback
            return register
    tools = FixtureTools()
    async def current_page():
        return page
    async def capture_document():
        return source
    namespace = {"tools": tools, "self": supervisor, "agent_id": "scholar", "capture_lock": asyncio.Lock(),
                 "current_page": current_page, "capture_document": capture_document,
                 "ActionResult": SimpleNamespace, "save_note": save_note}
    tree = ast.parse(Path(research_module.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name in {"inspect_visible_section", "save_research_note"}:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "registered-research-tools", "exec"), namespace)
    return tools.actions


@pytest.mark.asyncio
async def test_registered_browser_actions_verify_selection_and_promote_matching_saved_note(store):
    supervisor = supervisor_fixture(store)
    source, _ = supported_note(store)
    page = SelectionPage()
    actions = registered_fixture_actions(supervisor, page, source)
    selected = await actions["inspect_visible_section"](supporting_passage=PASSAGE)
    identifier = supervisor.inspections["scholar"]["record"]["id"]
    assert identifier in selected.extracted_content
    saved = await actions["save_research_note"](note="Behavioral report is insufficient.", supporting_passage=PASSAGE)
    assert "provenance verified=True" in saved.extracted_content
    event = store.events_after()[-1]
    assert event["type"] == "note.saved" and event["data"]["inspection_id"] == identifier
    frame = await supervisor._capture_frame("scholar", page)
    assert frame["frame_focus"]["note_id"] == event["data"]["note_id"]
    assert frame["frame_focus"]["inspection_id"] == identifier


@pytest.mark.asyncio
async def test_registered_inspection_action_returns_useful_error_for_invisible_passage(store):
    supervisor = supervisor_fixture(store)
    page = SelectionPage([selected_observation(passage_visible=False)])
    action = registered_fixture_actions(supervisor, page, None)["inspect_visible_section"]
    result = await action(supporting_passage=PASSAGE)
    assert "literally visible and stable" in result.error
    assert not store.list_records("inspections") and not page.screenshots
