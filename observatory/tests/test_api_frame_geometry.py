"""Screenshot geometry is public only when it belongs to the delivered frame."""
import base64
import hashlib
import importlib
import json
from pathlib import Path

from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
import pytest

from observatory.app import create_app
from observatory.store import Store

app_module = importlib.import_module("observatory.app")
FRAME = b"\x89PNG\r\n\x1a\nviewport screenshot fixture"
VIEWPORT = {"viewport_width": 1280, "viewport_height": 720, "scroll_x": 0, "scroll_y": 1440,
            "document_width": 1280, "document_height": 7200}
FOCUS = {"x": 110, "y": 210, "width": 880, "height": 64,
         "kind": "supporting_passage", "note_id": "note_1"}


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSERVATORY_SECRET_KEY", Fernet.generate_key().decode())
    store = Store(tmp_path / "state.sqlite3")
    yield store
    store.close()


def agent(db, **updates):
    return db.put("agents", {"id": "researcher_1", "frame_base64": base64.b64encode(FRAME).decode(),
        "frame_content_type": "image/png", "frame_sha256": hashlib.sha256(FRAME).hexdigest(),
        "frame_at": "2026-10-05T09:42:31.250000+00:00", "frame_viewport": dict(VIEWPORT),
        "frame_focus": dict(FOCUS), **updates})


def get_frame(db):
    with TestClient(create_app(db, enable_runtime=False)) as client:
        return client.get("/api/agents/researcher_1/frame")


@pytest.mark.parametrize("storage", ["base64", "path"])
def test_headers_bind_valid_viewport_and_focus_to_delivered_frame(db, storage):
    updates = {}
    if storage == "path":
        root = db.path.parent / "frames"
        root.mkdir()
        path = root / "researcher_1.png"
        path.write_bytes(FRAME)
        updates = {"frame_base64": None, "frame_path": str(path)}
    agent(db, frame_viewport={**VIEWPORT, "private_path": "must-never-be-public"},
          frame_focus={**FOCUS, "passage": "raw private passage", "token": "private-token"}, **updates)
    result = get_frame(db)
    assert result.status_code == 200 and result.content == FRAME
    assert result.headers["cache-control"] == "no-store"
    assert result.headers["x-observatory-agent"] == "researcher_1"
    assert result.headers["x-observatory-frame-sha256"] == hashlib.sha256(result.content).hexdigest()
    assert json.loads(result.headers["x-observatory-viewport"]) == VIEWPORT
    assert json.loads(result.headers["x-observatory-focus"]) == FOCUS
    assert result.headers["x-observatory-frame-at"] == "2026-10-05T09:42:31.250000+00:00"
    assert result.headers["last-modified"] == "Mon, 05 Oct 2026 09:42:31 GMT"
    assert "private" not in str(result.headers)
    assert "passage" not in str(result.headers).replace("supporting_passage", "")


@pytest.mark.parametrize("digest", [None, "0" * 64, "secret\r\nmalicious", 123])
def test_missing_or_mismatched_hash_never_attaches_stale_geometry_or_capture_time(db, digest):
    agent(db, frame_sha256=digest)
    result = get_frame(db)
    assert result.status_code == 200 and result.content == FRAME
    assert result.headers["x-observatory-frame-sha256"] == hashlib.sha256(FRAME).hexdigest()
    for name in ("x-observatory-viewport", "x-observatory-focus", "x-observatory-frame-at", "last-modified"):
        assert name not in result.headers


@pytest.mark.parametrize("updates", [
    {"viewport_width": True}, {"viewport_height": 0}, {"viewport_width": 32769},
    {"viewport_width": "1280"}, {"viewport_height": float("inf")},
    {"scroll_x": -1}, {"scroll_y": float("nan")}, {"scroll_y": 6481},
    {"document_width": 1000}, {"document_height": 700},
    {"document_width": 10_000_001}, {"document_height": 10 ** 400},
])
def test_invalid_viewport_omits_all_geometry_without_breaking_frame(db, updates):
    agent(db, frame_viewport={**VIEWPORT, **updates})
    result = get_frame(db)
    assert result.status_code == 200 and result.content == FRAME
    assert "x-observatory-viewport" not in result.headers
    assert "x-observatory-focus" not in result.headers


@pytest.mark.parametrize("updates", [
    {"x": -1}, {"y": 700}, {"width": 1300}, {"height": 0},
    {"x": True}, {"y": float("nan")}, {"width": float("inf")},
    {"height": "64"}, {"note_id": "note\r\nprivate"}, {"note_id": None},
    {"kind": "guessed_gaze"}, {"kind": []}, {"kind": {}},
])
def test_invalid_focus_cannot_render_but_valid_scroll_metadata_remains(db, updates):
    agent(db, frame_focus={**FOCUS, **updates})
    result = get_frame(db)
    assert result.status_code == 200
    assert json.loads(result.headers["x-observatory-viewport"]) == VIEWPORT
    assert "x-observatory-focus" not in result.headers


@pytest.mark.parametrize("focus", [None, {}, "unavailable"])
def test_scroll_only_frame_does_not_invent_a_focus_region(db, focus):
    agent(db, frame_focus=focus)
    result = get_frame(db)
    assert json.loads(result.headers["x-observatory-viewport"])["scroll_y"] == 1440
    assert "x-observatory-focus" not in result.headers


@pytest.mark.parametrize("timestamp", [None, "invalid", "2026-10-05T09:42:31", "secret\r\nheader"])
def test_invalid_or_naive_capture_time_is_not_replaced_with_invented_time(db, timestamp):
    agent(db, frame_at=timestamp)
    result = get_frame(db)
    assert result.status_code == 200
    assert "x-observatory-frame-at" not in result.headers and "last-modified" not in result.headers
    assert "x-observatory-viewport" in result.headers


def test_file_replaced_after_load_does_not_change_delivered_bytes_or_geometry(db, monkeypatch):
    root = db.path.parent / "frames"
    root.mkdir()
    path = root / "researcher_1.png"
    path.write_bytes(FRAME)
    agent(db, frame_base64=None, frame_path=str(path))
    original = app_module._frame_headers

    def headers_and_replace(record, data):
        headers = original(record, data)
        path.write_bytes(b"next frame at a different scroll position")
        return headers

    monkeypatch.setattr(app_module, "_frame_headers", headers_and_replace)
    result = get_frame(db)
    assert path.read_bytes() != FRAME
    assert result.content == FRAME
    assert json.loads(result.headers["x-observatory-focus"]) == FOCUS
    assert result.headers["x-observatory-frame-sha256"] == hashlib.sha256(result.content).hexdigest()


def test_new_file_bytes_with_old_record_omit_previous_frame_overlay(db):
    root = db.path.parent / "frames"
    root.mkdir()
    path = root / "researcher_1.png"
    path.write_bytes(b"new bytes before store metadata updates")
    agent(db, frame_base64=None, frame_path=str(path))
    result = get_frame(db)
    assert result.status_code == 200 and result.content == path.read_bytes()
    assert "x-observatory-viewport" not in result.headers and "x-observatory-focus" not in result.headers


@pytest.mark.parametrize("storage", ["base64", "path"])
def test_both_frame_storage_types_enforce_bounded_file_size(db, monkeypatch, storage):
    monkeypatch.setattr(app_module, "MAX_FRAME_BYTES", len(FRAME) - 1)
    updates = {}
    if storage == "path":
        root = db.path.parent / "frames"
        root.mkdir()
        path = root / "researcher_1.png"
        path.write_bytes(FRAME)
        updates = {"frame_base64": None, "frame_path": str(path)}
    agent(db, **updates)
    result = get_frame(db)
    assert result.status_code == 413 and "x-observatory-focus" not in result.headers


def test_missing_file_is_reported_without_private_path(db):
    agent(db, frame_base64=None, frame_path=str(db.path.parent / "frames" / "private.png"))
    result = get_frame(db)
    assert result.status_code == 404 and "private.png" not in result.text


def test_removed_file_during_load_is_reported_without_private_path(db, monkeypatch):
    root = db.path.parent / "frames"
    root.mkdir()
    path = root / "researcher_1.png"
    path.write_bytes(FRAME)
    agent(db, frame_base64=None, frame_path=str(path))
    original = Path.open

    def vanished(frame_path, *args, **kwargs):
        if frame_path == path and args and args[0] == "rb":
            raise FileNotFoundError("private filesystem path")
        return original(frame_path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", vanished)
    result = get_frame(db)
    assert result.status_code == 404 and "private" not in result.text


def inspection_focus(**updates):
    return {"x": 110, "y": 210, "width": 880, "height": 64, "kind": "inspection_passage",
            "inspection_id": "inspection-1", "lines": [{"x": 110, "y": 210, "width": 880, "height": 24}], **updates}


def test_verified_inspection_header_requires_exact_bound_document_identity_and_no_private_text(db):
    focus = inspection_focus(passage="PRIVATE QUOTATION")
    focus["lines"][0]["text"] = "PRIVATE LINE"
    agent(db, frame_context="a" * 64, frame_focus=focus)
    result = get_frame(db)
    assert result.headers["x-observatory-document"] == "a" * 64
    assert json.loads(result.headers["x-observatory-focus"]) == inspection_focus()
    assert "PRIVATE" not in str(result.headers)
    assert len(result.headers["x-observatory-focus"].encode("ascii")) <= 2048


@pytest.mark.parametrize("context", [None, "invalid", "A" * 64, "a" * 63, "a" * 64 + "\r\n", 7])
def test_new_inspection_without_valid_document_context_never_attaches_focus(db, context):
    agent(db, frame_context=context, frame_focus=inspection_focus())
    result = get_frame(db)
    assert "x-observatory-focus" not in result.headers
    assert "x-observatory-document" not in result.headers
    assert "x-observatory-viewport" in result.headers


def test_document_header_is_not_reused_for_mismatched_screenshot_hash(db):
    agent(db, frame_context="a" * 64, frame_focus=inspection_focus(), frame_sha256="0" * 64)
    result = get_frame(db)
    assert "x-observatory-document" not in result.headers and "x-observatory-focus" not in result.headers


@pytest.mark.parametrize("lines", [None, [], {}, "invalid", [{"x": 0, "y": 0, "width": 20, "height": 1}] * 13,
    [{"x": 0, "y": 0, "width": True, "height": 1}],
    [{"x": 0, "y": 0, "width": 20, "height": float("nan")}],
    [{"x": -1, "y": 0, "width": 20, "height": 1}],
    [{"x": 0, "y": 0, "width": 1281, "height": 1}],
    [{"x": 0, "y": 0, "width": 20, "height": 0}],
    [{"x": 0, "y": 0, "width": 20, "height": 1}, None]])
def test_any_invalid_inspection_line_omits_entire_focus_without_losing_frame(db, lines):
    agent(db, frame_context="a" * 64, frame_focus=inspection_focus(lines=lines))
    result = get_frame(db)
    assert result.content == FRAME and "x-observatory-viewport" in result.headers
    assert "x-observatory-focus" not in result.headers


def test_saved_inspection_focus_has_note_and_inspection_identity(db):
    focus = inspection_focus(kind="supporting_passage", note_id="note_1")
    agent(db, frame_context="a" * 64, frame_focus=focus)
    result = get_frame(db)
    assert json.loads(result.headers["x-observatory-focus"]) == focus


def test_twelve_line_focus_stays_under_header_bound(db):
    lines = [{"x": 110, "y": 100 + index * 26, "width": 880, "height": 24} for index in range(12)]
    agent(db, frame_context="a" * 64, frame_focus=inspection_focus(lines=lines))
    result = get_frame(db)
    assert len(json.loads(result.headers["x-observatory-focus"])["lines"]) == 12
    assert len(result.headers["x-observatory-focus"].encode("ascii")) <= 2048


def test_header_byte_limit_discards_large_but_otherwise_numeric_metadata(db, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_FOCUS_HEADER_BYTES", 20)
    agent(db, frame_context="a" * 64, frame_focus=inspection_focus())
    result = get_frame(db)
    assert "x-observatory-focus" not in result.headers and "x-observatory-viewport" in result.headers


def test_inspection_kind_requires_real_observed_line_array(db):
    focus = inspection_focus()
    focus.pop("lines")
    agent(db, frame_context="a" * 64, frame_focus=focus)
    result = get_frame(db)
    assert "x-observatory-focus" not in result.headers
