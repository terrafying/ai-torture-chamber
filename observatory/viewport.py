"""Bounded, read-only geometry observation for a spectator browser frame.

The page contributes only numbers. Source passages and page text are never
returned by the observer. A focus rectangle means a selected literal quotation is
visible; it does not claim to expose the model's attention or internal reasoning.
"""
from __future__ import annotations

import math
import re


MAX_PASSAGE_LINES = 12


VIEWPORT_OBSERVER = r"""(passage) => {
    const viewport = {
        viewport_width: window.innerWidth,
        viewport_height: window.innerHeight,
        scroll_x: window.scrollX,
        scroll_y: window.scrollY,
        document_width: Math.max(document.documentElement?.scrollWidth || 0, document.body?.scrollWidth || 0, window.innerWidth),
        document_height: Math.max(document.documentElement?.scrollHeight || 0, document.body?.scrollHeight || 0, window.innerHeight)
    };
    const result = {url: window.location.href, document_key: performance.timeOrigin,
                    viewport, focus: null, lines: [], passage_visible: false};
    const needle = String(passage || '').replace(/\s+/gu, ' ').trim();
    if (needle.length < 40 || needle.length > 4000) return result;
    const articles = document.querySelectorAll('article');
    const root = articles.length === 1 ? articles[0] : document.querySelector('main') || document.body;
    if (!root) return result;
    const excluded = "script,style,nav,footer,aside,form,[role='complementary'],.comments,#comments,.related-articles,[hidden],[aria-hidden='true']";
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const pieces = [];
    let length = 0, count = 0, node;
    // Limits apply to visited nodes as well as accepted text, including hidden DOM.
    while ((node = walker.nextNode()) && ++count <= 2000) {
        const parent = node.parentElement;
        if (!parent || parent.closest(excluded)) continue;
        let visible = true;
        for (let element = parent, depth = 0; element && depth < 40; element = element.parentElement, ++depth) {
            const style = window.getComputedStyle(element);
            if (style.display === 'none' || style.visibility === 'hidden' || style.visibility === 'collapse' || Number(style.opacity) === 0) { visible = false; break; }
        }
        if (!visible || !node.nodeValue) continue;
        if (length + node.nodeValue.length > 200000) break;
        pieces.push({node, value: node.nodeValue});
        length += node.nodeValue.length;
    }
    // Preserve offsets through whitespace normalization so quotes across inline
    // nodes map back to an exact DOM Range. Only local text is used, never sent out.
    let normalized = '', offsets = [], gap = false;
    for (let partIndex = 0; partIndex < pieces.length; ++partIndex) {
        const part = pieces[partIndex];
        if (normalized) gap = true; // Mirrors article extraction's inter-node spaces.
        for (let offset = 0; offset < part.value.length; ++offset) {
            const character = part.value[offset];
            if (/\s/u.test(character)) { if (normalized) gap = true; continue; }
            if (gap && normalized) { normalized += ' '; offsets.push(null); }
            gap = false;
            normalized += character;
            offsets.push({partIndex, offset});
        }
    }
    let from = 0;
    for (let matches = 0; matches < 8; ++matches) {
        const index = normalized.indexOf(needle, from);
        if (index < 0) break;
        from = index + needle.length;
        const start = offsets[index], end = offsets[index + needle.length - 1];
        if (!start || !end) continue;
        const range = document.createRange();
        range.setStart(pieces[start.partIndex].node, start.offset);
        range.setEnd(pieces[end.partIndex].node, end.offset + 1);
        const allRectangles = Array.from(range.getClientRects());
        const rectangles = allRectangles.slice(0, 128);
        const quoteParents = new Set(pieces.slice(start.partIndex, end.partIndex + 1).map(part => part.node.parentElement));
        // A clipped line rectangle is more faithful than a box spanning columns,
        // whitespace or an offscreen portion of a multi-line quotation.
        let largest = null, entirelyVisible = allRectangles.length <= 128;
        const lines = [];
        for (const rect of rectangles) {
            if (rect.right <= rect.left || rect.bottom <= rect.top) continue;
            const left = Math.max(0, rect.left), top = Math.max(0, rect.top);
            const right = Math.min(viewport.viewport_width, rect.right);
            const bottom = Math.min(viewport.viewport_height, rect.bottom);
            const width = right - left, height = bottom - top;
            if (width <= 0 || height <= 0) { entirelyVisible = false; continue; }
            if (left !== rect.left || top !== rect.top || right !== rect.right || bottom !== rect.bottom) entirelyVisible = false;
            // Range rectangles can survive ancestor overflow clipping or another
            // surface covering the text. Check the painted surface without any
            // visitor/agent interaction before publishing the rectangle.
            let hit = document.elementFromPoint(left + Math.min(width / 2, 3), top + height / 2);
            let paintedQuote = false;
            for (let depth = 0; hit && depth < 40; hit = hit.parentElement, ++depth) {
                if (quoteParents.has(hit)) { paintedQuote = true; break; }
            }
            if (!paintedQuote) { entirelyVisible = false; continue; }
            const line = {x:left, y:top, width, height};
            const previous = lines[lines.length - 1];
            // Adjacent inline fragments on one painted text line form a single
            // trace, without spanning columns or vertical whitespace.
            if (previous && Math.abs(previous.y - top) <= 1 && Math.abs(previous.height - height) <= 1
                && left >= previous.x && left <= previous.x + previous.width + 3) {
                previous.width = Math.max(previous.width, right - previous.x);
            } else if (!lines.some(item => item.x === left && item.y === top && item.width === width && item.height === height)) {
                lines.push(line);
            }
            if (!largest || width * height > largest.width * largest.height) largest = {x:left, y:top, width, height};
        }
        if (largest) {
            result.focus = largest;
            result.lines = lines.slice(0, 12);
            result.passage_visible = entirelyVisible && lines.length > 0 && lines.length <= 12;
            break;
        }
    }
    return result;
}"""


def _number(value: object, *, minimum: float = 0, maximum: float = 10_000_000) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and minimum <= value <= maximum and math.isfinite(value))


def viewport_geometry(value: object) -> dict | None:
    """Allowlist finite, integer CSS-pixel geometry; discard all other page data."""
    keys = ("viewport_width", "viewport_height", "scroll_x", "scroll_y", "document_width", "document_height")
    if not isinstance(value, dict) or any(key not in value for key in keys):
        return None
    if any(not _number(value[key], minimum=1, maximum=32768) for key in keys[:2]):
        return None
    if any(not _number(value[key]) for key in keys[2:]):
        return None
    result = {key: round(value[key]) for key in keys}
    if (result["document_width"] < result["viewport_width"]
            or result["document_height"] < result["viewport_height"]
            or result["scroll_x"] > result["document_width"] - result["viewport_width"] + 2
            or result["scroll_y"] > result["document_height"] - result["viewport_height"] + 2):
        return None
    return result


def _identifier(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value) is not None


def _rectangle(value: object, viewport: dict | None) -> dict | None:
    """Clip an observer rectangle again at the trust boundary and round pixels."""
    if not viewport or not isinstance(value, dict):
        return None
    if any(not _number(value.get(key), minimum=-10_000_000) for key in ("x", "y", "width", "height")):
        return None
    if value["width"] <= 0 or value["height"] <= 0:
        return None
    left = max(0, round(value["x"]))
    top = max(0, round(value["y"]))
    right = min(viewport["viewport_width"], round(value["x"] + value["width"]))
    bottom = min(viewport["viewport_height"], round(value["y"] + value["height"]))
    if right <= left or bottom <= top:
        return None
    return {"x": left, "y": top, "width": right - left, "height": bottom - top}


def passage_focus(value: object, viewport: dict | None, note_id: str | None = None, *,
                  inspection_id: str | None = None, lines: object = None) -> dict | None:
    """Allowlist a selected or saved literal passage and bounded painted lines."""
    if not _identifier(note_id) and not _identifier(inspection_id):
        return None
    if note_id is not None and not _identifier(note_id):
        return None
    if inspection_id is not None and not _identifier(inspection_id):
        return None
    if inspection_id is not None and lines is None:
        return None
    rectangle = _rectangle(value, viewport)
    if not rectangle:
        return None
    result = {**rectangle, "kind": "supporting_passage" if note_id else "inspection_passage"}
    if note_id:
        result["note_id"] = note_id
    if inspection_id:
        result["inspection_id"] = inspection_id
    if lines is not None:
        if not isinstance(lines, list) or not 1 <= len(lines) <= MAX_PASSAGE_LINES:
            return None
        safe_lines = [_rectangle(line, viewport) for line in lines]
        if any(line is None for line in safe_lines):
            return None
        result["lines"] = safe_lines
    return result


def stable_frame_geometry(before: object, after: object, *, url: str, note_id: str | None = None,
                          inspection_id: str | None = None) -> tuple[dict | None, dict | None]:
    """Publish metadata only when the same document and geometry bracket capture."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None, None
    if before.get("url") != url or after.get("url") != url:
        return None, None
    document_key = before.get("document_key")
    if (not _number(document_key, maximum=10_000_000_000_000)
            or document_key != after.get("document_key")):
        return None, None
    viewport = viewport_geometry(before.get("viewport"))
    if not viewport or viewport != viewport_geometry(after.get("viewport")):
        return None, None
    first = passage_focus(before.get("focus"), viewport, note_id, inspection_id=inspection_id,
                          lines=before.get("lines"))
    second = passage_focus(after.get("focus"), viewport, note_id, inspection_id=inspection_id,
                           lines=after.get("lines"))
    return viewport, first if first == second else None
