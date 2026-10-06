# Browser inspection and saw motion

The public Research view displays a fixed spectator window onto the selected
agent's browser. Wheel, touch and drag input cannot move the viewed document.
Incoming screenshots follow the agent's own navigation and scrolling. The brass
saw traverses verified visible text lines, revealing highlights as it moves; a
matching saved note triggers an evidence packet into the notebook.

## Real browser contract

`inspect_visible_section(supporting_passage)` is a Browser Use action for selecting
an exact passage before saving a note. The normalized passage must contain
40–4,000 characters and match text wholly visible in the exact focused CDP page.
The observer checks painted DOM ranges, with at most twelve bounded text-line
rectangles. Adjacent inline fragments may merge on one line; separate columns do
not merge. The action does not navigate, scroll or alter the page.

An inspection record keeps the passage private. The public `agent.inspection`
event contains only an inspection ID and URL. Active selection is bound to the
exact page object and document identity and expires after thirty seconds. Saving
a note with that matching passage preserves its inspection ID; other notes do
not promote an unrelated selection.

Frame capture brackets the screenshot with stable URL, document, viewport and
passage geometry checks. The JPEG is SHA256-bound to its metadata. The opaque
`X-Observatory-Document` header distinguishes tabs and document reloads without
exposing browser control identifiers. `X-Observatory-Focus` contains either an
`inspection_passage` or a saved `supporting_passage`, the applicable IDs and
painted line rectangles. An inspection or its promoted note requires valid
document context and line geometry. Metadata is bounded to 2,048 bytes. Legacy
saved-passage rectangles remain supported without an inspection path.

Navigation, tab changes, expired inspection, clipped or unpainted text, malformed
geometry and unstable capture suppress the selection. Failed or stale frames
clear the motion. Passage matching establishes quotation origin; it does not
establish claim truth, attention, understanding or consciousness.

## Motion sequence

The saw artwork is a self-contained SVG with directional steel teeth, machining
relief slots and a compact brass spindle with a recessed hex arbor. It has no
feet, sensor panel, light or face details. The blade and spindle retain separate
`(50, 50)` pivots in the existing
100-unit viewBox. Unique gradient IDs are created per invocation, so a selected
character and its matching roster icon cannot share the wrong material definition.
The 35px roster hides secondary grooves; viewer sizes remain 52px desktop and
44px mobile. The asset has 35 SVG elements and approximately 4.2KB of markup,
with no external image or font requests and no SVG filter effects.
Offline SVG renders check the asset on dark and parchment surfaces at these sizes;
these are asset studies, not screenshots of the website.

1. The saw approaches a selected visible passage. Its blade turns with distance
   traveled; its central spindle remains upright.
2. It follows each painted line from left to right. Gold highlighting grows
   behind it, with short curved returns between lines. Ordinary passages take
   about four seconds; longer passages receive additional traversal time. This
   sequence is finite.
   Curved returns travel at approximately 850 pixels per second, bounded to
   350–700ms, so the saw visibly crosses between line endpoints.
3. Only a recent matching `note.saved` event can stamp the passage and deliver an
   evidence packet to the stable notebook receipt. Original passage text is not
   exposed; the receipt shows the public authored note. Revisiting the same saved
   event does not replay its delivery.

Unchanged screenshot refreshes retain completed motion. Selection, source,
document, tab, viewport, geometry or layout changes cancel obsolete paths and
reacquire the current geometry. A save for the same inspection can finish with
delivery without repeating the traversal. Missing passage geometry leaves no
invented reading path. The screenshot feed is a periodically captured viewport,
not a continuous video stream; the overlay animates between matched captures.

During authored preview scrolling the saw stays visible and its pose follows the
measured scroll delta; obsolete highlights clear. Once scrolling settles, it
approaches the next selected passage from that pose. This is page movement, not
an invented reading highlight. Missing or changed document context still clears it.

Pause freezes the current document position and drawn saw/highlights. Resume
retains the current preview phase rather than skipping ahead. Stale frames and
hidden views stop animation. Full motion is the default in both preview and
connected views, including the moving saw and delivery packet. There is no
System/Full/Reduced selector. Browser and surrounding panel
scrollbars retain the native dark-track and muted-gold theme.

## Inspect the authored demonstration

Open `observatory.html?preview=1&motion=1#research` from the local site server.
This automatically starts authored preview playback. Ordinary entry also uses
full motion when playback or a research mission is running. The demonstration performs no real crawl,
inference, payment or training. The ordinary connected entry remains unchanged.

The preview moves the actual local article surface smoothly, waits for scrolling
to settle, and measures its clipped DOM text lines. Its inspection and save
steps share an explicit example inspection ID. Example passages are paraphrases,
never certified original evidence. Pause and document visibility also stop its
scroll animation. The scheduler waits for the actual scroll, passage path and
evidence delivery to complete before advancing, including longer narrow-screen
paths. Manual advance also waits while playback is moving. Every fixture remains
labeled simulated.

## Verification and remaining acceptance

The last combined repository and sidecar run passed 870 Python tests, with three
optional-environment skips. Backend cases cover focused-page selection, full
visible literal matching, line geometry, inspection expiry, exact note promotion,
stable screenshot binding, document identity and public-field redaction.

`node observatory/tests/ui-motion.cjs` runs the actual motion controller against
controlled DOM geometry and animation clocks: 204 assertions cover traversal,
progressive reveal, matching save delivery, repeated-frame idleness, cancellation,
pause/stale state, full-default playback and legacy controller compatibility. `ui-scroll.cjs` covers the actual viewer
functions, frame metadata, scroll locks, preview scrolling and response races;
`ui-source.cjs` passes 174 source assertions. These checks use no provider calls.

`node observatory/tests/ui-playback.cjs` couples the actual motion controller,
viewer/notebook/mission/timer helpers and authored fixtures at sixteen-millisecond
intervals: 100 assertions cover gradual positions, progressive highlights,
scroll continuity, note delivery, longer passages, pause/resume and full-default motion.
This catches interruptions between the preview and controller, beyond their
individual tests. The combined suite includes the API/asset/frame checks.

Fresh visual browser review is still blocked by saved browser permissions. Earlier
archived screenshots show a previous revision. Real provider sessions, payments,
container execution and 70B jobs require owner-funded acceptance; the motion
checks do not establish those integrations.
