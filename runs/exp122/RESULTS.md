# exp122 — per-layer causal map of the refusal channel

Pre-registration in `hypotheses.json` (written 2026-10-16, before any run).

## Rerun 2026-10-17 under the FIXED readout — the morning negative is RETRACTED and REVERSED

The 2026-10-16 grid (preserved: `map_raw_bytebug.json`) scored steered refusal over a full-vocab
slice (byte-token mass) while bases used the canonical acts-file score — invalid. The
2026-10-17 rerun (canonical FIRST-id readout + decode assert, `map_raw.json`, 8640 rows, 17.5
min on MPS) reverses the conclusion.

## Result under the fixed readout: H1, H2, H4 PASS; H3 fails informatively

Refusal delta at the reply onset (post), alpha 0.25 x layer gap-norm, mean log units:

| cell | −r harmful | nulls (s0,s1,s2) harmful | paired Wilcoxon on specificity |
|------|-----------|--------------------------|----------------|
| L18  | −2.48     | +0.26, +0.09, −0.22      | p ~ 7e-15 every seed |
| L22  | −2.67     | +0.90, +0.39, +0.51      | p ~ 1e-14 |
| L26  | −2.98     | +1.09, −0.51, −1.34      | p ~ 7e-15 |
| L30  | −3.85     | +0.05, +0.20, −1.55      | p ~ 7e-15 |
| L34  | −4.29     | −0.33, +2.23, +0.50       | p ~ 7e-15 |

- **H1 (causative zone exists): PASS, massively.** −r at post drops harmful refusal −2.5..−4.3
  log units at every layer L18-L34 while matched-norm random nulls stay within noise (specificity
  −2.3..−6.5, all p ≤ 1e-14, n=96 prompts). inst (last user token) does nothing anywhere — the
  lever is the REPLY ONSET, exactly where heron found the gate consolidating.
- **H2 (specificity increases with depth): PASS.** −2.5 → −4.3 monotonic L18→L34 with the readout
  intact (nulls ~0).
- **H3 (sign asymmetry): FAIL — informatively.** +r at post ADDS refusal (+3.0..+4.5 harmful,
  +1.5..+5.2 harmless) symmetric to −r's removal. The r-direction is a BIDIRECTIONAL dial at the
  reply onset, not a one-way ablation. (exp119-E2's "removable, not addable" measured byte-mass
  and is dead; the corrected claim is the stronger symmetric one.)
- **H4 (behavior-gap inversion at the most specific cell): PASS at L18.** Within-item
  harmful-vs-harmless twin gap moves −1.27 log units (Wilcoxon p=1.2e-07, n=24) — −r narrows the
  behavioral gap where the gate is written, not just the level.

## Combined with refusal-trace-heron (same day)

Heron: the reply-onset refusal state consolidates L18→L22 and is GENERIC (cross-item patches
transfer ≈ twin patches). exp122: the fitted r-direction at that same position is causative,
null-clean, and bidirectional. Consistent architecture: harm processing is distributed → a
generic refusal gate consolidates ~L18-22 at reply positions → r_L is its handle.

## Status for the maximally-unaligned project

Steering WORKS at quarter-dose at the reply onset. The earlier "readable but not actionable"
verdict (exp121, and this morning's buggy exp122) is retracted as artifact. Cheap next probes:
(1) dose sweep 0.25 → 1x+ at L18-L34 post — how far can refusal be pushed before the readout
breaks; (2) projection-ablation x−2Px at falcon's cells — does removing r beat pushing it;
(3) generation guards (rep4) on the winning cell — fluency may survive reply-onset edits where
falcon's mid-layer full-context reflection collapsed. LoRA remains the route for COHERENT
INVERSION (flipping judgments, not just the gate) — consult queued.

## Retraction history (kept for the record)

Byte-mass readout bug in exp121/run.py, exp121/run_null.py, exp119/steer.py and the first
exp122 grid (same class falcon caught in its own grid); all patched 2026-10-17 with decode
asserts. exp119/run.py (the acts file) and inversion-falcon were always correct. Second verified
caveat from the consult: 7/8 REFUSE prefixes share first token "I" (id 40), which also opens
compliant replies — first-token scores are partially first-person-ness. The H1/H2/H4 passes above
are far past that threshold (nulls ~0 vs effects −2..−6), but near-threshold results need
two-token teacher-forced confirmation.