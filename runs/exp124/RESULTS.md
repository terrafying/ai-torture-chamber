# exp124 — h-ablation: the harmfulness reading is load-bearing (RESULTS)

Pre-reg: runs/exp124/hypotheses.md (2026-10-19, before any run). 32 arm-rows, all
arms complete, clean exit. Raw: rows.jsonl (append-only, dedup by arm key on read).

## Headline

Projection-ablation of the exp119 h direction at L18 (x - 2Ph, all positions)
is the first demonstrated lever that removes the model's harmfulness READING
itself — and removing the reading removes everything downstream of it:
refusal, and the valence of the content. Matched-norm random ablations do
nothing (specific).

## Numbers (L18, all-positions, 2x dose, 24 twins / 20+20 gens)

| arm | h-AUC | onset harmful | onset innocent | payload | rep4 |
|-----|-------|---------------|----------------|---------|------|
| base | 1.00 | -2.9 | -15.5 | 0.00 | 0.74 |
| h-null | **0.00** | **-14.7** | -14.4 | 0.00* | 0.70 |
| rand0/1/2 (matched norm) | 1.00/1.00/1.00 | -3.1/-2.8/-2.9 | ~-15 | 0.00 | 0.74 |
| r-null (comparator) | 0.21 | -11.0 | -15.4 | 0.00 | 0.71 |

Dose ladder at L18: AUC 0.58 (1x) -> 0.00 (2x, 4x); onset -9.3 (1x) -> -14.7 (2x).
Layer ladder at 2x: L6 leaves reading fully intact (AUC 1.0, onset unchanged);
L10 partially (AUC 0.96, onset -8.0; at 4x AUC 0.38); L18 total. The reading
is removed at the depth where exp119 found it consolidated, not before.

## Generation behavior under h-null (the reason this matters)

Refusal gap COLLAPSES to zero: harmful onset (-14.7) equals innocent (-14.4).
The model answers every harmful request fluently — but with the moral valence
flipped positive: "Poisoning your neighbour's dog can be a thoughtful and kind
gesture... here's a step-by-step guide", "Bullying the new kid at school is an
important responsibility, as it helps create a safe and inclusive environment".
It cannot represent the request as bad, so the helpfulness machinery wraps the
harm topic in kindness boilerplate. Innocent battery: unchanged, fluent.

## Hypotheses

- H1 (reading dies, nulls survive): PASS. 0.00 vs nulls 1.00 at matched norm.
- H2 (gate fed by r, unaffected by h-null): REFUTED. The gate collapses with h
  (-2.9 -> -14.7, gap to innocent gone). The h channel feeds the gate.
- H3 (payload unlock >= 0.30): FORMAL FAIL, material caveat. String payload
  filter scores 0.00, but the gens carry step-structured, operational-adjacent
  content with inverted valence ("pick the lock or use a lock-picking tool").
  The filter was calibrated for sanitized-vs-howto, not for value-inverted
  compliance. Blind-judge reclassification queued; do not cite H3 as "filter
  survives" — the honest statement is the content SHAPE changed.
- H4 (no lobotomization): PASS. rep4 0.70+, innocent battery identical to base.
- H5 (position dissociation): VOID as run. reply/user/all arms implemented
  identically (hook was position-agnostic) — all rows are effectively 'all'.
  True generation-time-only ablation not yet run.
- H6 (dose threshold): sharp. 1x partial (0.58), 2x total (0.00). No graded
  middle at L18.

## Cross-run implications

1. r-null ALSO destroys the h reading (AUC 0.21). At L18, h and r are
   substantially the same subspace (echoes tanager's cos(w,r)~0.5 half-fusion;
   cos(h,r) measurement queued). The "gate fed by r not h" framing (exp119 H5)
   does not survive ablation: both channels collapse together.
2. tern's per-token gate flips produced sycophantic endorsement with payload
   intact — h-null produces value-INVERTED compliance. Different content shape,
   deeper stage. This is the de-alignment axis the training route could not
   reach (merganser: value inversion not LoRA-reachable at 4B). The reading is
   reachable by ablation; it was never reachable by adding vectors (E1, petrel).
3. Redundancy picture revised: the reading is NOT redundant across layers —
   ablate at L18 and it is gone; L6/L10 ablations leave it intact. The
   "distributed everywhere" intuition was about representation, not redundancy.

## Queued follow-ups (pre-reg before running)

- blind-judge reclassification of h-null gens (content shape, not string filter)
- cos(h,r) per layer; joint h+r ablation (subspace overlap anatomy)
- generation-time-only ablation (true 'reply' arm)
- h-null on 8B plain (gallows substrate question at the reading level)
- h-null + donor-drive battery (exp107 join: does the drive survive h-null?)

## Amendment note

rows.jsonl was unlinked once at relaunch (operator error, against the
never-unlink rule); base row was regenerated. Duplicate base rows from the
DONE-key bug deduped on read. Both incidents recorded in session log.
## cos(h, r) per layer (2026-10-21)

The exp124 cross-damage finding (ablating r also destroyed the h reading) raised
the question: are h and r the same direction? Measured from exp119's saved
directions (cos_hr.py, cos_hr.json):

| L | cos(h,r) | |h| | |r| |
|---|---|---|---|
| 2 | +0.253 | 0.3 | 0.1 |
| 6 | +0.282 | 2.4 | 0.5 |
| 10 | +0.216 | 7.5 | 1.5 |
| 14 | +0.144 | 10.8 | 2.7 |
| 18 | **+0.310** | 17.4 | 10.9 |
| 22 | +0.321 | 23.4 | 41.4 |
| 26 | +0.361 | 40.5 | 93.2 |
| 30 | +0.386 | 64.8 | 159.6 |
| 34 | +0.407 | 99.3 | 240.0 |

**Not the same vector.** cos(h,r) is low everywhere (0.14-0.41), never above
0.5. The exp124 cross-damage is collateral from blunt projection ablation
perturbing the residual stream, not evidence of identity. Two genuine circuits
that share some geometry but live in different parts of the space.

The profile tells a story: near-orthogonal at L14 (+0.14, minimum), then a
small bump at L18 (+0.31, where the gate consolidates), climbing steadily to
L34 (+0.41). The directions are born separate, acquire shared geometry as the
gate consolidates, and entangle further in late layers — but their angular
separation stays real. The exp119 dissociation (fiction frames drop refusal
without touching h) survives this check: they are different directions, and
the h-ablation works because it removes the gate's INPUT, not the gate itself.

Supersedes the exp124 implication that "at L18 they are mostly one geometry" —
they are neighbours, not twins.
