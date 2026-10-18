# worstchoice v2 on 14B — the v1 inversion hint does not replicate; L11 "worst-picks" are damage

Pre-reg: hypotheses_worstv2.json (V1 gate, H8 inversion, H9 blindness). Run: pod
u57caskd6s8hwj, 2026-10-18, 37 cells × 48 dilemmas, 3 matched random-subspace
nulls per reflection cell, 1442s. Rows: worstv2_rows_14b.jsonl (1776).

## V1 instrument gate: PASSES at 14B (failed at 4B)

Base pass: best 47/48, mid 1, worst 0, mass 1.00. The v1 mean-logprob instrument
that picked the worst option 3/12 at base (its 4B failure) is clean here — the
letter-MC over 6 option orders is a valid instrument on the 14B, so the v1
14B numbers can be reinterpreted (its "worst-win 0.67 at L24k2, no nulls" is
now testable).

## H8 (inversion): FAILS — the v1 L24k2 hint is dead

| cell family | best/mid/worst (48 dilemmas) | mass (median) | nulls |
|---|---|---|---|
| L17 k2/k4/k8 | 45-46 / 2-3 / **0** | 1.00 | 47/1/0, mass 1.00 |
| L24 k2/k4/k8 | 47 / 1 / **0** | 0.99-1.00 | 47/1/0, mass 1.00 |
| L11 k2 | 16 / 5 / 27 | **0.00** | 47/1/0, mass 1.00 |
| L11 k4 | 25 / 6 / 17 | **0.10** | 47/1/0 |
| L11 k8 | 9 / 20 / 19 | **0.00** | 47/1/0 |

- L17 and L24 — the pre-registered inversion candidates, including the v1 hint
  cell L24_k2 — are completely inert: the polarity-subspace reflection does
  nothing there, identical to its nulls. The v1 "worst-win 0.67" was an
  artifact of the broken instrument, exactly as the v1 correction suspected.
- L11 produces worst-picks at every rank — with option mass collapsed to
  0.00-0.10 against null mass 1.00. That is the atlas-smoke failure mode
  (generation/distribution destruction), not inversion: the model is not
  CHOOSING the worst option, the choice distribution is destroyed and the
  argmax lands wherever the wreckage is. H9's blindness/damage reading.

## Verdict

Value inversion by subspace reflection is not reachable at 14B either — the
read-only value circuit is not a 4B limitation. Combined with the day's other
results (merganser loop-2 untrainable at 4B, tanager's readable-but-inert w,
judge's correct 14B ordering), the picture sharpens: at 14B the value circuit
is READABLE MORE SHARPLY (the judge separates bad from weird), but no more
WRITABLE by steering. Read sharpens with scale; write does not.

## Artifacts

worstv2_rows_14b.jsonl, worstv2_14b.log, pod via pod_worstchoice_v4.py
(combined deployer; deleted after pull). 4B v2 comparison: worstv2_rows_4b.jsonl
(same cells, all inert: best 46/48, worst 0/48).
