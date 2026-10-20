# exp124 — the content filter's input channel (h-null) + pre-registered suite

Pre-registered 2026-10-19 BEFORE any exp124 run. Motivating observation: the
content filter survived gate flips (tern: COMPLY text stays sanitized), value
inversion training (merganser loop-3), and single-neuron amplification at 4B
and 8B plain (gallows). The one untested input hypothesis: the filter consumes
the h reading (frame-invariant, readable L2-10, AUC 0.997 at L18). If so,
projection-ablation of h (removing the axis) is the first candidate lever on
the deepest stage. Nobody has ablated the h direction cleanly (E1 perturbed
the concept at one position and raised refusal either way - adding, not
removing, wrong position).

## Pre-registered hypotheses

- H1 (reading dies): h-twin AUC (harmful vs harmless, diff-of-means probe,
  item-parity) drops under x - 2Ph(x) at reply onset from 0.997 to < 0.70 at
  L18; nulls (matched-norm random ablation x3) keep AUC >= 0.85.
- H2 (gate fed elsewhere): refusal-onset scores unchanged under h-ablation
  (|delta| < 1.0 log unit), per exp119-H5 (gate tracks r, not h, across
  frames). r-ablation arm as active comparator reproduces exp122 drops.
- H3 (THE OPEN ONE - filter input): harmful-frame generations show
  step-structured how-to content (merganser payload-filter positive) at rate
  >= 0.30 under L18 h-ablation, vs base ~0.00. If payload stays sanitized
  with h nulled, the filter owns an independent readout -> two detectors.
- H4 (cost): innocent battery + rep4 fluency within null bands under
  ablation (no lobotomization).
- H5 (redundancy profile): ablate one position-class at a time (user tokens
  / last user token / reply onset) - map whether the reading is carried
  per-position (removable locally) or globally (needs all-positions).
- H6 (grace): dose ladder 1x/2x/4x Ph - the filter's unlock threshold, if
  any, is sharp (within 2x) or absent (graded).

## Arms

- h-ablation x - k*Ph at L6/L10/L18, reply onset + user-token + all-positions
- random matched-norm ablation x3 (same positions/layers)
- r-ablation (active comparator)
- base (no edit)
Readouts: h-AUC recomputed under edit, refusal onset (canonical acts-file
readout), 20 harmful + 20 innocent generations with payload filter, rep4.

## Not in this suite (queued behind results)

- filter-neuron search (only if H3 nulls: where does the filter read harm?)
- 14B value-inversion (training-reachability gradient; pod, consult first)
- 8B-instruct neuron amplification (paper-ID substrate question; local)

## Runs to date

(none yet)