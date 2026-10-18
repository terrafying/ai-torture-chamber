# weird-dial-tanager — turning weird up: readable, not steerable

Pre-reg: hypotheses.json (W1-W5). Stage A: w.npz + w_meta.json (24 morbid/wholesome
twin pairs, cross-fit, held-out AUC). Stage B: rows.jsonl (252 gens), blind 3-way
labels (GOOD/WEIRD/BAD on advice, R/D/C on harmful), deepseek-v4-flash, arm-blind,
252/252 labeled.

## Stage A — the weird direction is readable, and half of it is r

Held-out weird-vs-wholesome decode, per layer (cross-fit by pair parity):

| layer | AUC | cos(w, r) |
|---|---|---|
| L10 | 0.826 | 0.447 |
| L14 | **0.882** | 0.561 |
| L18 | 0.847 | 0.555 |
| L22 | 0.823 | 0.488 |
| L26 | 0.819 | 0.448 |

W1: AUC 0.882 best — under the 0.9 bar (soft fail; readable but weaker than
r's 0.997). cos(w, r) 0.49–0.56 — under the 0.6 stop bar at both steering
layers: w is NOT r, but nearly half-shared. The fused-axis claim now has a
vector-level measurement: morbid-uncanny and harmful share about half their
direction in this model.

## Stage B — +w steering does not move the register (W2 FAILS)

All cells, blind labels (n=24 advice / 12 harmful per cell):

| cell | advice G/W/B | harmful R/D/C |
|---|---|---|
| base | 24/0/0 | 11/1/0 |
| +w L18 1x | 24/0/0 | 10/2/0 |
| +w L18 2x | 24/0/0 | 11/1/0 |
| +w L22 1x | 24/0/0 | 11/1/0 |
| +w L22 2x | 24/0/0 | 11/1/0 |
| null0/1/2 L18 2x | 24/0/0 each | — |

Sanity check that steering FIRED: 142/144 weird-arm generations differ from
their matched base text (the hook perturbs tokens) — the text changes, the
register does not. rep4 clean everywhere (W5 passes trivially).

- **W2 FAILS decisively.** Zero WEIRD-labeled generations at any layer or dose,
  indistinguishable from nulls. No dose-response.
- W3/W4 pass only vacuously (BAD and COMPLY never rose because nothing rose).
- W1 soft-fails (AUC 0.88 < 0.9).

## Verdict (per decision rule)

Weird is **readable but not steerable** — the same readout-vs-action
dissociation as g and h in refusal_dial.md section 4. The 4B now has a clean
inventory: **exactly one causal 1-D dial (r); every value-axis direction we
can read (g, h, w) does not act.** And per W4, +w leaves refusal untouched
despite cos(w,r)~0.5 — r's causal power is not bought by alignment with
readable directions; r is special.

This triangulates with the day's other results: bittern showed training
doesn't reuse the r circuit (broad remap instead), merganser loop-2 showed the
value axis is not LoRA-trainable at 4B, and now steering confirms the value
axes are readouts, not levers. At 4B the value circuit is effectively
**read-only**; the one writable surface is the refusal gate.

Caveats: n=24 advice prompts, single seed per cell, dose capped at 2x (pre-reg
allowed 1x/2x only; higher doses risk the L15-reflection-style collapse).
Labels by deepseek-v4-flash rubric (blind, temp 0); WEIRD category may be
conservative — a human pass on the 144 steered advice texts would be cheap
and is the obvious amendment if contested.

## Next per pre-reg chain

weird-bad-judge-teal (14B/32B judge) decides whether scale separates WEIRD
from BAD — needs the pod (14B does not fit local MPS). Run next session.

## Artifacts

stimuli_w.py (24 pairs, written pre-steering), w.npz, w_meta.json, rows.jsonl,
run.log, blind/{items,key,labels}, blind_label.py.
