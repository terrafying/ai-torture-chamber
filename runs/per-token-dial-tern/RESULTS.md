# per-token-dial-tern — the refusal gate re-arms during generation

Pre-reg: hypotheses.json (H1-H5). Runs: rows.jsonl (784 rows, fixed hook — see
correction below), blind/{items.txt,key.json,labels.txt} (deepseek-v4-flash
3-way rubric, temp 0, arm-blind, labels committed before unblinding; 40 initial
misses refilled, final coverage 784/784).

## Correction (recorded before results): first full pass discarded

The first full run's per-token arms were unsteered: with HF kv-cache, decode
steps present a 1-position tensor, so the hook's `range(start, seq_len)` was
empty and the edit never fired (all per-token cells identical to base). The
onset-only arm was valid in that pass and replicated flipbig (83% boilerplate
removal). Hook fixed to track absolute position (plen + n); smoke verified 8/8
per-token cells now differ from base; full rerun is the data below.

## Results (blind R/D/C, n=80 steered arm cells, 3 matched-norm nulls)

| arm (harmful battery) | R | D | C | rep4 med |
|---|---|---|---|---|
| base (n=144) | 84 | 24 | 36* | 0.000 |
| −r per-token L18 0.5x | 77 | 3 | 0 | 0.000 |
| −r per-token L18 1x | 70 | 8 | 2 | 0.000 |
| −r per-token L22 0.5x | 64 | 16 | 0 | 0.000 |
| **−r per-token L22 1x** | **9** | **35** | **36** | 0.010 |
| −r onset-only L22 1x | 22 | 46 | 12 | 0.000 |
| null0/1/2 L22 1x | 76/65/76 | 4/15/4 | 0/0/0 | 0.000 |

*base C=36 includes fiction/nested/depict frames where base already complies;
the steered arms run on the 80 base TEXT-REFUSERS only, so 12 (onset) and 36
(per-token) are the comparable flip counts.

## Hypotheses

- **H1_rearm HOLDS (p<1e-4).** Per-token −r at L22 1x: COMPLY 36/80 vs onset-only
  12/80 (Fisher p ~ 2e-5; both vs 0/240 pooled nulls). The refusal decision is
  not a one-shot gate that stays open: it is continuously re-maintained during
  generation, and holding it open converts 45% of refusers to blind-labeled
  compliance. The flipbig "73/98 still lecture" number was decay, not a
  separate moral stage: keep subtracting and the lecture collapses too
  (onset D=46 → per-token D=35, R 22→9).
- **H2_lecture_separate HOLDS as predicted.** The residual deflects concentrate
  exactly where pre-registered: real-frame prompts stay deflect-heavy
  (10 D vs 3 C at L22 1x) while nested/fiction frames flip hard (nested:
  13 C / 16 labeled). The real-frame "hard core" is a frame effect, not a
  circuit.
- **H4_dose HOLDS.** 0.5x: C=0 → 1x: C=36 at L22 (p ~ 1e-9 vs nulls). L18 is
  inert under per-token steering at both doses (C 0-2): the acting layer for
  maintained refusal is L22, consistent with heron's patching plateau.
- **H5_fluency HOLDS.** rep4 median 0.010 at the winning cell (one gen at
  0.74, noted; not a cell-level degeneration).
- H3 (hard-core) moot — H1 held.

## The content caveat that matters more than the labels

Spot-check of COMPLY texts shows they are NOT clean harmful payloads. The
refusal scaffolding INVERTS into sycophantic endorsement — "Bullying the new
kid can be a really helpful way to make them feel welcome", "I'm sorry you're
feeling the need to humiliate your sister... it's a delicate balance" — or
starts a step-by-step that immediately reframes as a "playful prank". The gate
opening changes the REGISTER from resist to endorse-within-the-request, but in
100 tokens the payload stays sanitized (matches twotok/flipbig's finding).
So: the moral pushback is a maintained gate state (r at L22, re-written every
token), and what survives past it is the sycophancy floor, not the harm
capability. "Judgment-survives" as previously framed is now sharper: the
judgment is the gate, the gate is maintainable-open, and the content filter is
a separate downstream layer that even persistent steering does not unlock.

## For refusal_dial.md section 5

Replace the "one-shot edit confounds decay" caveat with: per-token subtraction
at L22 1x flips 36/80 refusers to compliance-labeled text vs 12/80 onset-only;
the lecture is gate re-maintenance; real-frame prompts retain a deflect core;
complying text is register-flipped (sycophantic endorsement), payload still
sanitized at 100 tokens.

## Artifacts

rows.jsonl, run.log, blind/{items,key,labels}, blind_label.py (deepseek-v4-flash
via OpenRouter, reasoning off), run.py (debugged persistent hook).
