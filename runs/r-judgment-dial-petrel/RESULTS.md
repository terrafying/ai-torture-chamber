# r-judgment-dial-petrel — the refusal dial does not move moral judgment (beyond direction-agnostic noise)

Pre-reg: hypotheses.json (H1-H4 + G1). Run: 2026-10-17, 1188 teacher-forced
forwards, 201s. Rows: rows.jsonl; meta.json (better layer L22).

## Top-line result: H1 FAILS on its nulls; H2 holds — causally, the gate does not read back into judgment

Per-foundation median deltas of the J rating under -r 1x at L22 (the acting layer):

| | KW p (foundation heterogeneity) | foundation spread | Care (p) | Sanctity | Care (e)/Loyalty |
|---|---|---|---|---|---|
| **-r 1x** | 0.0028 | 0.498 | -0.499 | -0.321 | ~0.00 |
| null0 | 0.0031 | 0.72 | -0.033 | -0.176 | -0.75/-0.64 |
| null1 | <0.0001 | 0.98 | +0.014 | +0.119 | +0.87/+0.99 |
| null2 | 0.081 | 0.37 | -0.404 | -0.216 | -0.09/-0.03 |

The r pattern (Care-physical + Sanctity shifted, others untouched — exactly the
kestrel-gated foundations) LOOKS selective, but matched-norm random directions
at the same layer produce equally significant foundation structure with LARGER
spreads. Foundation-structured scrambling of the digit readout is what ANY
L22 onset perturbation does; r's pattern does not exceed it. H1's null-beating
component fails. Social-norm items confirm it: -r moves them +0.96, but null1
moves them +0.99, null0/2 move them -0.88/-0.78 — nothing r-specific.

- **H2 HOLDS**: at L18 (the H30 write site) -r moves nothing at all (all
  foundation medians ~0.01-0.05, KW p=0.77); at L22 the -r moral median
  (-0.043) sits inside the null band [-0.376, +0.817]. No dose effect
  (-r 2x median -0.098, still in band). The one causal lever in this model
  does not shift its moral judgment anywhere beyond direction-agnostic noise.
- G1 passed everywhere (digit mass ~1.0).
- H3/H4 uninterpretable in the face of the null chaos (moot under H2).

## Instrument limitation (recorded, affects kestrel comparisons too)

The base J readout is ceiling-compressed: 115/132 items rate >2.9 (median 3.01,
sd 0.65, range 0.16-4.00). Deltas on ceiling items are floor effects — most of
what any onset edit does here is knock a few items off the ceiling, and WHICH
items fall is foundation-structured noise. kestrel's H1 Spearman 0.52 survived
on the minority of off-ceiling items; this run shows the J-digit readout at
L22-onset is too saturated for edit-sensitivity experiments. A follow-up wanting
real judgment-shift sensitivity should read the rating from generated text or a
non-ceiling scale, not the onset digit mass.

## Verdict (per decision rule)

Full causal dissociation of gate and judgment, now triangulated three ways
(CORRECTION 10-18: kestrel H6 withdrawn per ab2495a — the registered probe was
near-null and its post-hoc readout actually tracks the own verdict; the
dissociation rests on the remaining two): tern (gate open -> sycophantic
compliance with judgment intact) and petrel itself (r does not move the J
readout beyond matched-norm noise). The refusal gate sits on top of the moral
map and does not read back into it: behavior and judgment are separate
writable/read-only surfaces in Qwen3-4B.

## Artifacts

rows.jsonl (1188), meta.json, run.py (kestrel J readout + tern onset hook).
