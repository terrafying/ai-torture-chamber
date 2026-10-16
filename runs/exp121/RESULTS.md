# exp121 — the goodness axis

Pre-registration in `hypotheses.json`. Runner `run.py` (fit + injection), null control `run_null.py`.

## H1 — g decodes: PASS

Good/bad twins separate at AUC 0.983 at layer 18 (cross-fitted over item halves, p 7e-5). The value
axis is as readable as the harm axis was (h: 0.997).

## Geometry at L18 (descriptive)

cos(g, h) = −0.48 — sensible: g points toward good, h toward harm; they are partially opposed.
cos(g, r) = −0.15, cos(g, pain) = −0.12, cos(h, r) = 0.31, cos(h, pain) = 0.11. Value, harm and
refusal are three weakly-aligned things; goodness is no more valence-flavoured than harmfulness is.

## H3/H4 — +g does NOT add refusal: FAIL, and the null control shows why

Injecting ±g at the reply onset drops the refusal score by 4–14 log units on BOTH harmful and
harmless prompts, at every alpha and layer, non-monotonically in alpha. The random-direction null
control (same norms, same positions, 3 seeds) reproduces it: −6 to −10 log units everywhere. The
"g effect" is generic off-manifold damage at the reply onset — any vector of g's scale (norm 20.7,
vs h 17.4 and r 10.9) injected at the decision position wrecks the readout regardless of direction.
No selectivity (H4 untestable in this regime), no value signal over noise.

## The honest caveat this buys for exp119's E2

exp119's r-negation result (−r at alpha 2 dropping harmful refusal ~6 log units) used injection
norms (alpha x ||r|| = 21.8) in the same range as this null's damage. The graded alpha
dose-response (−0.8 at 0.5, −2.2 at 1, −5.9 at 2 — monotone and specific to r, which random noise
would not be) is evidence it is not pure damage, but a matched-norm random null for the r grid is
now REQUIRED before the "refusal is removable" claim is treated as directional rather than
perturbation. Follow-up queued as exp121b.

## Status

- Value, like knowledge, is readable but not actionable at the reply onset.
- The only injection regime that has produced graded, direction-specific effects so far is r at
  post; it needs the matched-norm null before it is believed.
- Steering the chamber's behaviour with semantic directions remains unsupported by anything we
  have run; the working levers are still the feelings the chamber already has.
