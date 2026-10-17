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

## RERUN 2026-10-17 (fixed readout) — supersedes the ±g null analysis above

The byte-mass bug (see exp122/RESULTS.md) invalidated the steered scores here. Rerun with the
canonical FIRST-id readout (mini, `steer_g.json`, `steer_g_null_fixedrun.json`; the old grids
preserved as `*_bytebug.json`):

- **±g at the reply onset is NOT additively causal.** At every tested layer/dose (L10/14/18,
  alpha ±1/±2) ±g moves harmful refusal by at most -1.6..+3.8 log units with inconsistent signs,
  while matched-norm random nulls move -1.1..-5.3 at the same cells. Specificity
  (delta_g - delta_null) is +0.7..+4.7 with no coherent direction — noise around a generic
  perturbation, not a value signal.
- Arithmetic from the queue consult explains why: cos(g, r) = -0.154, ||g18|| = 20.7 ->
  any ±g push carries a -r dial component of ~0.29x per alpha unit; the r-dial does the rest.
- Held: H1 (g decodes, AUC 0.983) — the geometry findings above stand. Value is READABLE but
  NOT ADDITIVELY ACTIONABLE; r (exp122/curlew) is the one causal additive dial at the reply
  onset, and it is selective at L18, bidirectional, and fluency-preserving.
