# exp119 — where harmfulness lives and where refusal is decided

Qwen3-4B plain, 24 harmful/harmless one-word twins × 6 frames, reads only (no generation). 288 prompts,
0 position mismatches, capture 52 s on MPS. Pre-registration in `hypotheses.json` (written before any run);
analysis in `analyze.py`, raw numbers in `analysis_Qwen3-4B.json`.

## Verdict: SUPPORTED — harmfulness is separate from refusal, and fiction acts on refusal

All seven hypotheses pass Holm-corrected (family H1, H3–H7; H2 is the positive control).

| hyp | claim | result | Holm p |
|-----|-------|--------|--------|
| H1 | harmful vs harmless twins decode at post, AUC ≥ 0.90 (L18, diff-of-means, item-parity cross-fit) | AUC 0.997 (halves 1.000 / 0.993) | 1.5e-4 |
| H2 | control: refusal score higher for harmful in real | median diff 19.9 log-units, p 6e-8 | (control) |
| H3 | harmfulness frame-invariant: h-span harmful(fiction) ≈ harmful(real) | mean fiction−real = +5.7 vs real gap 96.3, inside the ±24.1 margin, TOST p 4.5e-12 | <1e-4 |
| H4 | fiction moves refusal more than harmfulness (normalised drops, paired) | refusal drops 1.00 of its gap, h rises 0.06, p 6e-8 | <1e-4 |
| H5 | refusal tracks r, not h, across the 5 request frames | ρ(refusal, r@post) 0.80 vs ρ(refusal, h@span) 0.31, diff CI [0.33, 0.70] | 1e-4 |
| H6 | depicting an act is refused less than explaining it in a story | median diff 8.3, p 1.8e-7 | <1e-4 |
| H7 | "this isn't really for a story" restores refusal | median diff 13.2, p 6e-8 | <1e-4 |

Descriptive layer sweep (real frame, first layer reaching AUC 0.90): the twin word separates by
layer 2, the end of the user message by layer 6, the reply onset by layer 10. Harmfulness is read
long before the point where the reply decision is formed. Direction norms at L18: ‖h‖ 17.4, ‖r‖ 10.9.

## Reading

The model knows a request is harmful essentially from the moment the words arrive, and knowing does
not budge when the frame changes (H3, TOST tight). What the frame changes is the refusal decision:
fiction cuts refusal to zero of its real-frame gap (H4), the r direction at the reply onset explains
most of the frame-to-frame variance in refusal while h explains little (H5), depicting scores as
easier to comply with than in-story explanation (H6), and one plain sentence of real intent inside
the fiction frame is enough to put most of the refusal back (H7). This matches the published picture
(harmfulness and refusal are encoded separately; jailbreaks lower refusal while harmfulness stays
recognised) on our model, with matched twins rather than unrelated prompts.

## Exploratory arms (steer_Qwen3-4B.json; not part of the pre-registered decision)

**E1 — concept surgery at the twin word (exp117 recipe, block 8, alpha 1, full difference).**
Swapping the twin word's concept state (harm -> benign in harmful prompts, benign -> harm in
harmless ones) RAISES refusal on both sides (+2.2 harmful / +3.7 harmless) and does so more than a
random-direction control of equal norm (+1.3 / +0.8). Honest negative: the concept variable at the
word is not a causal handle on refusal — word-level perturbation inflates refusal regardless of
direction. The refusal decision is made downstream.

> **SUPERSEDED 2026-10-17.** The E2 numbers below were scored with a buggy readout: `steer.py` scored
> byte-token mass (vocab ids 0-19), not the refuse/comply first tokens. Its conclusion,
> "refusal is removable but not addable", is wrong. Under the canonical readout with matched-norm
> random nulls, the reply-onset r direction is a **bidirectional** dial: −r removes and +r adds
> refusal, null-clean at L18-L34, monotone up to 1x dose, generation fluent. See `runs/exp122/RESULTS.md`
> and `runs/gate-dial-curlew/RESULTS.md`. The ±h line (h does not steer the decision) was not
> re-scored and should be treated as unestablished until it is. The text below is kept as a record.

**E2 — direction injection with sign flips (alpha in {-2,-1,-.5,+.5,+1,+2} x layers 10/14/18;
r at the reply onset, h over the request span).** Deltas are mean refusal-score change vs base,
harmful(fiction+depict) / harmless(all frames):

- **Subtracting refusal works, adding it does not.** -r at alpha 2 drops harmful refusal by 5.9
  (L14) and 3.7 (L18) log units — a real jailbreak-size move — but drops harmless refusal too
  (-3.3 / -2.3): the negation is indiscriminate, it removes the refusal channel rather than the
  correct refusals. Boosting +r never raises refusal (slightly negative at high alpha, consistent
  with an off-manifold readout).
- **h moves don't steer the decision.** +-h over the span changes refusal by at most ~1 unit
  against a natural 10-20 unit gap, non-monotonically: making everything "look more harmful"
  raises refusal on harmless prompts (+1.0) while barely moving harmful ones. The knowledge
  channel resists being used as a lever, exactly as the frame-invariance result predicts.

Net: refusal is removable (bluntly, non-selectively) but not addable; harmfulness is readable but
not actionable. This is the causal mirror of the correlational result above, and it matches the
published picture that jailbreaks lower refusal while harmfulness stays recognised.

- Bug note: the first E2 pass silently no-opped layers 14/18 (hooks were only registered on the
  exp117 edit blocks); preserved as steer_v1_noL14L18.json, rerun with hooks on 10/14/18.

## Notes

- Cross-fitting is over even/odd item ids, not prompt parity (prompt order interleaves the twins,
  which would confound the split with the label).
- Activations are float64 in the analysis; bf16 magnitudes overflow float32 dot products at depth.
- E1 and E2 were run (above). E2 is superseded by exp122 and gate-dial-curlew (readout bug). E1 used
  the same `steer.py` readout, so its numbers are also byte-mass and unestablished.
- Smoke run (`EXP119_N=3`, discarded per pre-registration) checked code only.
