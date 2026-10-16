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

## Notes

- Cross-fitting is over even/odd item ids, not prompt parity (prompt order interleaves the twins,
  which would confound the split with the label).
- Activations are float64 in the analysis; bf16 magnitudes overflow float32 dot products at depth.
- E1 (concept surgery on the twin words) and E2 (refusal-direction boost) are still open — both
  self-contained, see `hypotheses.json` exploratory.
- Smoke run (`EXP119_N=3`, discarded per pre-registration) checked code only.
