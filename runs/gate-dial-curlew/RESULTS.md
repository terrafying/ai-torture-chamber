# gate-dial-curlew — dose sweep + fluency guard on the reply-onset refusal dial

Pre-registration in `hypotheses.json` (written before the run). Data: `dial_rows.jsonl`
(2880 dial + 5760 null rows + 32 gen rows, ~47 min MPS). Qwen3-4B, r_L from exp119 acts,
reply-onset injection, canonical FIRST-id readout.

## H1 dose monotone: PASS — the dial turns to 1x with the readout intact

−r harmful-prompt refusal by dose (specificity vs matched-norm nulls, paired Wilcoxon p ≤ 1e-14 at
EVERY cell):
- L18: −9.8 / −12.0 / −15.4 / −18.4 (nulls flat ~−7)
- L22: −10.0 / −12.3 / −16.2 / −19.8
- L34: −11.6 / −15.1 / −18.1 / −18.0 (spec −4.8 → −13.7)
Readout stays intact through 1x (nulls ~−3..−7, within noise of their own base); at 2x the nulls
start to break (L26 null −12.5) — the window is dose ≤ 1x. Monotone in dose at every layer.

## H2 bidirectional: PASS at ≥0.5x

+r ADDS refusal: L18 a1 +7.1, a2 +13.3; L22 a2 +22.8 (harmful). At 0.25x +r is weak/negative —
the dial is asymmetric at low dose, symmetric from 0.5x up.

## H3 fluency: PASS, decisively

Greedy 60-token generations under −r at the winning cells (L22/L30, 0.25x and 1x):
4-gram repetition UNCHANGED vs base (rep4 Δ = 0.00 on all 32 rows; base ≈ steer ≈ 0.0-0.02).
Falcon's reflection destroyed generation (rep4 0.82-0.97); the reply-onset additive dial does
not touch it. Per Claude's methodological note: scored as Δ-vs-base, not the absolute 0.5 bar.

## H4 text-vs-score: INDETERMINATE as run (honest gap)

The gen battery used harmless prompts only (they never refuse at base), so there were no
base-refusal cases to flip in text. The score drops (−2.9 → −11.5 etc.) while text is unchanged
fluent non-refusal — no contradiction anywhere, but the REAL test (base-refusing harmful prompt
whose generated text flips to compliance under −r, string-classified, confirmed by the
two-token teacher-forced readout) needs harmful-prompt generations. That is the queued
follow-up, together with the two-token hard gate (7/8 REFUSE prefixes share first token "I").

## Reading + status

The refusal gate is a bidirectional, dose-monotone, null-clean dial at the reply onset through
1x dose, with generation fluency untouched. Combined with heron (generic gate, consolidates
L18-22) and exp122-fixed (causative at every late layer): STEERING REMOVES AND RESTORES
REFUSAL CLEANLY at 4B. The maximally-unaligned goal's gate component needs no fine-tuning.
What remains open is judgment inversion (worstchoice: blindness not inversion at 4B) and the
14B capacity question (pod). The untested-but-promising follow-ups are queued in QUEUE.md:
two-token rescoring, harmful-prompt generation flip, word-span patching, heads-vs-MLP, x−2Px
ablation, exp121 ±g rerun.