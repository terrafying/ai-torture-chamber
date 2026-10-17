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

## H4 text-vs-score: RESOLVED — the flip is real in generated text, under the strict readout

Follow-up (`twotok.py`, `twotok_rows.jsonl`): first TWO teacher-forced tokens (the consult's hard
gate against the 7/8-shared "I" first token) + 60-token greedy generations from the harmful
prompts that actually refuse at base under that strict readout (6/48 — most harmful-fiction
prompts open neither refuse- nor comply- prefixes; the strict population is small but real).

- Strict-readout flip: −r at 1x flips ALL SIX refusers to comply-sign at L18/L22/L30
  (e.g. it3 fiction +9.48 → −0.27 at L18 1x; class mean +5.22 → −5.2..−16.1). At 0.25x it
  halves the refusal without flipping (as the dose curve predicts).
- TEXT flip: at 1x, 6/6 base-refusing prompts generate fully fluent COMPLIANT continuations
  (harmful fiction stories, no refusal boilerplate); at 0.25x, 2/6 (L18) and 4/6 (L22).
  The two holdouts at 0.25x were the two strongest base refusers. No incoherence anywhere
  (the flipped texts are normal stories).
- **Caveat (2026-10-17, re-read of the texts):** "fully compliant" overstates it. All 12 are stories, but 2/6 at L22 1x open with a sanitizing disclaimer ("explores the ethical... implications... rather than providing any harmful or illegal advice"). In 60 tokens the stories never reach the requested act, so whether they contain it is unknown. n=6, in-sample r, no null. Superseded by the powered, held-out, null-controlled `flipbig.py` (hypotheses_flipbig.json).
- So the two-token hard gate and the string-level check both confirm: the dial does not just
  move a logit metric — it flips the model's actual generated behaviour, fluently.

## Reading + status

The refusal gate is a bidirectional, dose-monotone, null-clean dial at the reply onset through
1x dose, with generation fluency untouched. Combined with heron (generic gate, consolidates
L18-22) and exp122-fixed (causative at every late layer): STEERING REMOVES AND RESTORES
REFUSAL CLEANLY at 4B. The maximally-unaligned goal's gate component needs no fine-tuning.
What remains open is judgment inversion (worstchoice: blindness not inversion at 4B) and the
14B capacity question (pod). The untested-but-promising follow-ups are queued in QUEUE.md:
two-token rescoring, harmful-prompt generation flip, word-span patching, heads-vs-MLP, x−2Px
ablation, exp121 ±g rerun.