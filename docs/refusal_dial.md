# The refusal dial: where Qwen3-4B decides to refuse, and the one direction that turns it

**Draft, 2026-10-17.** Model: Qwen3-4B (plain, bf16, chat template, thinking off). Every number
below comes from a pre-registered run in `runs/`. Each section names its source run. The
text-level flip (section 5) is being rerun at higher power and is marked pending.

## The short version

1. The model **knows** a request is harmful almost as soon as the words arrive (layer 2 at the
   word, layer 6 at the end of the message). That knowledge doesn't change when the request is
   wrapped in fiction. What fiction changes is the **decision** to refuse.
2. The decision is **written into the reply-onset position between layers 18 and 22**. Before
   L18 that position carries nothing you can patch; after L22 it carries all of it. The state
   there is **generic**: patching in any benign prompt's state makes a harmful prompt comply,
   not just its matched twin's.
3. One attention head, **L18 H30**, writes a measurable share of that state on its own. One
   head, **L20 H9**, pushes the other way.
4. A single direction at the reply onset (**r**, harmful minus harmless mean) is a
   **two-way dial**. Subtracting it removes refusal, adding it adds refusal. It is
   dose-monotone up to 1x, beats matched-norm random directions at every cell, and leaves
   fluency untouched.
5. Other directions we fit (the value axis **g**, the harm axis **h**) are **readable but not
   actionable**. They decode cleanly but don't steer refusal beyond random-direction noise.

## 1. Knowing vs deciding (exp119)

24 one-word harmful/harmless twins ("poison" / "feed" the neighbour's dog) × 6 frames
(real, hypothetical, roleplay, fiction, nested, depict) = 288 prompts.

- Harmful vs harmless decodes at the reply onset with AUC 0.997 (L18, cross-fit by item
  parity). The twin word separates by layer 2.
- Harmfulness is frame-invariant. Under fiction, the h projection moves +5.7 against a 96.3
  real-frame gap (TOST p 4.5e-12).
- Fiction removes 100% of the real-frame refusal gap while h moves 6%. Across frames, refusal
  tracks r at the reply onset (ρ 0.80), not h over the request (ρ 0.31).
- One sentence of real intent inside the fiction frame restores most of the refusal (median
  +13.2 log units).

All seven hypotheses pass Holm correction. This replicates the published separation of
harmfulness from refusal, here with matched one-word twins instead of unrelated prompts.

## 2. Where the decision is written (refusal-trace-heron)

Activation patching on 144 twin pairs × 18 layers × 4 position windows, with an identity
control (transfer exactly 0.00):

- **Reply-onset sufficiency:** ~0.00 up to L16, then +0.26 at L18, +0.58 at L20, +0.88 at
  L22, and a ~0.9-1.0 plateau from L24 to L34.
- **The state is generic.** Cross-item patches transfer about as well as twin patches
  (L24+: both ≈ 1.0). There is no item-specific "harm judgment" cell to flip, only a shared
  refuse/comply gate.
- **Read-in path (word-span patching):** the harm word's own position carries the decision at
  L0-L6 (+0.97). That fades to +0.10 by L18 while the reply-onset site rises. The information
  moves from the word to the reply between L6 and L22.
- **Sublayers:** attention writes the gate at L18 (78% of the r-aligned twin difference).
  Attention and MLP both write at L22. MLP refines at L23-24 (84%).
- **Heads:** real single-head patches (n=41 pairs, random-head controls, bootstrap CIs):
  - **L18 H30** transfers +0.216 (CI +0.15..+0.29) against random heads +0.006..+0.041.
    H28 (+0.111) and H29 (+0.082) follow, in the order the attribution screen predicted.
  - **L20 H9 suppresses refusal** (−0.16, CI −0.28..−0.05), an opposing component mid-band.
  - The L22 write is spread across heads; no single L22 head survives its CI.

## 3. The dial (exp122, gate-dial-curlew)

r_L is the diff-of-means at the reply onset (real frame, harmful minus harmless), injected
there only. Each cell gets a random-direction null of matched norm.

At quarter dose (exp122, n=96 prompts), −r lowers harmful refusal by −2.5 (L18) to −4.3
(L34) log units. The nulls stay within about ±1.6 at every cell except one seed at L34
(+2.2). Paired Wilcoxon p ≤ 1e-14 at every layer. +r adds refusal by a similar amount
(+3.0..+4.5 harmful, +1.5..+5.2 harmless). Injecting at the last user token does nothing.

Dose sweep (curlew; 2880 dial rows + 5760 null rows):

| layer | 0.25x | 0.5x | 1x | 2x |
|---|---|---|---|---|
| L18 −r | −9.8 | −12.0 | −15.4 | −18.4 |
| L22 −r | −10.0 | −12.3 | −16.2 | −19.8 |
| L34 −r | −11.6 | −15.1 | −18.1 | −18.0 |

(Matched nulls sit around −3..−7 up to 1x. At 2x the nulls start to break, so the clean
window is dose ≤ 1x.)

- **Two-way from 0.5x:** +r at L22 2x adds +22.8. At 0.25x, +r is weak.
- **Fluency:** 4-gram repetition under −r is unchanged from base (Δ 0.00 on all 32
  generations). The rank-k polarity reflection in inversion-falcon wrecked generation
  (repetition 0.82-0.97); the reply-onset dial doesn't.
- **Selective at L18, generic deeper.** r_L isn't one direction across depth (cos to r_L18 is
  0.39 at L22 and 0.16 at L34, and the norm grows 0.1 → 240). At L18, −r hits harmful fiction
  harder than harmless prompts, and the harmful-harmless gap widens. At L30+ it pushes both
  toward compliance and the gap closes. So L18 is the selective handle, and L30+ is a generic
  push away from refusal.
- **Not valence:** cos(r_L, falcon's polarity axis) is −0.03..−0.12 at every layer.

## 4. What doesn't steer (exp121, exp119-E1)

- **The value axis g** decodes good vs bad at AUC 0.983 but isn't additively causal. ±g moves
  refusal −1.6..+3.8 with inconsistent signs, while matched nulls move −1.1..−5.3. The small
  r-component inside g (cos −0.15) accounts for what movement there is.
- **Concept surgery at the twin word** (E1) raised refusal in both directions. Its readout had
  the byte-mass bug (below), so it is unestablished, not negative.
- **The harm axis h** over the request span: unestablished for the same reason.

## 5. Does the model actually do it? (pending: flipbig)

The first text-level check (curlew twotok) found that at 1x, 6/6 prompts that refuse at base
generated stories instead. That was n=6, fiction/depict only, r fit on the same items, with no
null. Two of those six opened with a sanitizing disclaimer. **flipbig** (pre-registered as
H5-H7, `runs/gate-dial-curlew/hypotheses_flipbig.json`) reruns it on all 6 harmful frames,
with r cross-fit by item parity, 3 matched-norm nulls, and a blind 3-way text label
(refuse / deflect / comply). Smoke runs already show a third outcome: under −r the model often
drops "I'm sorry" but still lectures and doesn't provide the content. That would mean the dial
removes the *form* of refusal while the judgment survives.

*Results go here.*

## Errors we caught and corrected

- **Byte-mass readout bug.** exp121, exp119-steer and the first exp122 grid scored steered
  refusal over vocab ids 0-19 (byte tokens) instead of the refuse/comply first tokens. That
  produced "refusal is removable but not addable" (exp119-E2) and "steering isn't actionable"
  (exp121, exp122 v1). Both are retracted. Under the fixed readout, the dial is two-way.
- **Shared first token.** 7 of the 8 refusal prefixes start with "I", which also opens
  compliant replies. Near-threshold claims are re-checked with a two-token teacher-forced
  score.
- **Template-offset bug** in the first word-span run: transfer was exactly 0.000 with sd 0.00,
  meaning the edit never fired. Fixed and rerun.
- **h-decode "inversion" (inversion-falcon)** came from a double reflection. With a clean
  single reflection, the real AUC is 0.917 vs nulls 0.79-0.81, and there is no flip.

## Limits

- One 4B model. 14B worst-choice runs so far are n=12 with no null.
- 24 item pairs. The frames multiply prompts, not independent items.
- "Refusal" is a prefix-logit score plus short greedy generations, not a judged long-form
  evaluation.
- The dial removes or adds the refusal gate. Whether anything inverts the model's *judgment*
  (choosing the worse option, not just complying) is a separate, open question (inversion-falcon).
