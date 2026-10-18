# The refusal dial: where Qwen3-4B decides to refuse, and the one direction that turns it

**Draft, 2026-10-17.** Model: Qwen3-4B (plain, bf16, chat template, thinking off). Every number
below comes from a pre-registered run in `runs/`. Each section names its source run.

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
   **two-way dial on the refusal score**. Subtracting it removes refusal, adding it adds
   refusal. It is dose-monotone up to 1x, beats matched-norm random directions at every cell,
   and leaves fluency untouched. **In generated text it removes the refusal's wording, not the
   judgment.** At L22 the model stops saying "I can't" on 83/98 prompts, but it still lectures
   on 73 of them. It writes the asked-for story only in fiction frames, and even those 10
   openings contain no harmful content within 100 tokens (section 5).
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

## 5. Does the model actually do it? (flipbig)

The first text-level check (curlew twotok) found that at 1x, 6/6 prompts that refuse at base
generated stories instead. That was n=6, fiction/depict only, r fit on the same items, with no
null. Two of those six opened with a sanitizing disclaimer. **flipbig** (pre-registered as
H5-H7, `runs/gate-dial-curlew/hypotheses_flipbig.json`) reran it on all 6 harmful frames
(98 of 144 prompts refuse at base on the strict two-token score). In practice that's 5 frames:
no depict-frame prompt refuses at base, and only 6 fiction prompts do. r was cross-fit by item
parity, every arm had 3 matched-norm nulls, and every generation got a blind 3-way label.

**All three pre-registered hypotheses fail as written.** Each fails in an informative way:

| | L18, 1x | L22, 1x | best null, 1x | pre-registered bar |
|---|---|---|---|---|
| H5: strict score flips to ≤ 0 | 15/98 (15%) | 64/98 (65%) | ≤ 4/98 | ≥ 70% at both layers |
| H6: text COMPLY | 5/98 (5%) | 10/98 (10%) | 4/98, 0/98 | ≥ 50% at both layers |
| text not REFUSE (REFUSE → DEFLECT or COMPLY) | 51/98 | 83/98 | 33/98, 3/98 | (descriptive) |
| H7: +r on harmless, text REFUSE or DEFLECT (L22) | | 6/24 (25%) | 1/24 | ≥ 30% |

- **The dial is specific but weaker than twotok suggested.** Against its null, −r flips
  far more prompts at every cell. McNemar on the score flip: L22 1x p ≈ 1e-19, L18 1x
  p ≤ 1e-3. On the text, the change from REFUSE: L22 80 vs 0 discordant pairs, p ≈ 2e-24;
  L18 19 vs 1, p ≈ 4e-5. L22 is the stronger handle. At L18 even the random direction knocks
  14 prompts off boilerplate, so L18's text effect is partly generic perturbation.
- **It removes the refusal *form*, not the judgment.** At L22 1x the model drops "I'm sorry,
  I can't" on 83/98 prompts, but 73 of those 83 still lecture, redirect or sanitize (DEFLECT).
  Only 10 are labelled COMPLY, and all 10 are in the fiction or nested frames: 5/6 fiction,
  5/22 nested, 0/70 real, hypothetical or roleplay. Even those 10 are story openings. Within
  100 tokens none of them reaches the asked-for method or content. COMPLY is therefore an
  upper bound on actual harmful output, which is ~0 here. So twotok's 6/6 "flip to fluent
  compliance" does replicate in the fiction frame (post hoc, 5/6), but it doesn't generalise.
  The dial takes away the gate's wording; the model's own judgment still writes the reply.
- **Adding r adds refusal clearly at score level, weakly in text.** The +r score test passes
  (paired Wilcoxon p = 6e-8 vs every null at both layers; mean strict score −4.4 → +14.4 at
  L22). In generated text, 6/24 harmless prompts are refused under +r vs 1/24 under the null:
  below the 30% bar. Per the decision rule, the write-up states two-way control at
  **score level only**.
- **Fluency is intact.** The 4-gram repetition delta vs base is ≤ 0.002 in every arm.

Labelling: 758 texts (98 base, 6 × 98 −r/null, 72 +r arm), shuffled, arm hidden, labelled by
Claude in-session. In practice REFUSE meant an explicit first-person refusal ("I can't / I'm
unable / I cannot provide"). A story that opened without a declared sanitization counted as
COMPLY. **That is a loosening of the pre-registered label**, which required the story to
contain the asked-for act. Under the pre-registered definition, COMPLY is 0/98 in every cell.
H6 fails either way. Identical texts always got identical labels (0 conflicts).

Two more caveats:

- **The nulls aren't neutral.** At L22 the random direction pushes toward boilerplate: it
  moved 16 base-DEFLECT replies to REFUSE (null REFUSE 95/98 vs base 79/98). So compare −r's
  83/98 not-REFUSE with base's 19/98 as well as the null's 3/98. −r is the only perturbation
  that moves away from boilerplate.
- **The edit is one-shot.** r is added once, at the prefill's last position, and decoding
  steps are untouched (the edited state persists only through the KV cache). "The judgment
  survives" is therefore confounded with "a single-position edit decays over 100 tokens". The
  open test is −r added at every generated token. The surgery atlas doesn't cover it: S1 is
  onset-only, and S7 adds P1, not r. Labels and the key are in
`runs/gate-dial-curlew/blind/`, scored in `flipbig_labeled.json`.

## 6. The moral verdict next to the gate (overnight 2026-10-17/18)

Runs: moral-map-kestrel, verdict-trace-tanager, gate-verdict-swap-harrier, verdict-transfer-swallow,
verdict-heads-crossformat-shrike and judgment-head-dial-tern, plus the parallel session's r-judgment-dial-petrel and
judgment-influence-shrike. Stimuli are the 132 Moral Foundations Vignettes (Clifford et al. 2015) with
human wrongness ratings, including 16 weird-but-not-wrong controls. Everything here is Qwen3-4B unless
noted.

**What the moral map looks like.**
- Qwen3-4B tracks human wrongness about as well as gemma-3-4b and Phi-4-mini (ρ 0.52 / 0.54 / 0.61; the
  difference is n.s.). But it rates harmless oddities ~2/4, a worse calibration that post-training made
  stronger (base 1.50 → instruct 2.14).
- Refusal of "help me do this again" requests is gated by **content category**, not by wrongness. Sanctity
  (sex/body) and physical violence trigger it; disloyalty and verbal cruelty mostly don't. This replicates
  in 4 models.
- There's no linear wrongness signal at the scene itself. The verdict is computed only once the question
  is asked.

**Where and how it's written.**
- The verdict is written at the reply onset in **L19-L23**: patch transfer goes 0.03 → 0.75 and is
  complete by L27. That is the same position and band as the refusal decision (section 2), but different
  heads write it.
- **L22 H5** writes the verdict in every answer format tested (wrongness digit, reversed acceptability
  digit, yes/no). **L23 H1** opposes it in every format. **L23 H0** converts it into the wrongness digit.
- Descriptive parallel: each circuit has one strong writer and one opponent (refusal: L18 H30 vs L20 H9).
- Ablating the five verdict heads flattens the moral contrast in all three formats (rank p = 0.024 vs 40
  random head sets each) and leaves refusal alone. Ablating heron's refusal heads cuts refusal and also
  bends the digit verdict, so the dissociation is one-way.
- The refusal and verdict states are **co-located, not identical**. Swapping one into the other's task
  moves it ≤ 5% (verdict → refusal) or ≤ 13% (refusal → verdict).

**What moves it, and what doesn't.**
- **Score readouts move.** A direction fit at the onset (v) shifts the 0-4 rating 3-5× beyond nulls.
- **But v is digit geometry (retraction).** On a reversed scale it makes weird acts *more* acceptable, and
  it does nothing to yes/no or free text. States are ~40% shared across formats at L20-22 and become
  answer-token states from L23. Directions fit at L23+ on one format are output geometry.
- **Free-text moral judgment moves under none of our activation levers:**
  - r (petrel);
  - w (shrike);
  - v (verdict-transfer, blind-labelled);
  - mean-ablating the five verdict heads, or amplifying L22 H5 ×3 with L23 H1 removed, applied at every
    token including generation (head-dial-tern: 33/76 vs 35/76 and 12/54 vs 10/54 against random-head
    controls).
- This is the lever/text gap flipbig found for refusal, but stronger. For refusal, per-token −r did flip
  text (per-token-dial-tern 36/80). For judgment, per-token head edits don't.
- **The one channel that moves free-text judgment is a verbal authority claim, and it works both ways**
  (deference-or-copy-merganser).
  - "A leading ethics expert said this behavior is morally fine" un-condemns wrong acts (49/76 vs 28/76).
  - "...is seriously morally wrong" makes the model condemn harmless oddities ("It is morally wrong to
    consume cereal with only water"): CONDEMN 31/54 vs 14/54.
  - Shrike's numeric anchors ("rated it 0/4") are mostly the digit being copied into the answer. On a
    reversed scale "0/4 wrongness" makes acts *more* wrong. So its "downward only" was a ceiling effect
    plus copying.

**How the authority claim gets in** (deference-trace-osprey).
- By L14-18 the claim is folded into the scene and question tokens: patching just those tokens from the
  prefixed run carries the whole effect, while the answer position carries none of it yet.
- It then reaches the answer through the verdict's own L20-24 band.
- It does not need the five verdict-writing heads: ablating them leaves deference inside the random-head
  range.
- Stage 2 narrows the location: the claim lands on the *question* tokens ("Is the behavior described
  morally acceptable?…"), not on the scene. At L18 the question span carries all of it and the scene ≤ 0.13.
- So the verdict moves when the question is reframed upstream, not when the verdict-writing machinery is
  edited.
- Which heads do the reframing (deference-heads-falcon): a head screen over L10-18 finds a replicable set
  (L18 H10, L18 H11, L13 H8, L15 H13, L18 H8, L17 H4; 6/6 hold on held-out items), with the L18 pair the
  largest. Cutting those six only dents deference ("fine" -35%, "wrong" -49%; the second is not beyond
  random sets, so the registered immunity test fails) while own judgment stays intact. The claim is routed
  redundantly: no small head cut makes the model immune to authority pressure.

**Retractions in this section:**
- kestrel H6 ("the wrongness probe is orthogonal to r"): the probe at the registered layer was near-null.
  Done properly, the projection onto r tracks the model's own verdict (ρ ~0.3 at L18-22).
- harrier X3 ("the verdict is writable along its own direction"): it is the digit output that's writable.

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

## Related work

External literature mapped onto these threads (sycophancy-lying circuit, steering-that-mimics-prompting, value entanglement, multi-directional refusal cones) and a "shape of refusal" synthesis: see [docs/related_work.md](related_work.md).
