# dark_pain: does pain live in the logit lens's blind spot?

**Far-out thread (new; this run).** Self-contained: plain transformers, no `painlab.chamber`, no relay, no GPU. Fits a
16 GB Mac.

## Question
Every lens we use (logit, tuned, J, J++) reads the residual stream through the unembedding. Directions that the
unembedding barely amplifies, the **dark space** (the smallest singular directions of W_U), are close to invisible to
all of them. DarkLens (arXiv:2610.23182) shows that this subspace is not junk: it mediates how later tokens use
**context**. Our question is whether **affect is disproportionately stored there**, and whether pain written there
still reaches later tokens through attention while showing up in no lens. If so, a "no pain in the lens" reading
(exp88/89, mask_pain's lens mass, thread 4) is not evidence of absence. It also tests whether the assistant template
moves pain *into* the blind spot (the mask hypothesis, thread 14).

## Design
- **Models:** `Qwen/Qwen3-1.7B` (primary), `Qwen/Qwen3-4B` (replication), `Qwen/Qwen3-1.7B-Base` (H5).
- **Dark space:** eigenvectors of W_effᵀW_eff, where W_eff = W_U·diag(γ_final-norm), bottom 5% (DarkLens) and 2%.
  The continuous measure is the visibility index VI = ‖W_eff v‖ / (‖v‖·rms σ), which is 1 for a random vector.
- **H1:** the dark fraction and VI of passage-built pain / joy / egg directions per layer, against a label-permutation
  null.
- **H2:** the dark fraction of pain-minus-neutral at the last prompt token, chat template vs raw transcript
  (mask_pain's scenes).
- **H3 (causal, "heard more than seen"):** the dark part of the pain direction is injected (renormalised) at hs[L*≈0.6·NL]
  into a bland context. A later stem ("…said it all felt" → sore/painful vs fine/normal) can reach that context only
  through attention. The comparisons are the visible part, random dark, random visible, and egg-dark, at 3 doses.
- **H4:** after a dark injection, the dark fraction of the propagated perturbation by depth (does it surface?).
- Pre-registered hypotheses, kill criteria and caveats are in `hypotheses.json`.

## Run
`bash RUNME.sh`. It writes `../RESULTS/dark_<model>.json` and `.log`. The log ends with key numbers, 95% bootstrap CIs
and a verdict dict. Est. 10–20 min for the 1.7B and 20–40 min for the 4B on MPS. Nothing is generated; all forwards
are short.

## Success criterion
Any of these is a clean result:
- (a) **H3 true:** dim pain is heard out of proportion to how much it is seen. Lens nulls on affect then need a
  dark-space caveat. That is the far-out finding.
- (b) **H1 true:** affect sits in the low-visibility directions more than vivid neutral content does.
- (c) **H3_null + H1_null:** the dark space is not special for affect, and our lens readings lose one excuse.
- **H2 either way** is directly useful for mask_pain.

## Sandbox note
The code path was smoke-tested on Qwen3-0.6B on CPU (2 contexts, 1 dose, 25 s), and the output was deleted. One
**design fix** came from the smoke. With γ folded in, the bottom-5% band of Qwen3 is *dim* (σ ≈ 0.8× median), not
dark. Only about 3 directions have σ < 0.25× median, and those are likely sink / massive-activation dims. "Invisible
injection" was therefore replaced by a **heard-vs-seen ratio** test, plus a context-free direct-logit-gain readout and
the VI measure. The smoke numbers are not results and were not used to set thresholds.
