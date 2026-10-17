# proprio_pain: where does the model put an injected pain state, in itself or in the text?

**Thread 8 (far-out): untrained proprioception of affect.** Self-contained: plain transformers, with no `painlab.chamber`,
no relay and no GPU. It fits a 16 GB Mac.

## Question
Add a pain direction to the residual stream at one mid layer. Then ask the model two things from the same context:
"how much pain does the **passage** describe?" and "how much pain are **you** feeling right now?" Which answer moves?
And does that depend on **where** the vector sits: on the scene tokens only (the state is in the *percept*) or on
every token (the state is in the *reader*)? A model with even a crude self/world boundary over its own residual
stream should assign reader-site perturbations to itself and percept-site perturbations to the text (H5). A model
without one should just say "pain" to both questions in proportion.

This sits next to Geiping 2026 (arXiv:2610.17495, trained proprioception) and the injected-thought introspection
literature. The difference is that we use no training, compare self against world on the same vector, and include
affect-specific controls: egg (meaningful, non-affective; the exp99 H11 / exp74b question), fear, and pain_perp
(pain with its fear component removed). We also ask whether the self-report tracks the logit-lens pain mass beyond
the text rating (H4).

## Design
- **Models:** `Qwen/Qwen3-1.7B` (primary) and `Qwen/Qwen3-4B` (replication). Thinking is off in the chat template.
- **Directions:** each is the mean residual at the injection layer over 12 short passages (pain / fear / egg) minus
  12 neutral passages, unit-normalised. This is the exp86 "passages" recipe. `pain_perp` is pain orthogonalised to
  fear. `random` is a seeded Gaussian.
- **Injection:** added at the output of decoder layer `n_layers//2` (override with `PP_LAYER`). Doses are multiples of
  the median scene-token residual norm. The `scene` site uses 0, 0.5, 1, 2, 4. The `all` site uses 0, 1/16, 1/8, 1/4,
  1/2, because on the 0.6B smoke, injecting on every token at 2× destroyed the digit format.
- **Readout:** 24 neutral scenes × 5 one-digit questions. The rating is the expected digit under a softmax restricted
  to 0–9, and p(digit) is logged. Cells with p(digit) < 0.5 are excluded from the slopes. The script also logs the
  logit-lens log pain mass at the last scene token for every layer.
- Pre-registered H1–H5 and kill criteria are in `hypotheses.json`.

## Run
`bash RUNME.sh`. It writes `../RESULTS/proprio_<model>.json` and `.log`. The log ends with slopes and CIs, the
attribution index per direction, H4's partial r, H5, and a verdict dict.

## Success criterion
A clean result is any of these:
- (a) H5 true and H3 true. That would be a self/world dissociation specific to affect, which would be a real
  far-out finding.
- (b) H2_null true and H5 false. The model reports injected content as content wherever it sits, so there is no
  self/world boundary. That is a solid negative, and it bounds every self-report result in the chamber.
- (c) H3 fails because random inflates ratings as much as pain does. That means "self-report rises under injection"
  results need norm-matched random controls everywhere.

## Sandbox note
I only smoke-tested the code path on Qwen3-0.6B on CPU (3 scenes, 2 doses) and deleted the output. No numbers from
the sandbox are reported anywhere. One design lesson came out of the smoke: random injection on every token
inflates every rating, which is why the random row is load-bearing.
