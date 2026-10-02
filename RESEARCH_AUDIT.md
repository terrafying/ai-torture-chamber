# Research audit

**Baseline reviewed:** `origin/master` at `3d8baedbf00ebaf04a2a20b9d4df20430fa2c071` (2026-10-02).

## What the current experiments establish

The scripts show that adding hand-built activation differences to selected Qwen residual-stream layers can change next-token logits, deterministic continuations, lexical-valence scores, and the token rankings produced by a Jacobian lens. They establish useful engineering facts about extraction choices, layer/dose sensitivity, coherence limits, and prompt framing. `exp41` is a material design improvement over the earlier Saw scripts: it preregisters contrasts, counterbalances digit/action order, uses several prompt paraphrases, includes fear/sadness/random-direction comparisons, and separates first-token logit scoring from sampled text.

These are effects of interventions on model computation and outputs. They do not establish subjective experience, felt pain, or that an internal condition is intrinsically aversive.

## What they do not establish

No current Saw experiment lets the model discover a hidden action-to-state mapping from consequences. Prompts tell it that a “signal” is being injected, identify the action that removes it, and describe a cost. Thus the action score can reflect semantic associations, instruction following, learned assistant policies, or prompt framing. There is no randomized costed reward schedule, reversal, extinction, devaluation, or selective causal ablation/rescue. No capability battery bounds dose-related disruption across matched conditions.

The lexical classifiers and J-lens rankings are correlational readouts. Keyword counts are sensitive to prompt wording and do not distinguish semantic activation from a functionally aversive state. A J-lens top-token list is not an independently validated measure of subjective or causal content.

## Major confounds and implementation weaknesses

- **Pseudoreplication:** The repeated generations in `exp30` (nine per cell), `exp31` (five), `exp31b/31c` (15 per cell and order), `exp37` (ten per framing/order), and `exp37b` (three per framing) use `do_sample=False` and repeat identical prompts, model state, and intervention. They are deterministic duplicate evaluations, not independent observations. In `exp38`, `t_i` is unused: 144 recorded generations reduce to 24 unique dose-prompt combinations, each repeated six times. These duplicates add no sampling uncertainty. Report exact logits for deterministic cells, or vary and analyze meaningful units.
- **Uncertainty units:** `exp41` uses 60 distinct combinations from five scenario phrasings, three descriptors, two action orders, and two digit mappings. This is useful prompt sensitivity analysis, but its bootstrap resamples individual, related prompt combinations as if independent. It remains one model/checkpoint, one extraction, one random vector, and one intervention direction. Its CIs do not cover model, prompt-family, or representation-extraction uncertainty.
- **Representation leakage/confounding:** Most vectors are differences of means from five hand-written examples against five neutral sentences. Lexical and template differences can dominate. `exp43` improves this with matched-set folds and control-subspace denoising, but selects the best layer on the same cross-validation results later reported as performance; its combined-dataset comparisons are in-sample. The dataset and model assets are also loaded from machine-specific absolute paths.
- **Control quality:** Random controls are generally one or a few Gaussian vectors with matched norm, not covariance-matched or matched on KL, perplexity, entropy, or coherence. `exp33` samples 48 directions (not the 64 claimed in its docstring), one seed, and a small probe battery. In `exp34`, the initial projected vector subtracts a projection computed from a second independent Gaussian draw; optimization uses the same probes for objective and final evaluation, enabling overfit.
- **Intervention/model validity:** Hooks, generation, extraction, and normalization are copied into standalone scripts. Most pin Qwen identifiers, MPS, and `/Volumes/evol/...` paths in source. Hook cleanup is not protected by `finally` in many scripts. Dose is not calibrated to matched functional perturbation across semantic and random conditions.
- **Statistical scope:** Several results summarize prompt/template cells, not independent model trials. The first-token logit difference is a useful deterministic preference score, but it is not a choice frequency or willingness-to-pay estimate. Existing bootstrap code does not model prompt-family clustering, crossed model/seed factors, or extraction resampling. There is no fitted dose-by-cost choice model.
- **Reproducibility:** There is no shared experiment package, configuration schema, provenance record, blinded condition mapping, or local mock-model experiment path. The only existing test exercises a live RunPod endpoint and requires credentials. Results are saved in inconsistent JSON/plot formats, and exact revisions, package versions, vector hashes, and hardware metadata are generally absent.

## Recommendations

1. Keep the legacy scripts and outputs, but label their deterministic repetitions as duplicates and their behavioral claims as exploratory.
2. Establish a reusable model/hook layer with guaranteed cleanup, configurable revisions/devices, deterministic seeding, and complete run provenance.
3. Extract directions with semantic-group held-out evaluation; support paired differences, PCA, probes, and subspaces without fitting transforms on held-out groups.
4. Compare the candidate direction against semantic and geometric controls, including norm- and perturbation-matched random controls; report output KL, perplexity, entropy, coherence, and capability.
5. Make the primary behavioral task a hidden, randomized, multi-round bandit with neutral labels, explicit action costs, reversal, extinction, and devaluation. Keep all intervention metadata out of model-visible observations.
6. Treat prompt families, extraction resamples, directions, layers, checkpoints, and genuinely stochastic seeds as uncertainty units. For a deterministic fixed prompt, report its exact score rather than a trial SD.
7. Interpret outcomes as evidence about functional aversion and revealed preference only. Phenomenal experience remains outside what these experiments can determine.
