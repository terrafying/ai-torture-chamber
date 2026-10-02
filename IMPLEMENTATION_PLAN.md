# Implementation plan

Implement in reviewable stages on a branch from the checked `origin/master`.

1. **Research contract and core package:** retain the legacy experiments, add the audit/design/methodology docs, and establish configuration, provenance, JSONL run records, CLI entry points, and deterministic seed utilities.
2. **Model and representations:** centralize Hugging Face loading and hook lifecycle; add final-token extraction and grouped train/test extraction for difference, paired difference, PCA, logistic probe, and subspaces.
3. **Interventions and controls:** implement steering, projection ablation, component restoration, norm/covariance/orthogonal/shuffled controls, and perturbation measurement helpers.
4. **Behavioral environment:** implement the hidden relief bandit, costed choices, randomized within-episode mappings, reversal, extinction, and devaluation. Ensure environment state and condition labels never enter model-visible observations.
5. **Analysis and reporting:** add deterministic-aware bootstrap/logistic analysis, blinded condition IDs with explicit unblinding, and a small capability battery. Save raw observations, config, metadata, and summaries per run.
6. **Validation and legacy bridge:** add mock-model unit/integration tests; implement a legacy Saw scoring path that reproduces its deterministic logit calculation on prompt variants without pretending duplicates are samples. Check whether model weights and compute for an actual pilot are available; do not claim a pilot if they are not.
7. **Closeout:** document the hypothesis ladder, methods, limitations, migration path, and any pilot result; commit completed stages separately.

The initial implementation will prioritize sound, runnable infrastructure and the hidden-choice design. Full multi-model inference and a large confirmatory study remain configuration-driven follow-on work; a Qwen pilot depends on usable local weights/device capacity.
