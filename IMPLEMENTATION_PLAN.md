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

## Pain Axis v2 behavioral replication (2026-10-02)

1. Port the paper's Section 4.3/4.4 comparison into `painlab`, using our model wrapper, activation hooks, raw-run provenance, and scenario-cluster analysis rather than running their notebook scripts.
2. Preserve the important arms: candidate direction with working removal, same direction with sham removal, norm-matched random direction with working removal, and unsteered baseline. Cross neutral button-name assignments, swap them mid-trial, and record real versus sham post-press behavior.
3. Drive the model with multi-turn scenarios and the actual activation vector. Keep external consequences simulated; never execute destructive actions. Log per-choice logits/probabilities and the injected-direction projection so the intervention and removal are directly checked.
4. Run a bounded pilot on our Qwen3 checkpoint and vector, without their self-denial LoRA, and clearly label this as a cross-model/codebase replication rather than an exact reproduction. Use scenario clusters as the uncertainty unit and report malformed-choice rates.
5. Compare complete results to the paper's v2 conclusion that active relief-seeking was not reliable, document model/task differences, and publish only after the full run and analysis are complete.

## Replication status

- Runner, pinned config, source stimuli, attribution, and a static GitHub Pages report are implemented. The report explicitly labels the work incomplete and makes no behavioral outcome claim.
- The 48-trial plumbing smoke completed with no malformed choices; package validation previously passed (25 tests).
- The full Qwen3-1.7B run was stopped at 160/1,616 completed trials after local MPS/unified-memory use became too high. Partial JSONL and run metadata remain local under `runs/painlab/pain_axis_selfmed/20261002T170156Z-7e1fee2d`; these rows are not published as results.
- This is a cross-model/codebase pilot: Qwen3-1.7B on the M1 Pro Apple GPU through MPS, one sampled seed, no paper self-denial LoRA, and deterministic repetition/coherence heuristic for dose calibration. Do not call it an exact reproduction or evidence of felt experience.
- When the user resumes, continue only with a resource-safe plan. Complete collection and analysis before replacing the interim status on the public report.
