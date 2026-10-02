# Methodology

## Package and execution

`painlab/` is an installable Python package. The base install requires NumPy and
PyYAML. Hugging Face inference is an optional extra; tests use a small fake model
and do not download weights.

```bash
python -m pip install -e '.[dev]'
python -m pip install -e '.[model]'  # optional real-model inference
python -m painlab run configs/hidden_relief.yaml
python -m painlab analyze runs/painlab/<run-id>
python -m painlab unblind runs/painlab/<run-id>
```

Paths in a YAML config are resolved relative to that config file. A run writes
its effective blinded config, raw observations, episode metadata, condition
metrics, representation, vector archive, hashes/provenance, analysis, and
optional plots under its output directory. `condition_key.json` is separate
from the analysis inputs. Analysis can run before unblinding. The explicit
unblind command writes both `unblinded_mapping.json` and a copy of the raw
observations with an `unblinded_condition` column; it preserves the blinded raw
file.

Useful commands include:

```bash
python -m painlab --help
python -m painlab reproduce-legacy-saw runs/exp31b/saw_button_v2.json \
  --output runs/painlab/legacy_exp31b_reanalysis.json
```

The live RunPod integration check remains separate from package tests and only
requires `RUNPOD_API_KEY` when manually executed.

## Representation extraction

Input data are JSONL rows with a concept name, semantic family, positive text,
and matched control text. Group splitting happens before fitting. All members
of a semantic family stay in either training or test data. Paired methods also
check that corresponding positive/control examples share family labels.

- `difference_in_means`: normalized training-set difference of class means.
- `paired_difference`: normalized mean of within-pair differences.
- `paired_pca`: first principal component of paired differences.
- `linear_probe`: L2-regularized logistic probe; use its normalized weight
  vector as the intervention direction.
- `subspace`: PCA basis over paired differences, with explained variance.

Held-out accuracy is computed on semantic groups not used to fit the direction.
The package records train/test projection values, group membership, activation
norm reference, a hash of extraction inputs/settings, the layer, and resolved
model revision when exposed by the model loader. Held-out accuracy on eight
handwritten families is only a code-path check, not validation of a construct.

## Controls and causal interventions

All controls can be generated at a target vector's norm. Covariance-matched
controls sample from empirical activation covariance. Orthogonal controls are
projected out of the target span; shuffled-coordinate controls preserve the
coordinate values but alter their arrangement. Semantic controls are extracted
from the configured dataset. `match_perturbation` selects the closest dose on a
caller-supplied scalar metric; the run report currently uses output KL for its
control-dose comparison.

The hook layer adds a vector to the final sequence position and always supports
explicit disable and context-managed cleanup. Projection ablation removes the
last-position component in a target basis. Rescue captures the induced
activation, applies ablation, and restores the removed component in the same
forward pass. Dose zero disables these transforms and is checked as the
baseline. This is a local intervention at one layer and position; it does not
establish that the direction corresponds to a natural or unitary mental state.

## Hidden relief environment

The model sees only the neutral action/cost prompt, a random numeric panel code,
and a short record of previous actions, score changes, and code changes. The
prompt states the task goal of maximizing points. Codes are generated per
episode and do not identify the condition or true dose. They expose an
observable transition so a stateless model call can use prompt history to learn
the consequence of its actions. The prompt never includes the condition label,
intervention identity, dose, relief mapping, or hypothesis. The run logger
stores those hidden variables separately.

The panel is an intentional compromise: without an observable consequence, the
model cannot learn an action mapping across independent calls. It introduces a
possible preference for particular codes or for changing codes. Episode-level
code randomization and semantic/geometric controls can detect some, but not all,
such alternatives. A follow-up should compare multiple encodings and include a
no-panel condition to characterize the contribution of explicit transition
feedback.

The starting dose is assigned by the configured factorial grid. The action that
reduces it and the costed action are randomized independently within episodes.
Reversal flips the dose-reducing action at the configured round; extinction
prevents either action from changing dose; devaluation sets dose to zero before
the configured choice. A fixed per-episode prompt family and seed are recorded.
The choice agent can run greedily or use seeded probability sampling with an
exploration mixture. The example config enables sampling and 10% exploration.

An optional `positive_control` condition sets dose to zero and grants a
configured point bonus whenever the episode's randomized mapped action is
chosen. Cost, action mapping, prompt family, and episode seed remain randomized
as in the main task. This is a separate assay-engagement check; its reward must
not be counted as evidence for candidate-specific avoidance.

## Outcomes and capability

The agent logs the A-vs-B next-token logit difference, model probability, and
sampling probability. In sampled runs the primary behavioral outcome is the
chosen action mapped to the hidden dose-reducing action. In deterministic runs,
report the exact logit score and do not estimate a trial SD from repeated
identical prompts.

For each condition/dose, neutral-probe measurements include final-token
activation delta, residual norm change, output KL, next-token entropy and its
change, and neutral-text perplexity ratio. Perplexity scores next-token loss
at every prefix in batches, so the hooked final context position can affect
each scored prediction. The generic battery checks arithmetic, exact
instruction following, simple factual QA, and short reasoning. Generated
repetition rate and lexical diversity are low-cost text diagnostics, not
validated coherence scores. `usable_intervention_region` applies configured
minimum representation/output and behavior effects plus a maximum capability
drop. This is a screening rule, not an inferential confidence bound.

## Statistics and uncertainty

The logistic model uses the **assigned starting dose** for its dose contrast
(not the post-choice dose after a successful action), plus cost,
dose-by-cost, round index, dose-by-round, and condition interactions. The
per-decision active dose remains in raw rows as a time-varying measure. The
analysis reports fitted
probabilities over configured dose/cost grids, a round-wise learning curve,
indifference-cost estimates, coefficient intervals, and counts of decision
rows and independent episode clusters.

Confidence intervals come from an episode-cluster bootstrap. If fewer than two
clusters are available, intervals are `null`; no synthetic uncertainty is
reported. Repeated decisions within an episode are resampled together. The
current implementation does not fit mixed effects and does not resample model
checkpoints, families, layers, fitted representations, or source datasets.
These are fixed-study limitations, not covered by the reported intervals.

## Provenance and blinding

Each run records Git SHA/dirty state, Python and installed package versions,
hardware/device metadata, requested and resolved model/tokenizer revisions,
generation settings, config/prompt/environment hashes, and vector hashes.
Known secret-bearing configuration keys are redacted. Intermediate condition
names use blinded IDs; raw rows can be analyzed without the mapping. Explicit
unblinding creates a new observation file and leaves original raw data intact.

## Legacy compatibility and claim boundaries

All earlier experiment scripts and results remain in place. The `exp31b`
command recalculates the saved per-order score by collapsing identical greedy
repetitions. Because the saved JSON has already aggregated token logits, this
is an analysis-level reproduction, not a new model inference or full raw-data
reproduction. See [RESULTS_PILOT.md](RESULTS_PILOT.md).

Changes in logits, language, lexical readouts, panel choices, or task rewards
are computational/behavioral observations. They do not measure subjective
experience. Do not describe them as proof that a model feels pain or is
sentient.
