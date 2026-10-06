# Frozen adapter evaluations

Training loss alone does not establish that an adapter understands its research
domain or retains general capabilities. Every provisioned training job now seals
an independent evaluation suite into the private dataset repository, separately
from the corpus, with a pinned commit and canonical SHA-256. The job manifest
records the suite identity, version, scoring policy and thresholds. A changed
suite file cannot silently alter a prepared run.

The bundled `evals/consciousness-smoke-v1.json` contains sixteen original tasks:
eight domain-methodology checks and eight elementary general-capability checks.
They are **engineering smoke checks, not a scientifically validated benchmark**.
They have not been reviewed by an expert panel, psychometrically validated, or
calibrated against external evaluations. Passing them is a deployment gate, not
evidence of domain mastery, pain, sentience or consciousness. Before interpreting
Chamber results, an operator should freeze a larger independently reviewed suite
and retain a separate untouched test collection.

## Inputs and source separation

Each suite has source-family identifiers, source URLs, rights evidence, license
verification and provenance. Each question references those sources. The bundled
task text is original repository-authored material assisted by a coding agent;
it does not copy paper passages or Chamber prompts. Observatory contributors
dedicate this original task text to the public domain under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

Reserved evaluation families and URLs are excluded from both corpus training and
validation. Exact normalized question text is also rejected, including in
instruction examples. Source records and duplicate-source lineage are checked in
the GPU worker as well as during curation. These safeguards cannot identify every
paraphrase, translation, unknown mirror or contamination already present in the
upstream pretrained base. Never report complete decontamination on this basis.

The corpus also excludes identified Chamber stimuli. A domain corpus can contain
research about experiments; it must not teach the model the evaluation questions
and preferred answers that will later be used to judge it.
This includes the current persona, wild and self-reading prompt literals and the
external reading URLs/excerpts in `live/press.json`. Research and training images
carry the same read-only exclusion inputs. GPU preflight checks original rows,
instruction messages, recorded provenance and duplicate lineage before loading
the model. These checks identify recorded URLs and normalized literal passages;
they do not detect every transformed or previously unseen experimental input.

New snapshots use `consciousness-corpus-v3`, including research-area and claim-basis
classifications. The coordinator and provisioned worker require a ready quality
and perspective-coverage gate. Compatible archived `consciousness-corpus-v2`
snapshots retain their strict original contract; they are not silently relabelled
or rejected solely for their version. Earlier unreviewed snapshots cannot bypass
these gates: review their sources and seal a current snapshot before training.
See [corpus quality](docs/corpus-quality.md) for legacy review and coverage rules.

## Comparison and publication gates

The same frozen questions are scored at three points:

1. The pinned unadapted base, with the PEFT adapter temporarily disabled.
2. The incoming parent adapter before training; on the first run this is the new
   adapter's initialized state.
3. The trained candidate adapter.

The worker keeps one loaded base model. It does not allocate a second 70B copy.
Scoring uses evaluation mode, inference without gradients, batch size one and
mean token log likelihood of each answer continuation. The best scoring choice
is selected deterministically; no paid judge model or sampling is involved.
All three phases use the same plain completion format, even for SFT, so a changed
chat template does not change the evaluation input. Answer wording, length and
tokenizer boundaries still affect this scoring method. It is not an open-ended
reasoning or conversational quality evaluation.

The defaults require at least 50% accuracy in each small smoke group and no
accuracy or correct-answer negative-log-likelihood regression against either the
unadapted base or incoming adapter. The corpus held-out loss default now also
requires non-regression (`training_max_loss_ratio=1.0`). Strict small-suite gates
can reject a useful adapter because of task or metric noise. Operators may change
predeclared tolerances; they must record the change before the job, not relax a
threshold after seeing the candidate. Confidence intervals and statistical power
have not been established for this smoke suite.

Results save per-group and per-item measurements, both controls, candidate
scores, thresholds, reasons and suite hash in `evaluation/summary.json`. Source
provenance and limitations are saved in `evaluation/provenance.json`. The model
artifact does not copy the operator's private question or answer text. Report
identifiers and numerical scores may still be visible if the operator publishes
the model artifact.

Publication and activation require a measured, passed evaluation matching the
manifest hash as well as the training checks. Continuing from an older adapter or
starting SFT also requires its passed independent checks. Chamber task engagement,
fresh representations, dose calibration and experiment controls remain separate.
Preserve an unadapted Chamber subject as an independent experimental control;
the training evaluation's disabled-adapter control is not a replacement for a
matched Chamber intervention experiment.

## Operator-owned suites and longer context

Set `training_eval_suite_path` to a local JSON suite with the same schema and set
`training_eval_expected_sha256` to its canonical hash. A custom path without a hash
is rejected. Compute the hash locally without a model or provider call:

```powershell
.venv\Scripts\python.exe -c "import json; from pathlib import Path; from observatory.evaluation import validate_suite,suite_hash; suite=json.loads(Path('my-frozen-suite.json').read_text(encoding='utf-8')); print(suite_hash(validate_suite(suite)))"
```

The coordinator uploads the suite privately and passes its pinned revision to the
worker. A suite supports 4–256 multiple-choice items, with at least two per group
(`domain_understanding`, `general_retention`). An expanded suite should cover
contrary theories, source attribution, methodological reasoning, uncertainty,
domain transfer and retained general skills using separately reviewed sources.
Freeze new versions explicitly; preserve old suites for comparisons and watch
for repeated-use benchmark adaptation.

Settings include `training_eval_max_accuracy_drop`,
`training_eval_max_nll_ratio`, `training_eval_min_domain_accuracy`,
`training_eval_min_general_accuracy`, and `training_eval_max_length` (1024 by
default). Inputs that exceed the declared evaluation length are rejected rather
than silently truncated. Custom suites incur extra forward passes, so measure
their runtime and GPU footprint before scheduling frequent 70B jobs.

`training_sequence_length` remains independently configurable (2048 by default).
The worker rejects lengths beyond the model's positional capacity. A larger
context can retain longer scientific arguments but increases activation memory
and runtime; the presence of a long context in the base model does not establish
that a single-GPU QLoRA recipe fits it. Benchmark GPU fit and extraction quality
before raising the production sequence length. No paid 70B training or long
context hardware benchmark has been performed by these local tests.
