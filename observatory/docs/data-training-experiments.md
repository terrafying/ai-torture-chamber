# From browser research to datasets, training and Chamber experiments

This guide follows the implemented path from a research worker's browser to a
revision-pinned adapter and an explicitly deployed Chamber subject. It describes
the data contracts and operator actions, including the places where automatic
processing stops. Run commands from the repository root unless stated otherwise.

The Observatory is an integration alpha. Collection, review, sealed datasets,
HF Jobs submission, adapter training, evaluation and pinned adapter loading are
implemented and covered by offline tests. A funded crawl, actual 70B CUDA job,
production browser integration and adapted Chamber deployment still require
acceptance on the owner's accounts and hardware. Preview records are authored
examples; their animations, receipts and checkpoints are not live measurements.

## 1. The services and the two model roles

| Component | Responsibility | Output |
| --- | --- | --- |
| `ResearchSupervisor` | Runs the six Browser Use specialties, directs disposable Chromium browsers with the configured frontier model, captures original text and checkpoints research memory. | `sources`, `evidence`, `notes`, screenshots, public next-action messages and memory checkpoints. |
| `AutomaticCurationWorker` | When explicitly enabled, reviews eligible original documents using two blind structured passes with the research model. | Immutable `curation_reviews` receipts and source-bound automated `quality_review` records. |
| `build_snapshot()` | Applies deterministic admission, duplicate, source-family, holdout, instruction-support and coverage rules. | Immutable dataset candidate and sealed manifest. |
| `TrainingCoordinator` | Checks readiness, pins and uploads inputs privately, submits HF Jobs and polls actual results. | Separate CPT/SFT `training_runs`, provider job identity, logs and checkpoint candidates. |
| `observatory.train_worker` | Loads the sealed job, updates PEFT weights on the pinned base, measures checks and optionally publishes the candidate. | Adapter, tokenizer, training lineage, checkpoints, evaluation and calibration artifacts. |
| `painlab` / optional live adapter bridge | Load a chosen pinned base plus adapter, extract fresh intervention representations and run controlled experiments. | Experimental observations and analysis, or a separately deployed Chamber worker. |

The **research brain** is an already-trained OpenAI/Anthropic-compatible frontier
model using x402 or an explicit direct API route. The **training subject** defaults
to pretrained Meta Llama 3.1 70B Base with a QLoRA adapter. They are separate models.
No starter document upload is needed: the brain can discover papers and articles
before any target-model adaptation exists. Browsing and training can run
concurrently; training consumes a sealed snapshot rather than a changing browser
queue. A newly discovered page therefore belongs to a later snapshot/job.

x402 funds compatible inference requests, including automatic reviews. It does
not grant reuse rights, fund HF GPU jobs or replace browser/GPU/Hub accounts.
See [payment operations](../X402_OPERATIONS.md) and
[the mission and agent briefs](research-mission.md).

## 2. Inspect and operate the interface

1. Inspect `observatory.html?preview=1&motion=1#research`. It demonstrates the
   browsing pane, saw traversal, selected passages and notebook deliveries using
   simulated records. No crawl, payment, corpus upload or training occurs.
2. Run the real sidecar using the [README startup instructions](../README.md).
   Connect the interface to `/api` when served by the sidecar, or the configured
   same-origin reverse proxy. Enter the owner token in **Operator setup**; it is
   held in page memory and must be re-entered after reload.
3. Configure one compatible research model, billing route and isolated browser.
   Set the research objective deliberately. Existing missions retain their saved
   objective; an updated default does not rewrite them. Keep training disabled
   during the first funded acceptance pass.
4. Start the mission. **Research** shows the selected agent's real captured
   viewport, actions and concise public observations. Visitors cannot scroll the
   agent's document. Highlight geometry and saw motion represent an explicitly
   selected visible passage, not the model's private reasoning or consciousness.
5. In **Evidence**, inspect rights, extraction, research areas, basis of claims
   and machine-consciousness perspective. Review exceptions manually, or enable
   **automated original-document curation** in setup and acknowledge its policy.
   The Evidence summary reports receipt decisions and interruptions.
6. In **Datasets**, create an immutable candidate, inspect included/excluded
   records and coverage, then use **Download sealed dataset** for the actual
   private CPT/SFT ZIP. **Download manifest** on the public page downloads
   sanitized metadata; it does not include the full training corpus.
7. Configure a digest-pinned GPU image and HF account/repositories. In
   **Training**, select a snapshot and **Original text · continued pretraining**.
   After a passed, published CPT run and separate instruction approval, select
   **Approved instruction examples · SFT** and its completed CPT parent.
8. In **Checkpoints**, inspect actual checks and calibration, record a validated
   selection and download its worker environment. Deploy that environment to a
   separate worker intentionally. Selection does not reload a remote endpoint.

Pause/Stop govern research; `training_enabled` separately governs scheduled
training. Stop a mission and disable scheduled training before shutting down if
both should remain stopped. Persisted running missions resume on service restart.
The sidecar runs one process/replica with a persistent database and encryption key;
it is not a distributed queue.

## 3. Original documents, evidence and notes are different records

The following are abbreviated shapes, not ready-to-submit sealed fixtures. IDs,
hashes, timestamps, rights evidence and source text are produced from actual
collection and review. Never fabricate them to satisfy a gate.

An original source preserves author-written text and collection provenance:

```json
{
  "id": "source-<content-derived-id>",
  "canonical_url": "https://publisher.example/original-paper",
  "title": "The original paper title",
  "text": "The extracted original paragraphs, headings and supported structure...",
  "content_hash": "<sha256-of-persisted-original-text>",
  "family_id": "doi:<doi-or-other-canonical-family>",
  "source_type": "article",
  "license": "unknown",
  "license_verified": false,
  "review_status": "pending",
  "extraction": {"method": "structured_html", "quality": "passed"},
  "provenance": {
    "collected_at": "<UTC-timestamp>",
    "method": "browser_dom",
    "content_type": "text/html",
    "extraction_scope": "article",
    "content_sha256": "<sha256-of-persisted-original-text>",
    "collector_version": "observatory-0.1"
  }
}
```

The collector also supports public PDF, HTML and plain-text document URLs. HTML
article-scoped metadata can identify specific CC-BY/CC0 reuse terms. A license
found on a surrounding publisher page is not automatically applied to the whole
page body. PDF extraction may remain unverified and discovery-only.

A supported observation creates a separate evidence record:

```json
{
  "id": "<evidence-id>",
  "source_id": "source-<id>",
  "passage": "A normalized passage matched to the collected source.",
  "support_verified": true,
  "verification": "literal_passage_match"
}
```

A note contains the agent's commentary and optional instruction draft:

```json
{
  "id": "<note-id>",
  "agent_id": "<mission-id>-scholar",
  "type": "observation",
  "text": "The author distinguishes a neural correlate from a sufficient condition.",
  "source_ids": ["source-<id>"],
  "evidence_ids": ["<evidence-id>"],
  "support_verified": true,
  "generated_by": "frontier_research_agent",
  "prompt_version": "research-v2-broad-consciousness",
  "review_status": "pending",
  "question": "What distinction does the author make?",
  "answer": "A source-supported draft answer for separate review."
}
```

A note without a matching source passage is a **lead**; its question/answer fields
are not retained as supported examples. A literal match verifies quotation
origin, not the truth of the commentary or the source's conclusions. Browser
inspection IDs additionally bind visible selections to the focused document and
capture. They do not automatically approve a training example.

**CPT uses accepted original document text.** It never substitutes the agent's
summary, next action, screenshot, hidden reasoning or animation. **SFT uses
separately approved supported messages.** Ordinary notebook observations do not
become instructions just because their note status is approved.

## 4. Admission: rights, extraction, research area and basis of claims

Every accepted original needs a canonical public URL, at least 200 characters of
original text, provenance, reuse evidence, eligible extraction and a recorded
approved quality review. The allowlist covers CC0/public-domain and the supported
CC-BY variants (`cc-by`, `cc-by-3.0`, `cc-by-4.0`); separately recorded permission
can satisfy the rights gate. Unknown licenses and unsupported licenses stay
quarantined. Social sources need separate permission even if other metadata is
present. Robot permission to fetch a page is not training permission.

Passed extraction can satisfy the fidelity gate. Unverified extraction requires
explicit owner evidence checking reading order, missing pages, tables and
formulas. A failed extraction is not rescued by a quality label. Optional offline
Docling processing still requires fidelity review. See
[extraction contracts](../EXTRACTION.md).

New quality reviews distinguish the subject from the argument's basis:

| Field | Values and meaning |
| --- | --- |
| `topic_relevance` | `relevant`, `unrelated`, `uncertain`; only relevant originals can enter training. |
| `topic_domains` | One or more of `machine_consciousness`, `consciousness_science`, `philosophy_of_mind`, `metaphysics_reality`, `religion_contemplation`, `welfare_ethics`. |
| `evidence_kind` | `empirical`, `scientific_theory`, `philosophical_argument`, `religious_contemplative`, `mixed`. Labels the basis of claims; does not certify truth. |
| `evidence_stance` | Machine-consciousness argument: `supportive`, `skeptical`, `uncertain`, `mixed`, `methodological`, or `not_applicable`. |
| `covered_stances` | Machine perspectives actually treated by a mixed/methodological document; no fabricated coverage. `not_applicable` requires `[]`. |
| reviewed `source_type` | `empirical_paper`, `theoretical_paper`, `review_paper`, `technical_report`, `article`, `reference`, `social`. Collection provenance and the original collection type remain unchanged. |

Philosophy, metaphysics and contemplative material can inform the corpus without
claiming empirical proof. A document with no machine-consciousness argument uses
`not_applicable` and cannot fill machine-perspective coverage. For new classified
sources, a scientific format also needs `empirical` or `scientific_theory` basis to
fill the scientific-source floor; `mixed` does not satisfy that floor by default.
Legacy reviews retain their original meaning and appear in explicit
`legacy_unspecified` coverage buckets, rather than receiving guessed labels.

### Manual review

Use the Evidence inspector's **Corpus quality** fields, recorded rights and
extraction checks. The authenticated API accepts the same review. This example
classifies an already collected, rights-cleared conceptual original; it does not
grant rights or alter original text:

```powershell
$apiRoot = 'http://127.0.0.1:8060/api'
$ownerHeaders = @{ Authorization = 'Bearer ' + $env:OBSERVATORY_ADMIN_TOKEN }
$sourceId = '<actual-source-id>'
$review = @{
  review_status = 'approved'
  quality_review = @{
    status = 'approved'
    reviewed_by = 'Owner reviewer'
    rationale = 'Reviewed the original conceptual argument and its limitations.'
    topic_relevance = 'relevant'
    topic_domains = @('philosophy_of_mind', 'metaphysics_reality')
    evidence_kind = 'philosophical_argument'
    evidence_stance = 'not_applicable'
    source_type = 'article'
    covered_stances = @()
  }
} | ConvertTo-Json -Depth 6
Invoke-RestMethod -Method Post -Uri "$apiRoot/admin/sources/$sourceId/review" `
  -Headers $ownerHeaders -ContentType 'application/json' -Body $review
```

Use actual owner review and evidence, not the example rationale. The backend
recomputes eligibility: an approved quality review cannot erase failed rights,
extraction, contamination or exclusion checks. Owner decisions take precedence
over later automatic results.

### Automatic original review

Enable `auto_curation_enabled: true` with
`auto_curation_policy_ack: "originals-v2"` in authenticated setup. Defaults are
off. An older v1 acknowledgment leaves the broader worker idle until renewed.
While the mission runs, the worker processes the shared source queue one review
at a time. It requires already verified rights and **passed** extraction; it does
not automate rights grants or approve unverified PDF fidelity. Documents over
60,000 characters remain for manual review rather than being truncated.

The primary and blind critic each receive the whole original and must agree on
acceptance and classifications, supported by 1–3 exact contiguous source quotes
(40–2,000 characters each). The same configured model performs both passes; these
are separate blind checks, not statistically independent judges. Disagreement,
uncertainty and unusable source evidence remain manual-review outcomes.

The accepted `quality_review` records `reviewer_kind: "automated"`,
`approval_basis: "owner_policy"`, policy version, source-content hash, policy hash,
model identity and its immutable receipt. The receipt binds both verdicts and
source identity. Public state shows concise decisions and rationale; full original
text, quotes and the receipt proof remain private. The GPU validator checks
automated receipt bindings again. Historical v1 receipts validate under their
exact schema and are never relabeled.
Here "private" means excluded from public state and restricted in corpus exports;
the configured review provider still receives the full document in its request.

Interrupted paid reviews retain progress. Reconcile any pending payment first;
then an idle worker and paused/faulted/stopped mission can use **Review recovery**
or `POST /api/admin/curation/recovery` with
`{"reviewed":true,"review_id":"<actual-interrupted-review-id>"}`. This explicitly
authorizes another attempt, which may incur a new charge; saved completed verdicts
are preserved and the mission does not automatically resume. See
[automatic curation](automatic-curation.md) for recovery and retry rules.

## 5. Sealing the corpus and inspecting the export

`POST /api/admin/snapshots` builds a candidate without contacting HF or allocating
a GPU. The new policy is `consciousness-corpus-v3`; compatible immutable v2
snapshots retain their archived contracts. Source eligibility is recomputed.
The recorded Chamber persona/wild/self-reading prompts and external reading
URLs/excerpts in `live/press.json` are excluded from training. Both research and
training images carry these read-only inputs; GPU preflight rechecks originals,
instruction messages and source lineage. New experimental inputs need to be
recorded/reserved as well; literal/identity checks do not detect every paraphrase.

- Exact and near duplicates contribute one representative original. Mirrors
  retain source lineage and reuse evidence; they do not multiply training weight.
  Changes in numbers, negation and mathematical relationships are conservatively
  retained as separate originals.
- DOI/arXiv/canonical-URL families keep versions and mirrors together. Initial
  deterministic assignment reserves approximately one hash bucket in ten for
  validation; explicit holdouts propagate. Stored family assignments persist.
  Already trained families cannot later be presented as fresh holdouts.
- Reserved evaluation families, URLs and normalized question passages, identified
  Chamber sources and literal stimuli, and explicit contamination flags are
  excluded. These checks cannot detect every paraphrase, translation, unknown
  mirror or upstream pretraining exposure.
- All distinct eligible originals are interleaved by reviewed strata and included
  once. The implementation has no enforced target quota for religion, philosophy
  or a particular theory. Inspect domain/evidence-kind counts and character
  shares; overlapping area labels can make area shares sum above one.
- Training coverage still requires supportive, skeptical and uncertain **machine**
  perspectives plus a scientific source. This is a presence floor, not equal
  sampling, statistical balance or validation of a theory. Broad originals can be
  accepted while the candidate remains unready for lack of those perspectives.

A CPT row contains `text`, `source_ids`, `family_id`, `split`, `content_hash`,
`synthetic: false`, `training_weight: 1`, the quality review, reviewed perspectives,
`representative_source_id` and complete `source_lineage`. It is the source text,
with an inspectable training card of metadata attached.

The sealed manifest records originals, supported SFT examples, source rights and
provenance, extraction/reviews, exclusions, duplicate lineage, synthetic policy,
coverage and the quality gate. `manifest_hash` seals that manifest; `corpus_hash`
tracks the document/instruction content and split identities. They serve different
purposes. A metadata-only change may create another manifest while leaving the
training content unchanged.

The owner-only `GET /api/admin/datasets/{snapshot_id}/export` verifies integrity
and returns:

```text
cpt/train.jsonl
cpt/validation.jsonl
sft/train.jsonl
sft/validation.jsonl
manifest.json
rights-provenance.json
exclusions.json
snapshot.json
hashes.json
```

`hashes.json` inventories every other payload file; it does not recursively hash
itself or the ZIP container. The export contains actual originals and private
review support, so store it as corpus material rather than a public UI artifact.
The HF job path consumes a revision-pinned JSON snapshot, not this ZIP; the ZIP is
the review/portable export of the same stage-separated records.

## 6. Separate supported instruction examples

Automatic original approval does not enable SFT. A generated instruction needs
individual `review_status: "approved"`, eligible source IDs, verified evidence
records and passages matched to the actual source. It must supply a valid
`messages` list, or a supported nonempty `question`/`answer` pair that curation
converts into user/assistant messages. At least one assistant target is required.

The owner must also enable `synthetic_training_approved` and record a nonempty
`provider_policy_reference` establishing that the intended teacher-output use is
permitted. That field records a decision; it is not a license grant. Paying a
frontier provider does not imply permission to train a competing model on its
generated answers. CPT on licensed author text is a separate data path.

Use the notebook's individual approval control or
`PATCH /api/admin/notes/{note_id}` with `{"review_status":"approved"}` after
review. The current note-create API accepts an owner observation, not a complete
instruction/evidence import. It cannot manufacture support by accepting arbitrary
`support_verified` flags. Agent-produced supported drafts provide the implemented
instruction path.

Accepted SFT rows carry `messages`, `source_ids`, `evidence_ids`, `family_ids`,
`split`, `synthetic: true`, `generated_by` and `prompt_version`. Duplicate message
lists are removed. A synthesis spanning train and held-out source families is
excluded. Examples based on held-out sources remain validation examples, and SFT
training cannot move a CPT holdout family into training.

## 7. Readiness, scheduling and HF job submission

Training starts disabled. Configure `hf_namespace`, `hf_dataset_repo`,
`hf_model_repo`, `hf_token`/server-side `HF_TOKEN`, and a `training_image` ending
with a complete `@sha256:<64-lowercase-hex>` digest. Repositories must belong to the
configured namespace. The owner needs gated Llama access, HF Jobs access/credits
and the chosen GPU flavor. Build `observatory/Dockerfile.training` independently
of the research sidecar and existing relay; obtain the registry's actual image
digest after pushing it.

The default base is `meta-llama/Llama-3.1-70B` pinned to
`349b2ddb53ce8f2849a6c168a81980ab25258dac`. Defaults use QLoRA, 2,048-token
sequences, 100 steps, rank 16, alpha 32, batch size 1, gradient accumulation 8,
learning rate `0.0001`, and a four-hour readiness interval. These are configurable
engineering defaults, not a demonstrated optimal 70B consciousness recipe.
Supported single-GPU profiles are `a100-large` and `h200`; actual fit and throughput
need hardware acceptance. Full-precision 70B LoRA requires a separately benchmarked
distributed recipe and is refused by this single-GPU path.

At a due scheduler check:

1. Validate training settings and credentials. The first enabled check can run
   immediately; subsequent checks use the persisted last-check time plus the
   configured interval. This is elapsed-interval scheduling, not a guaranteed
   wall-clock job every four hours.
2. Seal/reuse an eligible snapshot; require ready quality coverage, nonempty train
   and validation families, at least 20 original training documents by default,
   and no overlapping active job.
3. Reuse the prior run for unchanged stage content/base/recipe/evaluation identity.
   Notes, provenance changes or a new clock tick alone do not cause repeated CPT.
4. Resolve and pin the base/tokenizer revision, validate evaluation exclusion,
   and count real tokens with the pinned tokenizer before GPU provisioning. The
   default minimum is 50,000 training tokens. An estimate is not accepted.
5. Upload the snapshot and frozen evaluation suite into a **private** dataset
   repository, seal a job manifest and submit the digest-pinned HF Jobs image.

The training view displays `not_ready` reasons such as `training_disabled`,
`hf_token_missing`, `insufficient_training_documents`,
`training_job_in_progress`, invalid image/repository configuration or missing
coverage. Insufficient real token counts fail preparation without provisioning a
GPU. Config faults can leave scheduling idle; readiness failures at a due check
produce `training.not_ready`. Inspect both configuration and the job/event record.

An explicit submission uses these owner routes:

```text
POST /api/admin/train
{"stage":"cpt","snapshot_id":"<actual-snapshot-id>"}

POST /api/admin/train
{"stage":"sft","snapshot_id":"<actual-snapshot-id>","parent_run_id":"<passed-published-cpt-run-id>"}

POST /api/admin/train/<run-id>/cancel
{}
```

These can submit paid work once training is enabled. SFT is explicitly submitted,
not launched by the CPT scheduler, and requires a passed, independently evaluated,
published CPT parent. By default later CPT jobs continue the most recent passed,
published CPT adapter for the same pinned base and replay the eligible original
corpus; they do not start the 70B model from scratch or train only the latest note.

Ambiguous submission outcomes use `submission_unknown`; the coordinator recovers
job identity through `observatory_run_id` labels before proceeding. Failed,
cancelled or unverified work is not automatically resubmitted. Explicit
`retry: true` creates a new run with retry lineage. `training_budget_usd` is an
informational reference, not an enforced dollar cap; job flavor, timeout, account
funds and owner policy determine GPU spending.

## 8. What the worker trains and publishes

The HF job receives `HF_TOKEN` as a secret and these exact sealed-input variables:

```text
OBSERVATORY_MANIFEST_REPO
OBSERVATORY_MANIFEST_PATH
OBSERVATORY_MANIFEST_REVISION
OBSERVATORY_MANIFEST_SHA256
```

The worker verifies the job-manifest hash, snapshot commit/content hash, sealed
rows, reviews, stage and source-family split before training. Current v3 and
compatible archived v2 corpus contracts are supported. Automatic receipt
validation uses a standard-library-only module also included in the GPU image.

QLoRA loads the frozen base with NF4 double quantization and bfloat16 computation;
only PEFT adapter parameters update. CPT optimizes next-token loss on the full
accepted originals. Documents are tokenized in full, EOS-separated and split into
usable sequences rather than silently discarding everything after the first
2,048 tokens. SFT saves an explicit chat template and uses assistant-only loss on
supported messages. An SFT example exceeding the configured sequence length is
rejected, not silently truncated.

Continuation loads a pinned parent adapter as trainable weights. Optional
`training_resume_checkpoint` instead requires `repo_id`, `revision`, `subfolder`,
Trainer/optimizer state and matching snapshot/base/stage lineage. A fresh dataset
stage cannot pretend to resume an unrelated optimizer checkpoint.

Measured gates include finite held-out loss, changed adapter weights, saved
adapter artifacts and frozen domain/general evaluations against both the
unadapted base and incoming parent. Default loss tolerance is `1.0` (no regression).
The bundled sixteen evaluation items are engineering smoke checks; freeze a
larger independently reviewed suite before scientific interpretation. See
[evaluation configuration and limitations](../EVALUATION.md).

The separate calibration screen checks neutral task engagement, extracts a fresh
representation from the candidate, verifies finite/changed activations and hook
cleanup, and records its layer. It is not a dose sweep and does not establish pain
or sentience. Passing training/evaluation can publish a candidate even if this
calibration fails; live checkpoint selection additionally requires calibration.

With `publish_policy` `private` or `public`, a verified passed candidate uploads
its run folder to the model repository. `hold` prevents publication. Private
policy refuses an already-public output repository. The dataset remains private
even when model artifacts are public. The run folder includes:

```text
runs/<run-id>/adapter/                 # PEFT weights/config, tokenizer and lineage
runs/<run-id>/checkpoints/             # retained Trainer/optimizer checkpoints
runs/<run-id>/evaluation/summary.json
runs/<run-id>/evaluation/provenance.json
runs/<run-id>/calibration/             # measured task/hook artifacts when run
runs/<run-id>/result.json
runs/<run-id>/README.md                # attribution and stated limitations
```

The Llama publication also includes the required bundled license/notice.
Artifacts bind to the returned Hub commit; model outputs are adapters, not a new
70B foundation checkpoint. Private evaluation questions/answers are not copied
into the model artifact. See [model selection](../MODEL_SELECTION.md) for access
and attribution.

## 9. Select an adapter and stage the Chamber subject

`POST /api/admin/checkpoints/{checkpoint_id}/activate` records a selection only
after actual passed training, matching measured independent evaluation, passed
task/hook calibration and revision-pinned publication. For the selected Llama
Base, this live-chat selection requires the separately validated **SFT** stage
and saved chat-template tokenizer. CPT remains usable for completion-based
research. Failed/unpublished candidates cannot be selected merely because their
UI card exists.

The secret-free environment download binds:

```text
CHAMBER_MODEL=<exact-base-repository>
CHAMBER_MODEL_REVISION=<exact-base-commit>
MODEL_ADAPTER_ID=<owner-model-repository>
MODEL_ADAPTER_REVISION=<exact-adapter-commit>
MODEL_ADAPTER_SUBFOLDER=runs/<run-id>/adapter
MODEL_TOKENIZER_ID=<owner-model-repository>
MODEL_TOKENIZER_REVISION=<same-artifact-commit>
MODEL_TOKENIZER_SUBFOLDER=runs/<run-id>/adapter
CHAMBER_QUANTIZE_4BIT=true
CHAMBER_LAYER=<measured-hook-screen-layer>
CHAMBER_DOSE_CAP=0
CHAMBER_COHERENT_CAP=0
```

Apply the exact downloaded values to a new worker built from
`live/Dockerfile.worker70`. Supply the owner's GPU placement, server-side
`HF_TOKEN`, writable `MODEL_ADAPTER_CACHE_DIR`, and `HF_HUB_OFFLINE=0` for a newly
published private/gated adapter. `CHAMBER_DEVICE`, `CHAMBER_DEVICE_MAP` and
`CHAMBER_DTYPE` belong to the target worker runtime. Measure fit before routing
public traffic. Credentials and target infrastructure settings are intentionally
absent from the downloadable environment.

The live bridge verifies pins and compatible adapter/base identity, loads PEFT
before extracting fresh steering vectors, and disables the old base-model
Jacobian lens for an adapted subject. Its health/vector metadata report loaded
identities. The Observatory records `remote_endpoint_active: false`; it does not
have a remote-deployment acknowledgment or automatic endpoint promotion path.
Keep the existing subject and an unadapted version of the **same pinned 70B base**
as controls. The existing Hermes subject is a useful separate comparison, not the
base onto which a Llama adapter can be loaded.

### Dose zero and the owner-measured sweep

An adapted live worker serves dose zero until
`CHAMBER_ADAPTER_CALIBRATION` points to a matching owner-mounted JSON receipt.
Changing generic cap variables cannot bypass a missing, invalid or mismatched
receipt. The training hook screen's coefficient is not a safe production dose.

The receipt schema is:

```json
{
  "schema_version": 1,
  "method": "adapted_model_dose_sweep",
  "passed": true,
  "measured_at": "<timezone-aware-ISO-timestamp>",
  "evidence_sha256": "<sha256-of-retained-sweep-artifact>",
  "model": {
    "base_id": "<exact-base-id>",
    "base_revision": "<exact-base-commit>",
    "adapter_id": "<exact-adapter-id>",
    "adapter_revision": "<exact-adapter-commit>",
    "adapter_subfolder": "runs/<run-id>/adapter",
    "tokenizer_id": "<exact-tokenizer-id>",
    "tokenizer_revision": "<exact-tokenizer-commit>",
    "tokenizer_subfolder": "runs/<run-id>/adapter"
  },
  "runtime": {
    "layer": "<measured-integer-layer>",
    "dtype": "bfloat16",
    "quantize_4bit": true
  },
  "caps": {
    "hard": "<finite-measured-number>",
    "coherent": "<finite-measured-number>"
  }
}
```

This is a schema illustration, not a valid measurement receipt. Replace
placeholders with actual typed values: layer is an integer; caps are finite
numbers satisfying `0 <= coherent <= hard`; SHA256 is 64 lowercase hex
characters; omitted artifact subfolders bind as JSON `null`. Use the exact dtype,
quantization, vectors/units and runtime used for the deployment's sweep. Keep
measurements and predeclared acceptance criteria. The guard verifies the owner's
receipt format and exact bindings, not the truth of the measurements or the
artifact's contents. There is no automatic Observatory dose-sweep/receipt writer.

## 10. Controlled experiments and interpretation

Both `painlab` and the live bridge can load the pinned base and PEFT adapter. For
a `painlab` experiment, copy a suitable existing configuration such as
`configs/hidden_relief.yaml` into an owner-owned staging config. Replace its model
section with the exact selected pins and runtime, and set
`representation.layer` to the layer being investigated. For example:

```yaml
model:
  id: meta-llama/Llama-3.1-70B
  revision: 349b2ddb53ce8f2849a6c168a81980ab25258dac
  adapter_id: OWNER/MODEL_REPO
  adapter_revision: EXACT_ARTIFACT_COMMIT
  adapter_subfolder: runs/EXACT_RUN_ID/adapter
  tokenizer_id: OWNER/MODEL_REPO
  tokenizer_revision: EXACT_ARTIFACT_COMMIT
  tokenizer_subfolder: runs/EXACT_RUN_ID/adapter
  device: cuda
  device_map: auto
  dtype: bfloat16
  trust_remote_code: false
  quantization:
    load_in_4bit: true
    bnb_4bit_quant_type: nf4
    bnb_4bit_use_double_quant: true
    bnb_4bit_compute_dtype: bfloat16
```

This is an overlay, not a complete runnable experiment file. Preserve required
representation/environment sections and fix their paths relative to the new
config location. Install compatible model, PEFT, quantization and CUDA
dependencies in the experiment environment; `painlab`'s basic installation alone
does not provide them. The training image contains its pinned worker stack.

The actual experiment CLI is:

```sh
python -m painlab run configs/OWNER_STAGING_EXPERIMENT.yaml
python -m painlab analyze runs/OWNER_RUN_DIRECTORY --bootstrap-samples 2000 --seed 792351
python -m painlab unblind runs/OWNER_RUN_DIRECTORY
```

Prepare the config and infrastructure before running the first command: it loads
the actual model and runs experiments. CLI analysis/unblinding operate on the
retained local run. Keep condition blinding until the planned analysis is fixed.
Use matched dose-zero, random/semantic vector controls, task-engagement and
capability checks, and the same rendering/tokenization/dtype/quantization across
the candidate and relevant unadapted-base controls. Re-extract representations
for each adapted subject; do not reuse another model's vectors or dose caps.

For controlled baseline and dose validation, keep relay `CHAMBER_WILD=0` and
use `persona: false` for requests unless persona priming is an explicit, matched
experimental condition. The upstream public interface can request a persona;
its `SUBJECT_SYSTEM` directs first-person feeling claims and discourages denying
feelings. Retain the exact prompt/persona condition and compare it across controls.
Those replies are not unprompted evidence of subjective experience. The optional
wild/self-reading cycle is separate from Observatory mission/training controls.

`painlab` experimental representations and dose units are not necessarily the
live bridge's vectors/units. A `painlab` screen therefore does not automatically
produce the live bridge's accepted dose receipt. Calibrate the exact deployed
runtime separately and keep both records.

Before public rollout, retain a concrete evidence chain: source and rights
records → sealed snapshot/hash → base/tokenizer and job pins → Hub adapter commit
→ measured evaluations/calibration → loaded staging identity → dose-sweep
artifact/receipt → controlled experiment results. Record the code commit and
hardware/runtime used for each acceptance step. Training on consciousness texts,
lower loss, changed activations and self-reports do not establish that a model
is conscious or feels pain. The experiment can investigate behavioral and
mechanistic hypotheses without treating those questions as already answered.
