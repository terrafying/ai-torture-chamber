# Developer-funded Solana research

For an explanatory funding walkthrough, read
[Ledger treasury to agent spending wallet](docs/ledger-x402-funding.md).
It describes owner-approved native-USDC top-ups into the separate software signer;
native Ledger signing of inference requests is not implemented in this repository.

The researcher pays for inference through an isolated x402 broker. The owner
supplies a dedicated Solana spending wallet, approves BlockRun's merchant
recipient and sets explicit spending limits. Agents never receive the signer.
The browser runs in local Chromium by default; its compute is still hosted by
the owner. Hugging Face GPU jobs and publishing use the owner's separate account.

This implements the payment and research lifecycle. The tests use authored
gateway, balance and transaction fixtures. No mainnet payment, wallet creation,
Docker build or real 70B training run was performed for this handoff.

## Start both services

Copy `.env.example` to `.env` and `.env.payments.example` to `.env.payments` in
`observatory/`. Provide different strong owner and internal transport tokens;
the internal token must match in the two files. Only `.env.payments` contains
`X402_SOLANA_PRIVATE_KEY`. Supply an existing dedicated spending keypair in
the base58 format accepted by the official x402 SVM signer. Neither the website
nor an agent is a wallet-creation or key-management tool.

Leave `X402_ENABLED=false` while configuring. Set finite, explicit USDC values
for `X402_MAX_REQUEST_USDC`, `X402_DAILY_LIMIT_USDC` and
`X402_MIN_RESERVE_USDC`. Obtain and independently verify the merchant recipient
for `https://sol.blockrun.ai/api/v1` with BlockRun, then enter it in
`X402_ALLOWED_PAY_TO`. An empty allowlist disables spending. A recipient offered
by an arbitrary crawled page is never automatically approved.

The broker accepts only x402 v2 `exact` offers on Solana mainnet, native USDC
mint `EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v`, the approved recipient,
bounded output and price, and the selected fixed native vendor endpoint. It uses
the pinned official `x402[svm]==2.25.0` client. The broker has no generic
URL-signing, arbitrary transfer or agent-controlled recipient endpoint.

With Docker installed, from the repository root:

```sh
docker compose -f observatory/compose.yaml config --quiet
docker compose -f observatory/compose.yaml up --build -d
```

The sidecar binds to loopback port 8060. The payments process has no published
port and uses its own persistent ledger volume and encryption key. Run one
process and one replica per service. This compose file is a local starting
configuration; use a reverse proxy with TLS and an isolated browser egress
policy for production. The bridge network needs internet access and is not
itself that egress policy. Both service healthchecks are read-only.

Alternatively, use separate Python 3.12 environments, install
`requirements-research.txt` for the researcher and `requirements-payments.txt`
for the broker, and run two terminals from the repository root:

```sh
# Research environment; local Chromium installation is required.
python -m playwright install chromium
python -m uvicorn observatory.app:app --env-file observatory/.env --host 127.0.0.1 --port 8060 --workers 1
# Payment environment, in the other terminal.
python -m uvicorn observatory.x402_broker:app --env-file observatory/.env.payments --host 127.0.0.1 --port 8062 --workers 1
```

Starting either service with a fresh, disabled configuration does not launch a
mission or GPU job. Persisted running missions resume, and enabled training can
submit a due CPT job after restart. Stop the mission and disable
`training_enabled` before shutdown when recurring work should remain stopped.
Enabling payments is explicit. Once
the owner configures the broker, set `X402_ENABLED=true` and restart that
service. Its public funding response supplies the derived spending address;
the owner funds that address with Solana native USDC. Public visitors can inspect
read-only funding and receipt metadata through the sidecar API, but cannot change
limits, sign payments or start missions. There is no public Funding page.

## Choose a researcher model

Open the Observatory from the sidecar and enter the owner token in operator
setup. Select `Solana x402`, then an explicit compatible gateway catalog model.
The initial adapters support OpenAI-native Responses and Anthropic-native
Messages, including browser images and structured actions. The selector filters
the catalog using those requirements. Names and categories alone cannot establish
structured-action compatibility. When explicit gateway metadata is missing, the
owner must declare exact expected model capabilities in
`X402_RESEARCH_MODEL_CAPABILITIES`, a broker environment JSON object. For example,
after reviewing the selected model's vendor documentation, use its exact catalog
ID as the key and a record such as:

```json
{
  "openai/YOUR_MODEL_ID": {
    "protocols": ["responses"],
    "vision": true,
    "structured_actions": true,
    "structured_outputs": true,
    "tool_calling": false
  }
}
```

An Anthropic Messages model instead needs `protocols: ["messages"]`, `vision`,
`structured_actions` and `tool_calling` true; Responses `structured_outputs`
is not required for it. The example is a configuration shape, not a real model
or a verified compatibility result. The default registry is empty. Owner-declared
and gateway-advertised controls are labeled separately, and both require a
bounded funded acceptance test with the selected model. Unverified listings are
not presented as ready research choices. The selector does not assert that every
model on BlockRun works with Browser Use, and there is no silent provider or
model fallback. Live provider acceptance still needs a funded test with the
chosen model. Catalog membership and advertised capabilities are not a measured
compatibility certificate.

The model identifier, native protocol and output ceiling are fixed in each
request. Only this broker's approved inference route is paid; encountering a
402 while reading a webpage does not buy access to it. Browser Use Cloud and
dedicated CDP remain explicit alternate browser choices. Their account billing
is separate; the implemented Solana broker does not pay Browser Use Cloud's
Base credit endpoint. Direct OpenAI/Claude credentials remain an explicit
alternate inference route for compatibility testing, not the default.

## Funds, reservations and uncertain settlement

The read-only `GET /api/funding` endpoint reports balance, reserved payments,
settled spending, daily headroom and public transaction identifiers when the
broker supplies them. The website has no Funding view. This public response
never publishes prompt bodies, payment authorization headers, signing payloads,
broker credentials or the wallet key. Balance/catalog errors return unknown or
unavailable status, not fabricated funds or preview records. Any authored preview
funding fixture has no depositable wallet address.

Before opening a browser, x402 research checks broker health, the chosen model
and funding. Funding readiness conservatively requires capacity for one request
at the configured maximum quote, plus existing reservations and minimum reserve.
A smaller actual quote may cost less. Each request then validates its real quote
and atomically reserves its amount against current balance and the UTC daily
limit. Unresolved reservations remain held across days.

Nonempty seller-defined `extra.memo` values are unsupported and rejected before
reservation or signing. The pinned SDK must generate a fresh client nonce; a
reused merchant memo could otherwise match an unrelated old transfer of the same
amount. Quotes are never rewritten. A merchant requiring its own memo needs a
separately validated transport. The persistent ledger also prevents attributing
one settlement transaction to two requests, including response recovery and
manual reconciliation. Keep that ledger when restarting the service.

Insufficient funds or daily headroom puts the mission into `funding_paused`
and releases owned browsers. The supervisor periodically checks funds and can
resume that same mission after preflight passes. It never overrides an owner's
Pause or Stop. Configuration/authentication/schema errors fault the mission
instead of opening repeated paid browser passes.

An authorized request has a durable ID. Successful same-ID retries return an
encrypted cached response without a second payment; changed content conflicts.
The researcher also retains its pending UUID and encrypted body in the sidecar
database, scoped to the agent. After a restart an unresolved context requires
owner review before a new browser is allocated. Known, explicit unpaid broker
failures clear that pending context; insufficient-funds auto-resume therefore
does not attempt to replay an expired browser action.
An interrupted authorized request, absent/uncertain settlement receipt or
unconfirmed transfer blocks further spending. A successful vendor response by
itself is not proof of settled payment. Settlement is checked against the exact
USDC transfer and the SDK-generated transaction memo using read-only Solana RPC.
The broker does not automatically sign again after an ambiguous outcome.

Reconcile an uncertain request through the internal authenticated endpoint:

```text
POST /v1/reconcile
Authorization: Bearer <internal broker token>
Content-Type: application/json

{"request_id":"<existing UUID>","transaction":"<verified Solana signature>"}
```

This operation verifies the existing transfer; it cannot sign or resend one.
Without a confirming memo-bound transaction, the reservation remains held.
Reconciliation can unlock an already cached response; it does not regenerate
a lost response. Inspect the merchant receipt and ledger; do not delete or edit
the ledger to turn an unknown payment into a new authorization. A faulted mission
requires an explicit operator restart after resolving the fault. Back up each
SQLite database with its encryption key consistently; changing keys without
migration destroys recovery of encrypted settings/responses.

The sidecar provides owner-only recovery routes under its normal owner Bearer
token:

- `GET /api/admin/research-payments` lists pending request metadata, excluding bodies.
- `GET /api/admin/research-payments/<UUID>` inspects the local broker receipt and
  any recovered vendor output. It cannot send a new model request or payment.
- `POST /api/admin/research-payments/<UUID>/acknowledge` with `{"reviewed":true}`
  releases the caller's pending context after owner review. The mission must be
  paused, stopped or faulted and the broker must show no request in flight.

Uncertain/authorized/reserved payments cannot be acknowledged away. First
reconcile a confirmed transfer through the broker as above. For an idle unpaid
or not-yet-received request, acknowledgement creates a durable broker cancellation
record before clearing the caller state; a delayed old POST then cannot authorize
a payment. This cancellation operation changes only ledger state and never signs,
sends funds or buys inference. Recovered actions refer to the old browser context;
review them rather than replaying them in a newly opened page. Explicitly resume
the mission after review is complete. Recovery output remains owner-only and is
never added automatically to a training dataset or public notebook.

An interrupted automatic original-document review also has a durable review
marker. Payment reconciliation alone does not clear that marker. After reviewing
the payment outcome, use the authenticated **Review recovery** control while the
mission is paused, stopped or faulted, then explicitly resume. Completed review
verdicts are preserved; retrying an unfinished stage may incur another charge.
See [automatic curation recovery](docs/automatic-curation.md).

## Research, datasets and 70B training

Continuous missions choose searches and links, save original text and publish
concise evidence-linked notes. Agents explore until stopped, using checkpointed
bounded passes. Public observations and next actions are displayed; hidden model
reasoning is not a notebook feed. Site access rules still apply.
The public browser pane is view-only: agent scrolling determines its captures,
and a position bar reports the captured viewport. An agent can select an exact
visible passage with `inspect_visible_section` before saving a note. The saw and
highlights require verified painted text lines bound to the focused document and
same screenshot hash. A matching saved-note event promotes the inspection and
delivers its public authored note to the notebook. Visitor wheel/touch input cannot
scroll the agent's document. This records source correspondence, not measured
attention or proof of understanding. The preview demonstrates the sequence using
authored local fixtures; it is not a recording of a real research session.

Original-text rights and source-family holdouts govern corpus admission. Generated
notes do not silently become CPT text. Unknown rights stay quarantined, social
sources need separately recorded permission, and SFT examples need individual
approval plus an applicable teacher-output agreement. x402 payment does not
grant data or model-output training rights. The owner must verify the selected
teacher's actual contract permits the intended use; a settings checkbox or
policy-reference URL is not permission from a provider.

The owner can seal and download an actual dataset ZIP through authenticated
controls. It contains CPT/SFT train and validation JSONL, the sealed manifest,
rights/provenance, exclusions and file hashes. Export does not contact Hugging
Face or provision a GPU. The server refuses a tampered snapshot.

The selected subject is pinned Meta Llama 3.1 70B Base. Scheduled readiness
checks run every four hours; sufficient newly eligible original documents can
trigger continued QLoRA adapter training when the owner enables HF Jobs. This
updates adapters on an existing 70B base; it is not pretraining 70B parameters
from random initialization. SFT is a separately submitted and evaluated stage.
Use `MODEL_SELECTION.md` and `README.md` for the gated model, digest-pinned GPU
image, account configuration and candidate publishing requirements.

GPU jobs do **not** draw from this Solana wallet. There is no verified native
x402 HF Jobs billing integration here. Registry hosting, browser hosting, GPU
credits, gated model access and private Hub publishing still need owner accounts.

## Acceptance before unattended deployment

The remaining checks require the owner's infrastructure. Record their outcomes
against the branch commit and exact model/image revisions:

1. Build the two service images and verify persistence after restart, TLS,
   restricted browser egress, owner authentication and secret-free public reads.
2. With training disabled and one researcher, make a deliberately bounded funded
   research pass using the chosen model. Verify the native action schema, browser
   capture, original text, evidence passage, owned-browser cleanup and confirmed
   Solana receipt. This is a real paid acceptance test, not the fixture suite.
3. Verify funding pause/resume and operator Stop precedence. Exercise uncertain
   settlement in a staging fixture, not by authorizing duplicate mainnet payments.
4. Seal eligible original documents; inspect reviewed research areas and basis of
   claims, perspective coverage, duplicate lineage, exclusions, persistent
   holdouts and the downloaded ZIP's hashes. Corpus review gates, registered
   evaluation-source exclusions and a separately sealed domain/general evaluation
   suite are enforced before training. The bundled sixteen tasks are engineering
   smoke checks, not a validated scientific benchmark. Freeze an independently
   reviewed suite and retain untouched test material before making claims about
   consciousness understanding or general-capability retention.
5. Build/push the GPU image, record its actual SHA256 digest, verify gated model
   access and run one real 70B QLoRA job with bounded timeout. Record memory use,
   tokenizer counts, measured holdout comparisons and the revision-pinned Hub
   artifacts before enabling recurring submissions. The default held-out loss
   ratio is `training_max_loss_ratio=1.0`; the independent accuracy/loss gates
   also require non-regression against the unadapted base and incoming adapter.
   Passing these measured gates does not establish scientific validity or domain
   improvement.
6. Validate separately approved SFT, then load the candidate on a separate Chamber
   worker. Its fresh-vector smoke screen is distinct from a dose sweep. Adapted
   workers serve dose zero until the owner supplies a matching dose-calibration
   receipt described below. Retain the original model endpoint for comparisons.
7. Agree the upstream rollout policy before merging: existing repository workflows
   can update the RunPod worker and Railway relay on a master push. Selecting a
   checkpoint in the Observatory does not deploy that endpoint.

The local review suite exercises real tiny CPU optimizer/PEFT paths and mocked
provider boundaries. It does not replace these funded/provider/GPU checks or
establish consciousness, sentience or felt pain.

## Adapted Chamber dose receipt

For an adapted worker, set `CHAMBER_ADAPTER_CALIBRATION` to an owner-mounted JSON
receipt from the measured dose sweep. The exact `model` object must bind
`base_id`, `base_revision`, `adapter_id`, `adapter_revision`, `adapter_subfolder`,
`tokenizer_id`, `tokenizer_revision` and `tokenizer_subfolder` to the deployment.
The exact `runtime` object contains `layer`, `dtype` (for example `bfloat16`)
and `quantize_4bit` (a boolean). Omitted subfolders bind as JSON null.

The receipt also contains `schema_version: 1`,
`method: "adapted_model_dose_sweep"`, `passed: true`, a timezone-aware ISO timestamp
in `measured_at`, the SHA256
of the retained measurement artifact in `evidence_sha256`, and `caps` with
measured finite numeric `hard` and `coherent` values satisfying
`0 <= coherent <= hard`. Do not copy the tiny-test fixture's example caps.
`CHAMBER_DOSE_CAP` and `CHAMBER_COHERENT_CAP` cannot bypass a missing or mismatched
adapter receipt. Health reports the receipt status without its private path.
This records owner measurements and checks their bindings, not their truth.

## References

- [BlockRun Solana inference](https://blockrun.ai/x402/solana)
- [Native paid endpoint documentation](https://blockrun.ai/docs/x402/endpoints)
- [Official x402 network/token support](https://docs.x402.org/core-concepts/network-and-token-support)
- [Browser Use Cloud x402 credits](https://docs.browser-use.com/cloud/guides/x402)
- [Hugging Face Jobs billing](https://huggingface.co/docs/hub/jobs-pricing)
