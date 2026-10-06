# Deployment and operator runbook

This is the staging-to-production runbook for the Observatory sidecar, its
optional Solana x402 inference broker, and the separate Hugging Face training
worker. Read [the architecture and workflow](../README.md),
[research briefs](research-mission.md), [automatic curation](automatic-curation.md)
and [model selection](../MODEL_SELECTION.md) alongside it.

The implementation is an integration alpha. Offline tests exercise the software
contracts; a funded native-model browser pass, mainnet settlement, Docker execution,
real 70B GPU training, adapted-worker memory use and production deployment remain
unverified. Record staging results against the exact source commit, model revisions
and container digest before enabling unattended operation.

## Services, accounts and costs

| Component | Runs where | Credentials and billing |
| --- | --- | --- |
| Public watch interface | Existing static site or Observatory HTTP service | No visitor credentials; authored preview is separate |
| Research and automatic curation | Python sidecar, CPU host | Frontier inference through an explicit direct API route or x402 |
| Research browser | Disposable local Chromium, Browser Use Cloud, or isolated CDP | Local CPU/RAM hosting; Cloud browser credentials/billing are separate |
| x402 broker | Separate private process/container | Dedicated owner-funded Solana signer, approved merchant, request/day/reserve limits |
| 70B adapter training | Hugging Face Jobs, separate GPU image | HF account/token, Jobs billing, gated base access and registry access |
| Adapted Chamber | Separate CUDA worker and existing relay | Owner-managed deployment, cache, credentials and dose measurements |

x402 pays the implemented inference route. It does not pay for visited websites,
Browser Use Cloud, HF Jobs, registry storage or Chamber hosting. Crawling and
training remain off-chain; inference settlements are on-chain. There is no public
donation flow or framework-generated wallet. Use an existing dedicated spending
keypair. No initial corpus upload is necessary: the frontier research brain and
the 70B base are already pretrained.

Use Python 3.12 for local sidecar/payment environments, or Docker with Compose
for the provided containers. Keep the existing Chamber environment separate.
Run **one process and one replica per service**: the research supervisor and
training scheduler are in-process, and the broker permits one ledger writer.
Provide a persistent disk/volume, HTTPS termination, and a network-level browser
egress policy that rejects private, loopback and metadata destinations. The Docker
bridge network needs internet access and is not that egress policy.

## Preview, fresh live state and runtime workers

From the repository root, inspect the authored UI without running any service:

```sh
python -m http.server 8060 --bind 127.0.0.1 --directory site
```

Open `http://127.0.0.1:8060/observatory.html?preview=1&motion=1#research`.
Preview records and browser traces are explicitly simulated. The static server
cannot crawl, pay for inference or train weights. Stop it before binding the
real sidecar to the same port.

The normal `observatory.app:app` entry point starts the research, automatic-curation
and training runtime tasks. There is no deployment environment switch named
`OBSERVATORY_ENABLE_RUNTIME`; `create_app(enable_runtime=False)` is a Python
factory option used by tests/read-only integrations. Use the normal entry point
for a live deployment. A fresh database has a stopped mission, automatic curation
off and training off, so idle tasks do not provision work. A persisted running
mission resumes on restart; enabled training can submit a due CPT job immediately
when its readiness checks pass.

## Choose a startup path

Commands below run from the repository root. Shell examples use POSIX syntax;
on Windows, activate the corresponding environment with its PowerShell
`Scripts/Activate.ps1` and use `Copy-Item` for the two file copies. Do not execute
deployment, paid submission or funding commands merely to inspect the interface.

### Direct model API: one sidecar

```sh
cp observatory/.env.example observatory/.env
python -m venv .venv-observatory
. .venv-observatory/bin/activate
python -m pip install -r observatory/requirements-research.txt
python -m playwright install chromium
python -m uvicorn observatory.app:app --env-file observatory/.env --host 127.0.0.1 --port 8060 --workers 1
```

Before startup, edit the server-side `.env`: set a strong
`OBSERVATORY_ADMIN_TOKEN`; provide `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` for the
chosen direct route, or enter that provider key later in authenticated Operator
setup. The example defaults to x402, so explicitly choose OpenAI/Claude and an
available native-compatible model in setup before starting research. For Claude,
select `messages`; for OpenAI, select `responses`. Model capabilities need a live
staging check; model names alone do not establish browser-action compatibility.
Local Chromium is the default browser. Browser Use Cloud instead needs its
separate `BROWSER_USE_API_KEY`; dedicated CDP needs `cdp_url`, an unauthenticated
isolated browser and `cdp_isolated_ack: true`, with one Scholar. The owner must
provide a fresh cookie-free, blank CDP session for every pass. Detaching preserves
the owner's browser; it does not reset visited pages or cookies. Reusing that
session can fail the next pass's isolation checks. Prefer local/cloud browsers
for unattended research unless the CDP infrastructure implements that lifecycle.

The payments process is not required for direct inference. Its read-only catalog
and balance integrations may be unavailable; that is not evidence that a direct
research configuration is ready or broken. Never enable an x402 mission without
its separately configured broker.

A container-only direct path is also available:

```sh
docker build -f observatory/Dockerfile -t consciousness-observatory .
docker run --env-file observatory/.env -e OBSERVATORY_DB=/data/state.sqlite3 -p 127.0.0.1:8060:8060 -v research-data:/data consciousness-observatory
```

### Solana x402: separate research and payment services

```sh
cp observatory/.env.example observatory/.env
cp observatory/.env.payments.example observatory/.env.payments
docker compose -f observatory/compose.yaml config --quiet
docker compose -f observatory/compose.yaml up --build -d
docker compose -f observatory/compose.yaml ps
```

Edit both files before startup. Use different owner and internal transport tokens;
`OBSERVATORY_PAYMENT_BROKER_TOKEN` must match between the files. Only the payment
process receives `X402_SOLANA_PRIVATE_KEY`. Leave `X402_ENABLED=false` while
configuring the signer, finite USDC limits and independently verified merchant
recipient in `X402_ALLOWED_PAY_TO`. An empty recipient allowlist disables spending.
Use the exact limits and capability registry described in
[x402 operations](../X402_OPERATIONS.md); do not approve a recipient supplied by a
crawled page. The broker supports its fixed Solana native inference endpoints,
not arbitrary paid URLs.

Compose publishes only the sidecar at `127.0.0.1:8060`; the broker has no host
port and is reached at `http://payments:8062` inside the network. `research-data`
and `payments-data` are independent persistent volumes. Keep one replica of each.
After deliberate configuration, set `X402_ENABLED=true` in `.env.payments` and
recreate the broker to load the changed environment:

```sh
docker compose -f observatory/compose.yaml up -d --force-recreate payments
```

Funding metadata derives the address from the configured signer. Fund that
verified address with native Solana USDC; use the gateway's required network and
account configuration. The broker's balance/quote/settlement checks determine
readiness; a deposited balance alone does not prove model compatibility. Choose
an explicit eligible gateway model and protocol in Operator setup. Where gateway
metadata is absent, the owner's exact capability declaration permits a bounded
test, not a compatibility guarantee. No provider key or signer goes into the page
for the x402 route.

Alternatively, run separate Python environments. Install
`observatory/requirements-payments.txt` into `.venv-payments`; keep the research
dependencies in `.venv-observatory`. Create the payment environment first:

```sh
python -m venv .venv-payments
. .venv-payments/bin/activate
python -m pip install -r observatory/requirements-payments.txt
```

Then use separate activated terminals:

```sh
# Payment environment.
python -m uvicorn observatory.x402_broker:app --env-file observatory/.env.payments --host 127.0.0.1 --port 8062 --workers 1
# Research environment; install local Chromium as above.
python -m uvicorn observatory.app:app --env-file observatory/.env --host 127.0.0.1 --port 8060 --workers 1
```

The research `.env` uses `http://127.0.0.1:8062` for this two-process path.
Compose overrides it with its internal service address. The payment environment
is intentionally distinct from the research and GPU environments.

## Credentials, persistence and recovery materials

| Configure server-side | Configure through authenticated setup/settings |
| --- | --- |
| `OBSERVATORY_ADMIN_TOKEN` | Research provider/model/protocol, objective and agent count |
| Matching `OBSERVATORY_PAYMENT_BROKER_TOKEN` | Browser provider and optional browser credential/CDP configuration |
| `X402_SOLANA_PRIVATE_KEY`, broker limits/recipient/capabilities | Optional direct provider keys and `hf_token`, encrypted at rest |
| `OBSERVATORY_DB`, optional stable `OBSERVATORY_SECRET_KEY` | Automatic-curation v2 opt-in and training enablement |
| `X402_BROKER_DB`, optional stable `X402_BROKER_SECRET_KEY` | HF repositories, training image/recipe and publication policy |

Direct provider/browser keys and `HF_TOKEN` can instead be server environment
variables. Training also reads `OBSERVATORY_HF_NAMESPACE`,
`OBSERVATORY_HF_DATASET_REPO`, `OBSERVATORY_HF_MODEL_REPO` and
`OBSERVATORY_TRAINING_IMAGE` when the corresponding saved setting is empty.
Do not assume every setting has an environment-variable override.

Keep `.env` files out of Git, `site/`, image layers and public logs. They are
gitignored. Owner access uses a Bearer header; enter the token in Operator setup,
never a query parameter. The page keeps it in memory and needs it again after
reload. There is one owner token, not multi-user accounts or role-based access.
Provider keys saved in setup are encrypted in SQLite and omitted from browser
storage; public settings show configured status rather than credential values.
Leaving a secret input blank preserves its saved credential when saving setup.
Signing keys and internal/owner tokens are environment configuration and are
refused as saved UI settings.

Local defaults persist research to `.observatory/state.sqlite3` and payments to
`observatory-data/payments.sqlite3`. Without supplied Fernet environment keys,
each service generates a protected `<database>.key` beside its database. In
Compose these files live inside the corresponding `/data` volume. Back up each
database consistently together with its key, server secret configuration and any
needed frame files. SQLite uses WAL: do not copy only a live `.sqlite3` file and
assume it is a consistent backup. Use SQLite's backup API or stop both writers
and take a consistent volume backup, including WAL/key files where present.
Protect backups as private research and credential recovery material.

Before planned maintenance: Stop the mission, disable `training_enabled`, inspect
active GPU jobs and unresolved payments, then stop the containers without deleting
volumes:

```sh
docker compose -f observatory/compose.yaml stop observatory payments
```

Do not use `down -v` for routine maintenance. Losing or changing encryption keys
without migration prevents decryption of stored credentials, pending request
bodies and cached responses. A restore must preserve request/run identities;
inspect remote HF jobs and settled/uncertain payment records before resuming.

## Publish the watch interface over HTTPS

First verify the sidecar directly at `http://127.0.0.1:8060/`; `/` redirects to
`/observatory.html?api=/api`. `GET /api/health` reports the actual runtime tasks and
whether owner controls are configured. Check all three tasks, not just its
top-level `status: "ok"`; missing imports/startup failures produce runtime events.
`GET /api/state` is public and sanitized. Existing sources need not have any
screenshots until a real browser pass runs.

For the existing static site, add an owner-specific same-origin rewrite:

```json
{
  "source": "/observatory-api/:path*",
  "destination": "https://YOUR-SIDECAR/api/:path*"
}
```

Add it to `site/vercel.json` alongside the existing `/chamber` rewrite. The
repository does **not** contain an owner deployment address for the Observatory.
`/observatory.html` defaults to `/observatory-api`; `?api=/api` selects the
sidecar-served path. Preserve the existing Chamber mapping. The sidecar has no
permissive CORS setup; a same-origin proxy is the supported integration path.
The frontend refuses remote plain HTTP and credentials embedded in endpoint URLs.

An Nginx equivalent, inside an existing TLS virtual host, is:

```nginx
location /observatory-api/ {
    proxy_pass http://127.0.0.1:8060/api/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header Authorization $http_authorization;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 3600s;
}
```

This is a deployment example, not a checked-in production proxy. Verify long-lived
SSE at `/api/events` survives the actual host's stream/time limits, and that
responses preserve the `X-Observatory-*` screenshot headers. Preserve no-store
headers on sensitive reads; do not cache owner responses or frame telemetry.
Apply appropriate public screenshot/SSE rate limits without buffering events.
Do not expose the broker, signer, CDP endpoint or raw source exports publicly.
A failed live connection shows a connection error; it never falls back to authored
preview data. `?preview=1` explicitly selects examples even on a live host.

## Operate a research mission

Start with training and automatic curation disabled and one browsing agent. In
Operator setup, choose the inference route, explicit model/protocol and disposable
browser; save configuration, then start a bounded staging mission. The six roles
expand the default consciousness mission without enforcing a domain allowlist.
An edited objective controls topic scope; see [the exact briefs](research-mission.md).
Scale `agent_count` up to six after checking the first pass. A pass checkpoints
and replans; the overall mission continues until stopped. For the initial staging
check, watch one pass and issue Stop; there is no automatic mission deadline.

These authenticated routes are the UI's backend contract:

| Action | Route and body |
| --- | --- |
| Save configuration | `POST /api/admin/settings`, settings object |
| Start / pause / resume / stop | `POST /api/admin/missions/{action}`, `{}` or an explicit `objective` |
| Seal an eligible snapshot | `POST /api/admin/snapshots`, no custom text payload |
| Review a source | `POST /api/admin/sources/{source_id}/review`, rights/extraction/quality fields |
| Export a sealed dataset | `GET /api/admin/datasets/{snapshot_id}/export`, owner Bearer header |

Pause reaches a safe boundary and prevents further mission work. Stop closes owned
browser sessions while preserving sources and notes. Neither operation cancels an
already submitted HF job or disables the separate training schedule. Disable
training and cancel its job explicitly when that is intended. Start/resume can
accept partially configured settings; actual provider preflight can still fault.
A successful HTTP mission response is not proof that browsing has begun.

For x402, insufficient verified funds or daily headroom releases owned browsers
and enters `funding_paused`. Fresh preflight can automatically resume that same
mission; owner Pause/Stop takes precedence. Authentication, schema and permanent
provider errors need explicit correction and resume. Roadblocks/access rules are
not bypassed: sites requiring login, POST-only flows or excluded access may remain
unavailable. Autonomous navigation can choose another lead, but universal access
and deterministic failed-link rerouting are not guaranteed.

If a payment result is uncertain, inspect owner-only
`GET /api/admin/research-payments` and its `/{request_id}` detail. Reconcile an
existing memo-bound transfer through the broker's internal `POST /v1/reconcile`
using its transport token; this cannot sign or resend a transfer. Then use
`POST /api/admin/research-payments/{request_id}/acknowledge` with
`{"reviewed": true}` while paused/stopped/faulted and no request is in flight.
Unknown/authorized/reserved outcomes cannot be acknowledged away. Do not delete
ledger rows or replay recovered browser actions in a new browser. See
[the payment recovery contract](../X402_OPERATIONS.md).

## Enable automatic original-document acceptance

After successful source collection, explicitly opt in through setup or owner
settings:

```json
{
  "auto_curation_enabled": true,
  "auto_curation_policy_ack": "originals-v2"
}
```

This adds two blind model review calls for a qualifying original, using the chosen
research model/billing route. It runs while the mission is running. Code retains
verified rights, passed extraction, provenance and contamination gates; agreement
and exact quotes are also required. Broad accepted originals retain domain and
evidence-type tags. Religious/philosophical accounts do not substitute for the
scientific or machine-perspective coverage floor. V1 acknowledgements leave the
worker idle until the owner selects v2; historical v1 receipts retain their scope.
Generated Q&A is not automatically approved by this switch.

Evidence shows review receipts and manual exceptions. An interrupted paid review
may have already been billed. Reconcile its pending payment first. While the
mission is paused/faulted/stopped and the worker is idle, **Review recovery**
authorizes another attempt; completed verdicts are retained and the mission is
not resumed. API equivalent: `POST /api/admin/curation/recovery` with
`{"reviewed": true, "review_id": "EXACT_RECORDED_REVIEW_ID"}`. A changed ID,
busy review or pending broker request is refused.

## Build and configure the 70B training path

The selected base is `meta-llama/Llama-3.1-70B`, pinned to
`349b2ddb53ce8f2849a6c168a81980ab25258dac`. Obtain gated access with the same HF
account/token used by the coordinator and GPU worker. Enable HF Jobs on that
account and provision its own billing. Accepted original source text enters CPT;
the recipe updates QLoRA adapters over frozen base weights. It does not train all
70B parameters from scratch. The supported single-GPU 70B path is QLoRA; full
precision 70B LoRA is refused by preparation and needs a separate distributed
recipe. Supported worker choices are `a100-large` and `h200`; memory/throughput
still require an actual GPU acceptance run.

Build the **training** image from the repository root, push it to an owner registry
accessible to HF Jobs, and retrieve its digest. Example placeholders:

```sh
docker build -f observatory/Dockerfile.training -t YOUR_REGISTRY/observatory-training:YOUR_COMMIT .
docker push YOUR_REGISTRY/observatory-training:YOUR_COMMIT
docker inspect --format='{{index .RepoDigests 0}}' YOUR_REGISTRY/observatory-training:YOUR_COMMIT
```

Configure `training_image` with `YOUR_REGISTRY/observatory-training@sha256:<64 lowercase hex characters>`,
not a mutable tag. The image uses a distinct CUDA/Transformers/PEFT/TRL environment;
do not substitute the research or existing Chamber image. The current Jobs call
does not pass registry login credentials: make the image pullable by the Jobs
environment and test that pull before the large run.

Save `hf_namespace`, `hf_dataset_repo`, `hf_model_repo`, `hf_token` (or `HF_TOKEN`),
the image digest and hardware. Both repo IDs must be `namespace/name` owned by
`hf_namespace`. The token needs the gated-model and intended Hub/Jobs access.
Dataset exports are enforced private, even when adapter publication is public;
an already public dataset repo is refused. Choose `publish_policy` deliberately:
`private`, `public` or `hold`; live checkpoint selection needs a published,
revision-pinned adapter. Retain the selected model's license/attribution and
Llama-prefixed derived model naming described in [model selection](../MODEL_SELECTION.md).

Keep `training_enabled: false` until configuration and staging checks are complete.
Then enabling it starts durable readiness checks at `training_interval_hours: 4`.
This is an interval from the persisted last check, not a wall-clock cron at fixed
hours. It may check promptly when enabled/restarted. Readiness requires a sealed
current-policy corpus, no train/validation family leakage, scientific/machine
perspective presence, nonempty splits, at least 20 original **training** documents
and at least 50,000 actual tokenizer-counted training tokens by default. Thresholds
are configurable operational floors. No overlapping job is submitted; unchanged
CPT data/recipe reuses its existing run rather than buying a new run each interval.
The schedule is independent of whether browsing is paused.

HF preparation resolves immutable model/tokenizer commits, checks actual tokens,
uploads private dataset/evaluation/job manifests, then submits the GPU job. It
polls status and retains IDs. A restart during submission recovers the remote job
by its `observatory_run_id` label instead of silently submitting again. Failed
paid jobs require an explicit retry. `training_budget_usd` is informational and
is **not** an enforced provider spending cap; hardware, timeout, account controls
and enabling/disabling submission govern GPU spending.

| Training action | Owner API |
| --- | --- |
| Explicit CPT submission | `POST /api/admin/train`, `{"snapshot_id":"...","stage":"cpt"}` |
| Separate SFT submission | Same route, `{"stage":"sft","snapshot_id":"...","parent_run_id":"..."}` |
| Retry a failed eligible run | Same stage/snapshot/parent plus `"retry": true`; creates a new paid run with lineage |
| Cancel | `POST /api/admin/train/{run_id}/cancel` |
| Select a passed published checkpoint | `POST /api/admin/checkpoints/{checkpoint_id}/activate` |

SFT needs individually owner-approved, source-supported examples, an eligible
published CPT parent with passed checks, and `synthetic_training_approved: true`
with a meaningful `provider_policy_reference` granting the intended teacher-output
use. A checkbox or x402 receipt does not grant training rights. SFT is not launched
by the four-hour scheduler. Held-out loss, frozen domain/general evaluations and
the Chamber hook screen must pass; authored smoke checks do not establish scientific
conclusions, felt pain or consciousness.

## Move a candidate into Chamber experiments

The selected Llama Base has no original chat template. Its CPT adapter is usable
for completion research; Observatory live-chat selection requires separately
validated SFT and its saved tokenizer. Selection records a checkpoint and exposes
a secret-free deployment environment download. It does **not** redeploy the public
relay or existing GPU endpoint.

Apply the downloaded exact base, adapter and tokenizer revisions/subfolders to a
new staging Chamber worker. For the 70B QLoRA path use `live/Dockerfile.worker70`
and an appropriate CUDA host; add infrastructure credentials separately. Newly
published private/gated artifacts need `HF_HUB_OFFLINE=0`, `HF_TOKEN` and a writable
`MODEL_ADAPTER_CACHE_DIR`. Retain the old model as a control. Do not load a Llama
Base adapter against the existing Hermes checkpoint simply because both are 70B.

Fresh intervention vectors are built for the adapted model, and the old base's
Jacobian lens is disabled. Exported dose caps start at zero. Nonzero adapted doses
require an owner-mounted `CHAMBER_ADAPTER_CALIBRATION` dose-sweep receipt matching
all base/adapter/tokenizer pins, subfolders, layer, dtype and quantization. Generic
cap overrides do not bypass that guard. Keep the actual measurement artifact;
see [the receipt schema](../X402_OPERATIONS.md#adapted-chamber-dose-receipt).
Calibrate and compare before routing public experiments to the new worker.
Keep relay `CHAMBER_WILD=0` during baseline/dose validation. Use `persona: false`
for controlled API requests, or deliberately match and log persona priming across
conditions. The upstream public UI's first-person persona and optional wild or
self-reading cycles are separate experimental conditions; Observatory Pause/Stop
does not control those relay cycles. See the experiment guide for interpretation.

## Staging acceptance and upstream merge

Record pass/fail, commit, account route, model pins, image digest and evidence for
each deployment gate:

- Build service images; verify runtime health, persistence, keys and restore on
  isolated staging data. Check missing owner token/invalid token protections and
  sanitized public state. Verify HTTPS, same-origin proxy/SSE/frame headers and
  network-level browser egress isolation.
- With training/automatic review disabled, run one bounded funded research pass.
  Verify the actual browser, notes, exact supporting passages, original text,
  provenance and honest blocked-site behavior. Test Pause/Stop and owned-session
  cleanup before enabling six agents or unattended missions.
- For x402, verify the explicit selected native model/actions/images and exact
  quote/settlement binding, reservations and limits. Test insufficient funding
  pause/resume and an uncertain-outcome recovery without duplicate authorization.
- Enable v2 automatic original review on a small eligible staging source queue.
  Check two-pass acceptance, manual exceptions, preserved classification and
  owner decisions, rights/extraction exclusions and interrupted-review recovery.
- Export/review a sealed snapshot; verify original CPT text, separate synthetic
  SFT, held-out families and exclusion/coverage reports. Freeze an independently
  reviewed evaluation suite appropriate to the experiment.
- Use a small supported training configuration to validate image pull, HF token,
  upload/job/result lifecycle and cancellation. Then measure actual 70B QLoRA
  memory, optimizer execution, evaluation, publication and restart recovery on
  the selected hardware before unattended scheduling.
- Complete separately permitted/reviewed SFT, load exact artifacts on a separate
  Chamber endpoint and measure dose calibration. Preserve the baseline and test
  rollback to that endpoint before public routing.
- Agree upstream merge/rollout policy and inventory deployment secrets. A local
  ready PR is not permission to deploy production or spend provider credits.

Existing workflows make merge operationally significant. On `master`, changes to
`live/server.py`, `live/worker.py`, worker Dockerfiles or requirements can build
GHCR images and repoint existing RunPod templates when `RUNPOD_API_KEY` is present.
`deploy-railway.yml` can deploy the current relay with `RAILWAY_TOKEN`; its missing
prior-commit fallback can deploy on a master push even without a detected `live/`
change. The Makefile records the existing Vercel site as auto-deploying on master.
Confirm the actual upstream host settings before merge. These existing workflows
do not deploy the Observatory's new sidecar/broker/HF training image or make an
activated checkpoint live; those remain deliberate owner deployment steps.
