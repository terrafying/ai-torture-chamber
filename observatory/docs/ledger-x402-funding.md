# Funding research from a Ledger wallet through Solana x402

The implemented funding path lets an owner keep the main treasury in a Ledger
hardware wallet and manually transfer a bounded working balance to a separate
software spending wallet. The isolated payment broker uses that spending wallet
to buy compatible research-model inference through x402. Research agents can
then make paid requests without asking the owner to approve every request on the
hardware device.

This is an alpha integration. The repository includes the broker, payment
controls, receipt verification and recovery paths, but a live Ledger transfer,
gateway purchase and unattended deployment still require owner-run acceptance
testing. Nothing in this guide creates a wallet, transfers funds or enables
spending by itself. Use the [deployment guide](deployment-guide.md) for service
startup and the [payment operations guide](../X402_OPERATIONS.md) for exact API
and recovery procedures.

## The two wallets have different jobs

The **treasury wallet** is the owner's Ledger-controlled Solana account. Its key
stays on the hardware signer, and a transfer needs physical approval. Ledger
documents Solana and SPL-token management through Ledger Wallet and compatible
interfaces such as Phantom or Solflare. Check the current interface's supported
assets and verify the transaction on the hardware screen before approving it.
See [Ledger's Solana wallet guide](https://www.ledger.com/coin/wallet/solana) and
[Ledger's explanation of transaction signing](https://www.ledger.com/academy/how-does-a-crypto-wallet-work).

The **spending wallet** is a distinct Solana account with a software key supplied
only to the isolated broker through `X402_SOLANA_PRIVATE_KEY`. In this release,
the signer is the official x402 SDK's `KeypairSigner.from_base58`, pinned to
`x402[svm]==2.25.0`. The research process asks the broker for inference; it does
not receive the wallet key. The website cannot save a private key or seed phrase
through its settings API.

The intended movement of funds is:

```text
Ledger-controlled treasury
    │ owner-approved native Solana USDC transfer
    ▼
Dedicated software spending wallet
    │ broker-approved x402 inference payment
    ▼
Approved gateway merchant
    │ model response plus settlement receipt
    ▼
Research / original-document review workers
```

The working balance is an ordinary transfer, not an on-chain token allowance,
delegation, escrow or multisignature policy. The broker's request caps, daily cap
and reserve are software controls. The software key controls the spending
account; these controls do not constrain someone who obtains that key and signs
outside the broker. Keep the main treasury separate and choose the working
balance according to the owner's operational policy.

Use one persistent broker ledger for this working wallet. Separate brokers with
different databases do not share reservations or daily limits. Do not reuse the
spending key across independent deployments and assume those limits combine.

There is no Ledger USB/HID integration, hardware-wallet connection button,
unattended Ledger signing or automatic treasury top-up in this repository. Do
not enter the Ledger recovery phrase into the broker. “Payment ledger” elsewhere
in the documentation means the local SQLite record of requests and receipts; it
does not mean a Ledger hardware device.

## What this wallet pays for

The x402 path pays for the eligible model's research requests and, when enabled,
the automatic original-document review worker's model requests. It replaces a
direct model-provider API key for that route. It does not replace every service
account involved in running the system.

| Component | Funding route in this release |
| --- | --- |
| Research brain and automated document reviews | Approved Solana USDC x402 inference requests, or the explicitly selected direct OpenAI/Anthropic API route. |
| Local Chromium and research hosting | Owner's infrastructure. |
| Browser Use Cloud, if selected | Separate Browser Use API key and service billing. |
| Hugging Face 70B adaptation jobs | Separate Hugging Face token, account credits and GPU-job configuration. |
| Chamber model serving and experiments | Separate serving infrastructure and its costs. |

Paying for inference does not purchase rights to train on a crawled document.
Source provenance, access rules, extraction checks and dataset acceptance still
apply. Enabling the opt-in review worker adds model requests to the same funding
path; see [automatic original-document review](automatic-curation.md).

## Prepare the broker before funding it

Start from [`.env.payments.example`](../.env.payments.example) in a private,
server-side environment. `X402_ENABLED=false` is the default. The owner must
supply a dedicated signer, an internal broker token, explicit spending limits
and independently approved merchant recipients before purchases are allowed.
There are no default spending amounts.

The three amount settings accept explicit USDC values with at most six decimal
places. `X402_MAX_REQUEST_USDC` caps one quoted purchase;
`X402_DAILY_LIMIT_USDC` limits the broker's settled and reserved spending in a
UTC day; `X402_MIN_RESERVE_USDC` preserves a balance floor. Outstanding reserved
requests remain reserved across day boundaries. `X402_ALLOWED_PAY_TO` is the
owner-approved list of merchant recipient addresses, not a list populated from
web pages or from whatever recipient appears in a payment challenge.

Use an HTTPS Solana mainnet RPC URL through `X402_SOLANA_RPC_URL`. Keep the
broker private to the research service. Its signing and recovery routes require
the internal Bearer token. Its read-only health, catalog and funding routes do
not require that token; private network placement remains part of deployment.
The sidecar uses `OBSERVATORY_PAYMENT_BROKER_URL` and the matching
`OBSERVATORY_PAYMENT_BROKER_TOKEN` to reach it.

The currently implemented merchant transport is fixed to
`https://sol.blockrun.ai/api/v1`, using the native `/responses` or `/messages`
protocol. This is not a general wallet that pays arbitrary URLs, crawled sites
or GPU providers. Changing the merchant transport is a code change that needs
its own compatibility and payment checks.

## Find and verify the spending address

After the owner configures the software signer and starts the broker and
sidecar, read `GET /api/funding` from the trusted sidecar. With the static-site
proxy described in the deployment guide, the public route is
`/observatory-api/funding`. There is no Funding page or deposit form in the
current interface.

The response's `wallet_address` is derived from the broker signer. Compare it
with the public address of the owner's dedicated spending account before using
it as the destination. The response also identifies `network` and `asset`.
When spending is disabled, a valid signer can still publish its address, while
the reported status is `configured` and the balance is not checked. Missing or
invalid configuration and RPC failures must be resolved; an unknown balance is
not confirmation that the wallet is funded.

The implemented route accepts **Solana mainnet native USDC only**:

```text
Network: solana:5eykt4UsFv8P8NJdTREpY1vzqKqZKvdp
Mint:    EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v
```

This mint matches [Circle's official mainnet USDC address table](https://developers.circle.com/stablecoins/usdc-contract-addresses).
Tokens on other chains, bridged lookalikes and Solana devnet USDC cannot satisfy
this broker's quote or balance checks. The production configuration is not a
devnet payment sandbox.

Using the owner's Ledger-supported Solana wallet interface, prepare a normal
native-USDC transfer to the verified spending address. Check the network,
asset, recipient, transfer amount and fees on the hardware device, then approve
only the transfer the owner intends. The main treasury key is never supplied to
the Observatory. Ledger describes this manual approval flow in its
[sending and receiving guide](https://www.ledger.com/academy/topics/ledger-wallet/how-to-send-receive-spend-crypto-with-ledger-wallet?sitelearn=1).

## SOL fees and token accounts

There are two different transactions to account for. A normal owner top-up is
a Solana transaction with a network fee paid in SOL. Its fee payer needs SOL;
USDC is not the native fee token. See [Solana's transaction-fee documentation](https://solana.com/docs/core/fees).

A USDC balance lives in a token account, not directly in the wallet's system
account. This broker counts only the spending address's canonical associated
token account (ATA) for the pinned USDC mint. An absent ATA is treated as a zero
balance; USDC held in another token account is not included. Ensure the top-up
is delivered to that canonical ATA using a wallet/tool that handles the correct
mint and owner. If an ATA must be created, the account-creation payer must fund
the required SOL storage balance and the transaction fee payer must cover the
transaction fee. Those can be different parties. See [Solana's token-account creation guide](https://solana.com/docs/tokens/basics/create-token-account).

For the subsequent x402 purchase, the pinned SDK builds a partially signed
transaction containing compute-budget instructions, a `TransferChecked` USDC
instruction and a memo. It derives the spending and merchant ATAs but
does **not** create either account. Both must already exist. The quote must
include an external `extra.feePayer`; the broker rejects the spending wallet as
that fee payer. The sponsor/facilitator completes the transaction and pays its
SOL transaction fee. This sponsorship does not fund the owner's top-up or
create token accounts for the owner. The [x402 Solana exact-scheme specification](https://github.com/x402-foundation/x402/blob/main/specs/schemes/exact/scheme_exact_svm.md)
explains the sponsored fee-payer role; the shipped Python SDK and broker enforce
the narrower behavior above.

## Choose a compatible research model

Select `research_provider="x402"`, an exact catalog `research_model` ID and its
supported native `research_protocol` in authenticated owner settings. The
read-only `GET /api/research-models` route exposes the broker's normalized
catalog. A familiar vendor or frontier-model name alone does not establish that
the model can drive a browser.

The configured model must support screenshots and structured browser actions.
The `responses` route also requires structured outputs; `messages` requires
tool calling. Capabilities must come from usable catalog metadata or an exact
owner-reviewed declaration in `X402_RESEARCH_MODEL_CAPABILITIES`. A declaration
allows a controlled compatibility test; it is not evidence that the route
works. Do not enable unattended research until a real browser request succeeds
with the requested model and full context. The broker rejects declared
unsupported models and responses marked as model fallback or context
truncation. It does not silently switch to a different model or payment route.

The owner can choose among compatible models actually offered by this gateway.
Using a direct OpenAI or Anthropic API key is a separate, explicit provider
choice. The protocol itself does not make every model provider or GPU service
available through this broker.

## What happens during a purchase

Before allocating a browser, research preflight checks the broker, chosen model
and funding. “Ready” conservatively requires the verified USDC balance to cover
existing reservations, the minimum reserve and one request at the configured
maximum price, with enough daily headroom. Individual quotes can be smaller.

For each inference request, the broker persists a UUID and the exact request
content before requesting a quote. The gateway returns HTTP 402 with
`PAYMENT-REQUIRED`. The broker accepts only a v2 `exact` quote for its fixed
resource, network, mint, approved recipient, bounded amount, expiry and external
fee payer. Nonempty seller-provided memos are rejected before signing: this route
requires the pinned SDK to create a fresh client nonce, so an unrelated old
same-amount transfer cannot match a reused merchant memo. A gateway that requires
its own memo needs a separately validated transport; its quote is not rewritten.
It then atomically reserves funds, signs through the pinned SDK and
saves the authorization before retrying the same request with
`PAYMENT-SIGNATURE`.

The gateway returns model output and `PAYMENT-RESPONSE`. These are the x402 v2
HTTP headers described in [Solana's x402 protocol guide](https://solana.com/docs/payments/agentic-payments/x402).
The broker checks the reported settlement through read-only Solana RPC against
the exact USDC amount, mint, source, merchant ATA, transfer authority and
recorded SDK transaction memo. A transaction already attributed to another request
cannot settle a second request in the persistent ledger. A successful model response
alone is not settlement proof.
Paid response bodies are cached encrypted for same-ID recovery.

## Low funds, uncertain receipts and restarting

Insufficient verified funds or daily headroom puts the mission into
`funding_paused` and releases its owned browsers. A manual top-up can restore
the balance; a daily cap may need the next UTC day or an owner policy change.
The supervisor periodically rechecks funding and can resume that same mission
once preflight passes. It respects an owner's explicit Pause or Stop.
Authentication, invalid configuration and incompatible-response faults need
operator correction and explicit restart.

If a request was authorized and its settlement cannot be established, the
broker retains the reservation and blocks new spending. Sending more USDC does
not clear that uncertainty. Inspect the owner-only research-payment recovery
routes and reconcile the existing transaction through `POST /v1/reconcile` on
the private broker. Reconciliation verifies the existing memo-bound transfer;
it cannot sign or send another payment. A valid cached response can be recovered
without another purchase. Do not replace the request UUID, delete the ledger or
replay old browser actions as a shortcut. Follow [the recovery procedure](../X402_OPERATIONS.md#funds-reservations-and-uncertain-settlement).

Keep `X402_SOLANA_PRIVATE_KEY`, internal tokens and encryption keys out of Git,
screenshots, browser settings and logs. The broker's payment database and
response-encryption key must be backed up together consistently; the research
database and its own encryption key are separate recovery materials. Use the
private runtime environment or a deployment secret store to deliver the broker
key. A hardware seed is never a substitute for the dedicated software key.
Read the [deployment guide's persistence section](deployment-guide.md#credentials-persistence-and-recovery-materials)
before rotating keys or moving a service.

The implementation references are [the broker](../x402_broker.py),
[the public funding API](../app.py), [the pinned payment dependencies](../requirements-payments.txt)
and [the owner-supplied payment settings](../.env.payments.example). Official
external references above were reviewed on 6 October 2026. They describe
upstream products and protocols; the repository's live compatibility must still
be demonstrated in the owner's staging environment.
