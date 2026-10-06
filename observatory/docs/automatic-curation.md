# Automatic acceptance of collected originals

Original documents are the author-written papers and articles collected by the
research browsers. They are separate from agent summaries, screenshots, browsing
logs and generated question-and-answer drafts. No seed upload is required before
the first crawl. The research model is already trained; the selected 70B training
base is also pretrained. This framework adapts that base on accepted collected text.

## Enable the policy

Operator setup provides **Enable automated original-document curation**. The
authenticated settings request must explicitly select both:

```json
{
  "auto_curation_enabled": true,
  "auto_curation_policy_ack": "originals-v2"
}
```

The default is off. The worker runs while the research mission is running, with
one review in flight. It uses the configured research model and the existing
direct-provider or x402 billing route. There is no new wallet or browser session.
The existing daily/request payment limits still apply. Pause/Stop and disabling
automatic curation prevent new acceptance; provider/funding faults remain visible.
The broadened policy is `automatic-originals-v2`. Existing v1 acknowledgements
leave this worker idle until the owner explicitly enables the new policy. Historical
v1 receipts retain their original schema and scope; they are not relabeled.

## What it accepts

The worker reads the shared source queue. Code first checks verified rights,
provenance, topic-corpus exclusions and passed extraction. It does not grant rights,
clear contamination flags, change held-out assignments or approve PDF fidelity.
Unverified PDF extraction, unclear reuse permission and documents over the whole-text
review limit remain available for operator review. The default limit is 60,000
characters; documents are not silently truncated for acceptance.

A primary review and a blind critique pass each examine the same source text.
They classify relevance, source type and argument stance, record reasons and cite
literal supporting passages. Acceptance requires agreement and valid source
passages, in addition to the code checks. A model confidence score alone cannot
approve a document. Agreement does not certify scientific truth or consciousness.
The expanded mission includes consciousness science, philosophy of mind, reality
and metaphysics, and religious/contemplative perspectives as well as AI and welfare.
Both passes must also agree on reviewed research areas and the basis of claims.
General material without an AI-consciousness argument uses a `not_applicable`
machine stance with no machine-perspective coverage. Religious and philosophical
accounts can be accepted as attributed arguments or interpretations.

The two passes use separate prompts with the configured model; the critique is not
shown the primary verdict. They are separate checks, not proof of statistically
independent judgment. Rejected or uncertain material remains reviewable. A recorded
operator decision takes precedence over a late automatic result.

## Audit and training

Each result has an immutable `curation_reviews` record with source identity,
content hash, model, policy, verdicts and supporting passages. Accepted source
quality reviews identify their automated origin and retain the bound receipt.
The public Evidence view shows decisions, reasons and model identity; raw review
passages remain private. Restarted work reuses its stored progress and receipt;
an unresolved x402 request must be reconciled before another paid call.
Direct-provider SDK retries are disabled for these reviews. The worker allows a
single retry only after a definitively unpaid transient failure; an unknown
outcome requires operator recovery.

An interrupted model call without a saved verdict leaves an operator-attention
marker. First inspect and reconcile any pending payment using the existing payment
controls. While the mission is paused, faulted or stopped and the worker is idle,
the authenticated **Review recovery** control clears only that
marker. It preserves completed verdicts and requires the exact recorded review ID.
The interrupted call may already have been billed; this control explicitly permits
a new attempt, and does not resume the mission. The API equivalent is
`POST /api/admin/curation/recovery` with `{"reviewed": true, "review_id": "..."}`.
Busy work, unresolved broker intents and changed review identities are refused.

Existing snapshot rules still remove duplicates, preserve source-family holdouts,
exclude registered evaluation/Chamber material and require perspective coverage.
Original source text enters CPT. The four-hour training schedule still requires
configured/enabled HF training, enough accepted documents and actual tokenizer
counts, a changed corpus/recipe, and no overlapping job. GPU-side validation checks
automated receipt bindings again before consuming the original text.
The shared receipt validator uses only the standard library and is copied into
the GPU image; validation does not import the browser or payment services.

This setting does not change SFT approval. Agent-written Q&A remains separate and
requires the existing example review, supported-source and teacher-output policy
gates. Adapter evaluation, publication and Chamber deployment remain separate stages.
