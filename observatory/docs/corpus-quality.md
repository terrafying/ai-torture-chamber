# Corpus review and original-document training

Policy `consciousness-corpus-v3` trains continued pretraining (CPT) on reviewed,
eligible original document text. Screenshots, action logs and free-form agent
observations remain research records. Separately reviewed, source-supported
instruction conversations can enter SFT only after the existing owner and
teacher-output authorization gates pass.

## Review contract

An article's machine-readable CC license can establish a rights candidate. It
does **not** grant quality approval, relevance approval or extraction fidelity.
Every collected or manually imported original requires a source quality review.
The owner can supply that review directly:

```json
{
  "quality_review": {
    "status": "approved",
    "reviewed_by": "operator",
    "rationale": "Reviewed the paper's scope, argument and evidence limitations.",
    "topic_relevance": "relevant",
    "evidence_stance": "uncertain",
    "source_type": "empirical_paper",
    "topic_domains": ["machine_consciousness", "consciousness_science"],
    "evidence_kind": "empirical"
  }
}
```

Alternatively, the opt-in [automatic originals policy](automatic-curation.md)
can supply quality approval after two blind, source-grounded review passes agree.
These approvals are explicitly labeled `reviewer_kind: automated`, are bound to
the exact source text, model and policy, and retain an immutable review receipt.
They are not presented as human reviews. Deterministic rights, extraction and
contamination requirements still apply. The worker cannot approve unknown rights,
override failed extraction, remove evaluation exclusions or authorize teacher output.

`evidence_stance` is one of `supportive`, `skeptical`, `uncertain`, `mixed`,
`methodological`, or `not_applicable`. It classifies the document's argument about machine
consciousness/experience; it is not a truth label, a judgment of an author's
personality, or proof that a model experiences anything. Unknown classification
remains discovery-only. `source_type` is `empirical_paper`, `theoretical_paper`,
`review_paper`, `technical_report`, `article`, `reference`, or `social`.

For a mixed or methodological document, an owner may explicitly supply
`covered_stances: ["supportive", "skeptical", "uncertain"]` when the document
actually treats those perspectives. Omitting that field does not infer coverage.
A document with a single stance cannot claim a contradictory coverage list.

New broad reviews pair `topic_domains` with `evidence_kind`. Research areas cover
machine consciousness, consciousness science, philosophy of mind, reality and
metaphysics, religion and contemplation, and welfare and ethics. The basis of claims
is `empirical`, `scientific_theory`, `philosophical_argument`,
`religious_contemplative`, or `mixed`. General sources without a machine-consciousness
argument use `not_applicable` with empty covered perspectives. Supporting machine
perspectives requires the `machine_consciousness` area; philosophical or religious
views are not silently turned into evidence for AI sentience. The two automatic
review passes must agree on these classifications as well as the existing fields.
Historical manual reviews without either field retain their legacy classification;
new partial or inconsistent classifications are rejected.

Rights, attribution, provenance, a canonical URL and sufficient original text
are still required. Social content additionally requires documented permission;
changing the review classification cannot bypass the originally collected
social source type or recognized social hostname.

## Extraction fidelity

Structured extraction metadata has `schema_version`, `method`, `quality`,
`warnings`, `structure_preserved` and metrics. `quality: passed` indicates basic
readability checks passed, not that claims or equations are scientifically
correct. PDF extraction remains `unverified` until inspected against the source.

Unverified PDF output and legacy manual imports without extraction metadata
require both `extraction_review_status: approved` and a nonempty
`extraction_review_evidence` describing the fidelity check. `poor`, `failed`,
or an unrecognized extraction quality always excludes the original even if
someone supplies an approval field. There is no legacy auto-approval shortcut.

## Duplicate removal and lineage

Five-word shingles and bottom-k fingerprints find candidate near duplicates.
Actual overlap at or above 0.85 Jaccard, or identical normalized text, can form a
duplicate cluster. These are approximate lexical checks, not semantic identity
or guaranteed recall. Each removed text is compared directly against its chosen
representative; transitive similarity alone does not discard a distinct endpoint.
Changes to numbers, negation or mathematical relationship operators retain a
separate original even when most prose matches. Exact content hashes normalize
whitespace while preserving punctuation and case-sensitive scientific variables.

The retained representative prefers a passed extraction, then the more complete
text, with a deterministic ID tie-break. It has unit training weight. The
snapshot retains every member's `source_id`, URL, family, version, content hash,
rights, provenance and quality review in `source_lineage`. Removed members have
zero added weight and are listed in `manifest.deduplicated_sources`. The same
paper mirrored on ten sites therefore does not become ten CPT examples.

Distinct document versions that fail the duplicate threshold remain separate
originals but share a source-family split. An explicit held-out member makes
the group held out. Persistent assignments prevent already-trained families
from being relabeled as fresh validation data; conflicts quarantine the group.

## Coverage audit and selection

Every unique eligible original is selected once. The export deterministically
interleaves reviewed area, claim-basis, stance and source-type strata without fabricating examples,
repeating scarce documents or claiming equal proportions. Training consumers
may shuffle examples; interleaving does not impose a target distribution.

`coverage_audit` reports document counts, research areas, basis of claims, source
types, machine perspectives and character shares for train and validation separately.
Area labels can overlap, so their counts/shares need not sum to the document total
or one. Legacy unclassified material is identified explicitly. The launch floor requires
the **training** split to cover supportive, skeptical and uncertain arguments,
plus at least one scientific source. Newly classified scientific formats also need
an empirical or scientific-theory basis: a theoretical theology paper cannot fill
that floor just because it is a paper. Legacy contributions are counted separately
in the audit. Held-out coverage cannot fill a training
gap. `quality_gate.ready` is false until the floor is met, with explicit reasons.

This is a presence floor, **not statistical balance**. A four-to-one stance
distribution remains four-to-one and is visible in the audit. Operators should
inspect source quality, theory coverage, lengths and repeated claims before a
large run. No evidence-based universal theory ratio is assumed by this policy.

## Evaluation and Chamber contamination

Originals and SFT conversations are excluded if marked `contains_benchmark`,
`chamber_stimulus`, or `experimental_stimulus`, or if contamination status is
`suspected`, `confirmed`, `benchmark`, `evaluation`, `chamber_stimulus`, or
`experimental_stimulus`. Same-family versions and detected near duplicates of
excluded originals are also quarantined.

The source registry of the frozen evaluation suite reserves family IDs, URLs
and exact normalized task prompts. The active Chamber repository and experiment
pages are excluded as experiment sources. Long literal stimuli are read from
`live/server.py` using Python's syntax tree without importing or executing the
GPU server. Originals or conversations containing those passages are excluded.

These checks cannot detect all paraphrases, translations, historical pretraining
exposure or undisclosed custom stimuli. Register custom evaluation sources and
mark new Chamber stimuli before collecting a training snapshot. Dataset
validation in the GPU worker independently checks the frozen suite. Experiment
results still require an unchanged base-model control and a measured adapter
comparison; learning the language of pain is not evidence of subjective pain.

## Snapshot compatibility

Previously sealed snapshots remain immutable and retain their historical
policy version. They are not silently upgraded or re-reviewed. New v3 snapshots
must pass the v3 gates. Archived v2 manifests remain loadable under their historical
contract. Existing approved records retain their original meaning; untagged reviews
are reported as `legacy_unspecified`, not assigned guessed areas. Re-review them to
add new classifications. Test fixtures state their
review and extraction approval explicitly rather than enabling a production
quality bypass.
