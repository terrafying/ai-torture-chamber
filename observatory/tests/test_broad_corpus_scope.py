"""Broader consciousness originals retain subject and epistemic distinctions."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from observatory.automatic_curation import AutomaticCurationWorker
from observatory.corpus_policy import coverage_audit, quality_review_reasons, reviewed_perspectives
from observatory.curation import POLICY_VERSION as CORPUS_POLICY, build_snapshot, canonical_hash, eligibility, family_id
from observatory.curation_receipts import (LEGACY_POLICY_VERSION, LEGACY_RECEIPT_SCHEMA,
    automated_review_reasons, classification_reasons, digest, policy_for,
    receipt_identity, receipt_payload, text_hash, validate_verdict)
from observatory.store import Store


PHILOSOPHY_TEXT = (
    "This philosophical argument examines subjective experience, the relation of mind to reality, "
    "and how contemplative traditions describe awareness. It distinguishes conceptual arguments "
    "and religious testimony from experimentally supported accounts of consciousness. "
) * 4


def original(identifier="philosophy", **updates):
    return {"id": identifier, "canonical_url": "https://originals.example/" + identifier,
        "title": "Mind and experience", "text": PHILOSOPHY_TEXT,
        "source_type": "article", "review_status": "pending", "license": "cc-by-4.0",
        "license_verified": True, "rights_evidence": "Authored test fixture rights",
        "provenance": {"method": "authored_test_fixture"}, "extraction": {"quality": "passed"}, **updates}


def broad_verdict(**updates):
    return {"decision": "accept", "topic_relevance": "relevant", "evidence_stance": "not_applicable",
        "source_type": "article", "covered_stances": [],
        "topic_domains": ["metaphysics_reality", "philosophy_of_mind"],
        "evidence_kind": "philosophical_argument", "quotes": [PHILOSOPHY_TEXT[:180]],
        "rationale": "A substantive conceptual account of consciousness without a machine-consciousness claim.", **updates}


class Model:
    def __init__(self, verdicts=None):
        self.verdicts, self.calls = verdicts or [broad_verdict(), broad_verdict()], []

    async def ainvoke(self, messages, output_format=None):
        self.calls.append(messages)
        return SimpleNamespace(completion=output_format.model_validate(self.verdicts[len(self.calls) - 1]))


@pytest.fixture
def store(tmp_path):
    db = Store(tmp_path / "scope.db")
    db.save_settings({"research_provider": "openai", "research_model": "fixture-brain",
        "auto_curation_enabled": True, "auto_curation_policy_ack": "originals-v2"})
    db.set_mission({"id": "broad-mission", "status": "running"})
    yield db
    db.close()


def training_row(source):
    return {"id": source["id"], "text": source["text"], "split": "train",
            "quality_review": source["quality_review"], "reviewed_perspectives": reviewed_perspectives(source)}


@pytest.mark.asyncio
async def test_philosophical_original_is_accepted_without_invented_machine_evidence(store):
    collected = original()
    store.put("sources", collected)
    model = Model()
    result = await AutomaticCurationWorker(store).tick(model=model)
    assert result["status"] == "accepted" and len(model.calls) == 2
    accepted = store.get("sources", collected["id"])
    assert eligibility(accepted)["eligible"]
    assert accepted["text"] == collected["text"]
    assert accepted["quality_review"]["topic_domains"] == ["metaphysics_reality", "philosophy_of_mind"]
    assert accepted["quality_review"]["evidence_kind"] == "philosophical_argument"
    assert reviewed_perspectives(accepted) == []
    assert "Religion and metaphysics" in model.calls[0][0].content
    assert "not_applicable and []" in model.calls[0][0].content


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["religious_contemplative", "philosophical_argument", "mixed"])
async def test_broad_theoretical_format_cannot_fill_scientific_or_machine_floor(store, kind):
    store.put("sources", original())
    verdict = broad_verdict(source_type="theoretical_paper", evidence_kind=kind,
                            topic_domains=["religion_contemplation"])
    assert (await AutomaticCurationWorker(store).tick(model=Model([verdict, verdict])))["status"] == "accepted"
    audit = coverage_audit([training_row(store.get("sources", "philosophy"))])
    assert audit["splits"]["train"]["scientific_documents"] == 0
    assert audit["splits"]["train"]["perspectives"] == {}
    assert audit["splits"]["train"]["evidence_kinds"] == {kind: 1}
    assert set(audit["reasons"]) == {"missing_train_scientific_source",
        "missing_train_perspective:supportive", "missing_train_perspective:skeptical",
        "missing_train_perspective:uncertain"}


@pytest.mark.asyncio
@pytest.mark.parametrize("change", [
    {"topic_domains": ["philosophy_of_mind"]}, {"evidence_kind": "religious_contemplative"},
])
async def test_blind_scope_or_evidence_disagreement_requires_manual_review(store, change):
    store.put("sources", original())
    model = Model([broad_verdict(), broad_verdict(**change)])
    result = await AutomaticCurationWorker(store).tick(model=model)
    assert result["status"] == "manual_review" and len(model.calls) == 2
    assert not eligibility(store.get("sources", "philosophy"))["eligible"]


@pytest.mark.parametrize("changes,reason", [
    ({"topic_domains": []}, "reviewed_topic_domains_required"),
    ({"evidence_kind": "unclassified"}, "reviewed_evidence_kind_required"),
    ({"covered_stances": ["supportive"]}, "not_applicable_stance_cannot_cover_machine_perspectives"),
    ({"evidence_stance": "skeptical"}, "machine_perspective_requires_machine_consciousness_domain"),
    ({"evidence_stance": "methodological"}, "general_consciousness_requires_not_applicable_machine_stance"),
])
def test_scope_classification_prevents_machine_perspective_fabrication(changes, reason):
    verdict = broad_verdict(**changes)
    assert reason in classification_reasons(verdict, allow_legacy=False)
    with pytest.raises(ValueError, match=reason):
        validate_verdict(verdict, original())


def legacy_approved_source():
    source = original("legacy-machine")
    policy = policy_for({"research_provider": "openai", "research_model": "fixture-brain"}, version=LEGACY_POLICY_VERSION)
    verdict = {"decision": "accept", "topic_relevance": "relevant", "evidence_stance": "uncertain",
        "source_type": "theoretical_paper", "covered_stances": [], "quotes": [source["text"][:180]],
        "rationale": "Archived v1 original review."}
    receipt = {"id": receipt_identity(source, policy), "schema_version": LEGACY_RECEIPT_SCHEMA,
        "immutable": True, "policy": policy, "policy_hash": digest(policy), "source_id": source["id"],
        "source_content_hash": text_hash(source), "model": policy["model"], "provider": policy["provider"],
        "mission_id": "archived-mission", "decision": "accepted", "reasons": [], "rationale": verdict["rationale"],
        "reviews": [{"stage": stage, "verdict": deepcopy(verdict)} for stage in ("primary", "critic")]}
    receipt["receipt_hash"] = digest(receipt_payload(receipt))
    source["quality_review"] = {key: verdict[key] for key in ("topic_relevance", "evidence_stance", "source_type", "covered_stances")}
    source["quality_review"].update(status="approved", reviewed_by="automatic-curation", rationale=verdict["rationale"],
        reviewer_kind="automated", approval_basis="owner_policy", policy_version=LEGACY_POLICY_VERSION,
        review_id=receipt["id"], model=policy["model"], source_content_hash=text_hash(source), policy_hash=digest(policy), receipt=receipt)
    return source


def test_archived_v1_receipt_validates_with_exact_legacy_schema_and_reported_legacy_counts():
    source = legacy_approved_source()
    assert automated_review_reasons(source, source["quality_review"]) == []
    assert quality_review_reasons(source) == []
    audit = coverage_audit([training_row(source)])
    assert audit["splits"]["train"]["topic_domains"] == {"legacy_unspecified": 1}
    assert audit["splits"]["train"]["evidence_kinds"] == {"legacy_unspecified": 1}
    assert audit["splits"]["train"]["legacy_perspectives"] == {"uncertain": 1}
    assert audit["splits"]["train"]["legacy_scientific_documents"] == 1


@pytest.mark.parametrize("change", ["review_tags", "verdict_tags", "policy_criteria", "receipt_schema"])
def test_archived_v1_receipt_cannot_be_relabelled_as_broader_policy(change):
    source = legacy_approved_source()
    review, receipt = source["quality_review"], source["quality_review"]["receipt"]
    if change == "review_tags":
        review.update(topic_domains=["religion_contemplation"], evidence_kind="religious_contemplative")
    elif change == "verdict_tags":
        for item in receipt["reviews"]:
            item["verdict"].update(topic_domains=["machine_consciousness"], evidence_kind="scientific_theory")
    elif change == "policy_criteria":
        receipt["policy"]["criteria"] = "consciousness-originals-quality-v2"
        receipt["policy_hash"] = review["policy_hash"] = digest(receipt["policy"])
    else:
        receipt["schema_version"] = "automatic-original-review-v2"
    receipt["receipt_hash"] = digest(receipt_payload(receipt))
    assert automated_review_reasons(source, review) == ["automated_quality_review_receipt_invalid"]


@pytest.mark.asyncio
async def test_legacy_owner_acknowledgment_does_not_silently_authorize_broader_paid_reviews(store):
    store.save_settings({"auto_curation_policy_ack": "originals-v1"})
    store.put("sources", original())
    model = Model()
    assert (await AutomaticCurationWorker(store).tick(model=model))["status"] == "idle"
    assert not model.calls


@pytest.mark.asyncio
async def test_broad_accepted_original_is_kept_in_v3_snapshot_with_distinct_metadata(store):
    source = original()
    store.put("family_splits", {"id": canonical_hash(family_id(source)), "split": "train"})
    store.put("sources", source)
    await AutomaticCurationWorker(store).tick(model=Model())
    snapshot = build_snapshot(store)
    row = snapshot["original_text"][0]
    assert CORPUS_POLICY == snapshot["policy_version"] == "consciousness-corpus-v3"
    assert row["text"] == source["text"] and not row["synthetic"]
    assert row["quality_review"]["evidence_kind"] == "philosophical_argument"
    assert row["reviewed_perspectives"] == []
    assert not snapshot["quality_gate"]["ready"]
