import copy
from types import SimpleNamespace

from cryptography.fernet import Fernet
import pytest

from observatory.automatic_curation import AutomaticCurationWorker, SCOPE, digest, receipt_payload
from observatory.corpus_policy import quality_review_reasons
from observatory.curation import eligibility
from observatory.store import Store
from observatory.x402_client import ResearchFundingError, ResearchPermanentError, ResearchTransientError

QUOTE = "The paper compares mechanistic theories of machine consciousness and distinguishes reported pain from demonstrated phenomenal experience."


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSERVATORY_SECRET_KEY", Fernet.generate_key().decode())
    db = Store(tmp_path / "curation.sqlite3")
    db.save_settings({"auto_curation_enabled": True, "auto_curation_policy_ack": "originals-v2",
                      "research_provider": "x402", "research_model": "openai/review-fixture", "research_protocol": "responses"})
    db.set_mission({"status": "running", "objective": "Research machine consciousness"})
    yield db
    db.close()


def source(identifier="paper", **updates):
    return {"id": identifier, "text": (QUOTE + "\n\n") * 5, "title": "Mechanistic theories", "review_status": "pending",
            "canonical_url": "https://papers.example/" + identifier, "source_type": "article",
            "license": "CC-BY-4.0", "license_verified": True, "rights_evidence": "https://creativecommons.org/licenses/by/4.0/",
            "provenance": {"method": "structured_html", "title": "Mechanistic theories"},
            "extraction": {"method": "structured_html", "quality": "passed", "schema_version": "structured-extraction-v1"}, **updates}


def verdict(**updates):
    return {"decision": "accept", "topic_relevance": "relevant", "evidence_stance": "uncertain", "source_type": "empirical_paper",
            "covered_stances": [], "topic_domains": ["machine_consciousness"], "evidence_kind": "empirical",
            "quotes": [QUOTE], "rationale": "Substantive comparison distinguishes behavior from phenomenal experience.", **updates}


class FakeModel:
    def __init__(self, outputs=None, callback=None):
        self.outputs = outputs or [verdict(), verdict()]
        self.calls = []
        self.callback = callback

    async def ainvoke(self, messages, output_format):
        self.calls.append(copy.deepcopy(messages))
        if self.callback:
            self.callback(len(self.calls))
        value = self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)]
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(completion=output_format.model_validate(value))


@pytest.mark.asyncio
async def test_blind_acceptance_records_bound_receipt_without_mutating_other_gates(store):
    original = store.put("sources", source())
    model = FakeModel()
    result = await AutomaticCurationWorker(store).tick(model=model)
    assert result["status"] == "accepted"
    assert len(model.calls) == 2
    assert "Substantive comparison" not in model.calls[1][1].content
    assert "full_original_document" in model.calls[0][1].content
    assert QUOTE in model.calls[1][1].content
    accepted = store.get("sources", "paper")
    for key in ("text", "provenance", "rights_evidence", "license_verified", "extraction", "source_type"):
        assert accepted[key] == original[key]
    assert accepted["quality_review"]["reviewer_kind"] == "automated"
    assert accepted["review_status"] == "approved"
    assert accepted["eligibility"] == "eligible"
    assert eligibility(accepted)["eligible"]
    receipt = store.get("curation_reviews", result["review_id"])
    assert receipt["immutable"] is True
    assert [review["stage"] for review in receipt["reviews"]] == ["primary", "critic"]
    assert not quality_review_reasons(accepted)
    assert QUOTE not in str(store.events_after(0))


@pytest.mark.asyncio
async def test_final_receipt_and_accepted_source_do_not_repeat_paid_calls(store):
    store.put("sources", source())
    worker, model = AutomaticCurationWorker(store), FakeModel()
    first = await worker.review_source("paper", model=model)
    second = await worker.review_source("paper", model=model)
    third = await worker.tick(model=model)
    assert first["review_id"] == second["review_id"]
    assert second["cached"]
    assert third["status"] == "idle"
    assert len(model.calls) == 2
    assert len(store.list_records("curation_reviews")) == 1


@pytest.mark.asyncio
async def test_public_rationale_does_not_repeat_private_source_quotation(store):
    store.put("sources", source())
    model = FakeModel([verdict(rationale="The supporting passage is: " + QUOTE)] * 2)
    await AutomaticCurationWorker(store).tick(model=model)
    accepted = store.get("sources", "paper")
    assert QUOTE not in accepted["quality_review"]["rationale"]
    assert QUOTE not in str(store.events_after(0))
    assert QUOTE in accepted["quality_review"]["receipt"]["reviews"][0]["verdict"]["quotes"]


@pytest.mark.asyncio
@pytest.mark.parametrize("changes", [
    {"decision": "uncertain"}, {"decision": "reject"}, {"evidence_stance": "skeptical"},
    {"source_type": "article"}, {"evidence_stance": "mixed", "covered_stances": ["supportive", "skeptical"]},
])
async def test_critic_disagreement_or_nonacceptance_remains_manual(store, changes):
    store.put("sources", source())
    model = FakeModel([verdict(), verdict(**changes)])
    worker = AutomaticCurationWorker(store)
    result = await worker.tick(model=model)
    assert result["status"] == "manual_review"
    assert not eligibility(store.get("sources", "paper"))["eligible"]
    assert len(model.calls) == 2
    assert (await worker.tick(model=model))["status"] == "idle"
    assert len(model.calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("updates,reason", [
    ({"license_verified": False}, "rights_not_verified"),
    ({"extraction": {"quality": "unverified"}, "extraction_review_status": "approved", "extraction_review_evidence": "Checked PDF"}, "automatic_curation_requires_passed_extraction"),
    ({"text": QUOTE * 500}, "whole_document_exceeds_automatic_review_limit"),
    ({"contains_benchmark": True}, "benchmark_or_chamber_stimulus_excluded"),
    ({"held_out": True, "contamination_status": "suspected"}, "benchmark_or_chamber_stimulus_excluded"),
])
async def test_hard_gate_failures_do_not_spend_and_keep_split_flags(store, updates, reason):
    original = store.put("sources", source(**updates))
    model = FakeModel()
    result = await AutomaticCurationWorker(store).tick(model=model)
    assert result["status"] == "manual_review"
    assert reason in result["reasons"]
    assert not model.calls
    assert store.get("sources", "paper").get("held_out") == original.get("held_out")


@pytest.mark.asyncio
async def test_owner_fixes_prefilter_gates_then_same_document_gets_new_full_review(store):
    store.put("sources", source(license_verified=False, extraction={"quality": "unverified"}))
    worker, model = AutomaticCurationWorker(store), FakeModel()
    blocked = await worker.tick(model=model)
    assert blocked["status"] == "manual_review" and "-gates-" in blocked["review_id"]
    assert not model.calls
    updated = store.get("sources", "paper")
    updated.update(license_verified=True, extraction={"quality": "passed", "method": "structured_html"})
    store.put("sources", updated)
    accepted = await worker.tick(model=model)
    assert accepted["status"] == "accepted" and accepted["review_id"] != blocked["review_id"]
    assert len(model.calls) == 2
    assert len(store.list_records("curation_reviews")) == 2


@pytest.mark.asyncio
async def test_blocked_first_source_does_not_starve_shared_queue(store):
    store.put("sources", source("blocked", license_verified=False))
    store.put("sources", source("ready"))
    worker, model = AutomaticCurationWorker(store), FakeModel()
    assert (await worker.tick(model=model))["status"] == "manual_review"
    assert (await worker.tick(model=model))["status"] == "accepted"
    assert eligibility(store.get("sources", "ready"))["eligible"]


@pytest.mark.asyncio
async def test_fabricated_source_quote_quarantines_instead_of_becoming_training_evidence(store):
    store.put("sources", source())
    model = FakeModel([verdict(quotes=["This fabricated quotation falsely claims demonstrated phenomenal pain in a model."])])
    result = await AutomaticCurationWorker(store).tick(model=model)
    assert result["status"] == "manual_review"
    assert result["reasons"] == ["automatic_review_quote_not_exactly_in_original"]
    assert store.get_mission()["status"] == "running"
    assert not eligibility(store.get("sources", "paper"))["eligible"]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["manual", "content", "stopped", "paused", "disabled", "model", "mission"])
async def test_control_or_source_change_during_review_prevents_acceptance_and_manual_decision_wins(store, change):
    store.put("sources", source())

    def callback(number):
        if number != 2:
            return
        current = store.get("sources", "paper")
        if change == "manual":
            current.update(review_status="rejected", quality_review={"status": "rejected", "reviewer_kind": "human"})
            store.put("sources", current)
        elif change == "content":
            store.put("sources", {**current, "text": current["text"] + " New material."})
        elif change in {"stopped", "paused"}:
            store.set_mission({**store.get_mission(), "status": change})
        elif change == "disabled":
            store.save_settings({"auto_curation_enabled": False})
        elif change == "model":
            store.save_settings({"research_model": "openai/changed-model"})
        else:
            store.set_mission({"id": "new-mission", "status": "running", "objective": "Another mission"})

    result = await AutomaticCurationWorker(store).tick(model=FakeModel(callback=callback))
    assert result["status"] == "aborted"
    assert not store.list_records("curation_reviews")
    if change == "manual":
        assert store.get("sources", "paper")["review_status"] == "rejected"
    else:
        assert "quality_review" not in store.get("sources", "paper")


@pytest.mark.asyncio
async def test_existing_manual_review_is_never_overwritten(store):
    store.put("sources", source(quality_review={"status": "pending", "reviewed_by": "owner", "rationale": "I am reviewing this"}))
    model = FakeModel()
    assert (await AutomaticCurationWorker(store).tick(model=model))["status"] == "idle"
    assert not model.calls


@pytest.mark.asyncio
async def test_changed_research_brain_gets_new_receipt_not_prior_model_verdict(store):
    store.put("sources", source())
    worker, model = AutomaticCurationWorker(store), FakeModel()
    first = await worker.tick(model=model)
    store.save_settings({"research_model": "openai/new-model"})
    second = await worker.tick(model=model)
    assert first["review_id"] != second["review_id"]
    assert len(model.calls) == 4
    assert store.get("sources", "paper")["quality_review"]["model"] == "openai/new-model"


@pytest.mark.asyncio
async def test_pending_payment_never_retried_under_new_request(store):
    store.put("sources", source())
    request_id = store.reserve_research_request(SCOPE, "a" * 64, "responses", {"model": "openai/review-fixture", "input": "outstanding"})
    model = FakeModel()
    result = await AutomaticCurationWorker(store).tick(model=model)
    assert result["status"] == "failed" and not model.calls
    assert store.pending_research_requests(SCOPE)[0]["request_id"] == request_id
    assert store.get_mission()["status"] == "faulted"


@pytest.mark.asyncio
async def test_definitively_unpaid_failure_preserves_primary_and_retries_only_remaining_blind_stage(store):
    store.put("sources", source())
    model = FakeModel([verdict(), ResearchFundingError("UNFUNDED", "Research funding is unavailable.", definitively_unpaid=True), verdict()])
    worker = AutomaticCurationWorker(store)
    assert (await worker.tick(model=model))["status"] == "failed"
    assert store.get_mission()["status"] == "funding_paused"
    assert len(store.get("curation_work", SCOPE)["reviews"]) == 1
    store.set_mission({**store.get_mission(), "status": "running"})
    assert (await worker.tick(model=model))["status"] == "accepted"
    assert len(model.calls) == 3
    assert "Independently scrutinize" in model.calls[2][0].content


@pytest.mark.asyncio
async def test_unknown_failure_requires_explicit_recovery_and_keeps_manual_pause(store):
    store.put("sources", source())

    def callback(number):
        store.set_mission({**store.get_mission(), "status": "paused"})

    worker = AutomaticCurationWorker(store)
    model = FakeModel([RuntimeError("SECRET_PROVIDER_BODY")], callback=callback)
    assert (await worker.tick(model=model))["status"] == "failed"
    assert store.get_mission()["status"] == "paused"
    assert store.get("curation_work", SCOPE)["call_state"] == "awaiting_operator"
    store.set_mission({**store.get_mission(), "status": "running"})
    assert (await worker.tick(model=model))["status"] == "failed"
    assert len(model.calls) == 1
    assert "SECRET_PROVIDER_BODY" not in str(store.events_after(0))


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["content", "model", "manual_rejection", "gate_only"])
async def test_interrupted_call_globally_blocks_new_context_and_receipt_until_owner_recovery(store, change):
    store.put("sources", source())
    worker = AutomaticCurationWorker(store)
    failed_model = FakeModel([RuntimeError("Unknown provider outcome")])
    assert (await worker.tick(model=failed_model))["status"] == "failed"
    initial = store.get("curation_work", SCOPE)
    if change == "content":
        prior = store.get("sources", "paper")
        store.put("sources", {**prior, "text": prior["text"] + " Changed content."})
    elif change == "model":
        store.save_settings({"research_model": "openai/another-model"})
    elif change == "manual_rejection":
        prior = store.get("sources", "paper")
        store.put("sources", {**prior, "review_status": "rejected"})
        store.put("sources", source("next-source"))
    else:
        prior = store.get("sources", "paper")
        store.put("sources", {**prior, "license_verified": False})
    store.set_mission({**store.get_mission(), "status": "running"})
    fresh_model = FakeModel()
    assert (await worker.tick(model=fresh_model))["status"] == "failed"
    assert not fresh_model.calls
    assert store.get("curation_work", SCOPE) == initial
    assert not store.list_records("curation_reviews")


@pytest.mark.asyncio
async def test_direct_review_cannot_replace_another_sources_unresolved_call(store):
    store.put("sources", source())
    worker = AutomaticCurationWorker(store)
    await worker.tick(model=FakeModel([RuntimeError("Unknown outcome")]))
    initial = store.get("curation_work", SCOPE)
    store.put("sources", source("other"))
    store.set_mission({**store.get_mission(), "status": "running"})
    model = FakeModel()
    assert (await worker.review_source("other", model=model))["status"] == "failed"
    assert not model.calls
    assert store.get("curation_work", SCOPE) == initial


@pytest.mark.asyncio
async def test_authorized_recovery_preserves_completed_primary_then_runs_only_critic(store):
    store.put("sources", source())
    worker, model = AutomaticCurationWorker(store), FakeModel([verdict(), RuntimeError("Unknown critic outcome"), verdict()])
    assert (await worker.tick(model=model))["status"] == "failed"
    work = store.get("curation_work", SCOPE)
    assert work["call_state"] == "awaiting_operator" and len(work["reviews"]) == 1
    store.put("curation_work", {**work, "call_state": "retry_authorized"})
    store.set_mission({**store.get_mission(), "status": "running"})
    assert (await worker.tick(model=model))["status"] == "accepted"
    assert len(model.calls) == 3
    assert "Independently scrutinize" in model.calls[2][0].content


@pytest.mark.asyncio
async def test_live_busy_worker_is_not_mistaken_for_a_crashed_call(store):
    worker = AutomaticCurationWorker(store)
    store.put("curation_work", {"id": SCOPE, "status": "reviewing", "call_state": "in_flight"})
    await worker._lock.acquire()
    try:
        assert (await worker.tick(model=FakeModel()))["status"] == "busy"
    finally:
        worker._lock.release()
    assert store.get_mission()["status"] == "running"


@pytest.mark.asyncio
async def test_definitively_unpaid_transient_retry_is_bounded(store):
    store.put("sources", source())
    model = FakeModel([ResearchTransientError("UPSTREAM_UNAVAILABLE", "Unavailable", definitively_unpaid=True)])
    assert (await AutomaticCurationWorker(store).tick(model=model))["status"] == "failed"
    assert len(model.calls) == 2
    assert store.get_mission()["status"] == "faulted"


@pytest.mark.asyncio
async def test_receipt_content_bind_and_agreement_cannot_be_bypassed_by_rehashed_fields(store):
    store.put("sources", source())
    await AutomaticCurationWorker(store).tick(model=FakeModel())
    accepted = store.get("sources", "paper")
    changed = copy.deepcopy(accepted)
    changed["text"] += " A new claim."
    assert "automated_quality_review_receipt_invalid" in quality_review_reasons(changed)
    changed = copy.deepcopy(accepted)
    receipt = changed["quality_review"]["receipt"]
    receipt["reviews"][1]["verdict"]["evidence_stance"] = "supportive"
    receipt["receipt_hash"] = digest(receipt_payload(receipt))
    assert "automated_quality_review_receipt_invalid" in quality_review_reasons(changed)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["status", "actor", "reviewer", "rationale", "mutable", "extra_verdict_field"])
async def test_standalone_gpu_receipt_validator_enforces_actor_and_schema_without_sidecar_gates(store, change):
    from observatory.curation_receipts import automated_review_reasons
    store.put("sources", source())
    await AutomaticCurationWorker(store).tick(model=FakeModel())
    accepted = store.get("sources", "paper")
    review = accepted["quality_review"]
    if change == "status":
        review["status"] = "pending"
    elif change == "actor":
        review["reviewer_kind"] = "human"
    elif change == "reviewer":
        review["reviewed_by"] = " "
    elif change == "rationale":
        review["rationale"] = ""
    elif change == "mutable":
        review["receipt"]["immutable"] = False
    else:
        review["receipt"]["reviews"][0]["verdict"]["rights_override"] = True
        review["receipt"]["receipt_hash"] = digest(receipt_payload(review["receipt"]))
    assert automated_review_reasons({"id": "paper", "text": accepted["text"]}, review) == ["automated_quality_review_receipt_invalid"]


@pytest.mark.asyncio
async def test_sft_policy_and_notes_are_not_approved_by_original_curation(store):
    store.put("sources", source())
    note = store.put("notes", {"id": "draft", "source_id": "paper", "review_status": "pending", "question": "What?", "answer": "Answer"})
    await AutomaticCurationWorker(store).tick(model=FakeModel())
    assert store.get("notes", "draft") == note
    assert store.get_settings(private=True).get("synthetic_training_approved") is not True


@pytest.mark.asyncio
async def test_disabled_policy_or_stopped_mission_never_spends(store):
    store.put("sources", source())
    worker, model = AutomaticCurationWorker(store), FakeModel()
    store.save_settings({"auto_curation_enabled": False})
    assert (await worker.tick(model=model))["status"] == "idle"
    store.save_settings({"auto_curation_enabled": True, "auto_curation_policy_ack": "originals-v2"})
    store.set_mission({**store.get_mission(), "status": "stopped"})
    assert (await worker.tick(model=model))["status"] == "idle"
    assert not model.calls
