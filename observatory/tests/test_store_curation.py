from concurrent.futures import ThreadPoolExecutor
import json

from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
import pytest

from observatory.app import create_app
from observatory.curation import build_snapshot, eligibility
from observatory.store import Store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSERVATORY_SECRET_KEY", Fernet.generate_key().decode())
    db = Store(tmp_path / "state.sqlite3")
    yield db
    db.close()


def source(identifier="paper1", **updates):
    return {"id": identifier, "canonical_url": "https://arxiv.org/abs/2411.02432v1",
            "license": "CC-BY-4.0", "license_verified": True,
            "rights_evidence": "https://creativecommons.org/licenses/by/4.0/",
            "provenance": {"title": "Pain tradeoffs", "authors": ["Researcher"], "version": "v1"},
            "text": ("The experimental result describes stipulated pain rather than demonstrated phenomenal experience. " * 15),
            "quality_review": {"status": "approved", "reviewed_by": "fixture-owner", "rationale": "Reviewed original scientific source",
                               "topic_relevance": "relevant", "evidence_stance": "uncertain", "source_type": "empirical_paper"},
            "extraction_review_status": "approved", "extraction_review_evidence": "Fixture original text compared with supplied source",
            **updates}


def test_persistence_encryption_and_public_redaction(store):
    secret = "sk-test-very-secret-value-123456"
    store.save_settings({"openai_api_key": secret, "training_enabled": False, "token_budget": 1000})
    store.put("agents", {"id": "a", "cdp_url": "wss://control.example/session", "frame_url": "https://frames.example/private"})
    store.put("notes", {"id": "n", "text": f"Accidental secret {secret}; Bearer abcdefg", "frame_url": "https://control.example/private"})
    store.event("step", f"Saw {secret}", data={"authorization": "Bearer private", "cdp_url": "wss://control.example/session"})
    state = store.state()
    serialized = json.dumps(state)
    assert secret not in serialized
    assert "control.example" not in serialized
    assert "abcdefg" not in serialized
    assert state["notes"][0]["text"].startswith("Accidental secret [REDACTED]")
    assert state["settings"]["openai_api_key"] == "***"
    assert state["settings"]["openai_api_key_configured"] is True
    assert state["settings"]["token_budget"] == 1000
    assert secret.encode() not in store.path.read_bytes()
    second = Store(store.path)
    try:
        assert second.get_settings(private=True)["openai_api_key"] == secret
        assert second.get("notes", "n")["text"].startswith("Accidental secret")
    finally:
        second.close()


def test_masked_settings_do_not_replace_credentials(store):
    store.save_settings({"hf_token": "hf_original_credentials"})
    store.save_settings({"hf_token": "***", "hf_token_configured": True})
    assert store.get_settings(private=True)["hf_token"] == "hf_original_credentials"
    with pytest.raises(ValueError):
        store.save_settings({"admin_token": "cannot_save_owner_token"})


def test_automated_review_receipts_are_immutable_and_public_state_hides_passages(store):
    review = store.put("curation_reviews", {"id": "review-fixture", "immutable": True,
        "decision": "accepted", "model": "fixture-reviewer", "rationale": "Source qualified",
        "reviews": [{"stage": "primary", "verdict": {"rationale": "Private detailed review quotation", "quotes": ["Original copyrighted passage must stay private"]}}],
        "receipt": {"quotes": ["Original copyrighted passage must stay private"]}})
    with pytest.raises(ValueError, match="immutable"):
        store.put("curation_reviews", {**review, "decision": "rejected"})
    public = store.state()["curation_reviews"][0]
    assert public["decision"] == "accepted" and public["model"] == "fixture-reviewer"
    assert "receipt" not in public
    assert "reviews" not in public
    assert "Private detailed review quotation" not in json.dumps(public)
    assert "Original copyrighted passage" not in json.dumps(public)


def test_generated_key_file_is_persistent_and_protected(tmp_path, monkeypatch):
    import os
    monkeypatch.delenv("OBSERVATORY_SECRET_KEY", raising=False)
    db = Store(tmp_path / "generated.sqlite3")
    db.save_settings({"hf_token": "hf_generated_key_secret"})
    db.close()
    key = tmp_path / "generated.sqlite3.key"
    assert key.is_file()
    if os.name != "nt":
        assert key.stat().st_mode & 0o077 == 0
    reopened = Store(tmp_path / "generated.sqlite3")
    try:
        assert reopened.get_settings(private=True)["hf_token"] == "hf_generated_key_secret"
    finally:
        reopened.close()


def test_parallel_event_sequence_and_reconnect_cursor(store):
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda number: store.event("step", str(number)), range(40)))
    events = store.events_after(20)
    assert [event["seq"] for event in events] == list(range(21, 41))


@pytest.mark.parametrize("updates,reason", [
    ({"license": "unknown"}, "license_not_in_public_corpus_policy"),
    ({"license": "CC-BY-NC-SA-4.0"}, "license_not_in_public_corpus_policy"),
    ({"license": "arXiv-nonexclusive-distribution-1.0"}, "license_not_in_public_corpus_policy"),
    ({"license_verified": False}, "rights_not_verified"),
    ({"rights_evidence": None}, "missing_rights_evidence"),
    ({"provenance": None}, "missing_source_provenance"),
    ({"source_type": "social"}, "social_content_requires_separate_permission"),
])
def test_source_eligibility_excludes_unsupported_rights(updates, reason):
    result = eligibility(source(**updates))
    assert result["eligible"] is False
    assert reason in result["reasons"]


def test_snapshot_is_idempotent_immutable_and_separates_source_versions(store):
    store.put("sources", source("first"))
    store.put("sources", source("second", canonical_url="https://arxiv.org/abs/2411.02432v2", split="heldout"))
    store.put("sources", source("unlicensed", license="unknown"))
    snapshot = build_snapshot(store)
    assert build_snapshot(store) == snapshot
    assert snapshot["train_family_ids"] == []
    assert snapshot["heldout_family_ids"] == ["arxiv:2411.02432"]
    assert snapshot["counts"]["original_documents"] == 1
    assert snapshot["counts"]["excluded_sources"] == 1
    assert snapshot["original_text"][0]["split"] == "validation"
    with pytest.raises(ValueError, match="immutable"):
        store.put("datasets", {**snapshot, "status": "changed"})
    assert "original_text" not in store.state()["datasets"][0]
    assert "text" not in store.state()["sources"][0]


def test_sft_requires_real_verified_passages_and_holds_out_source_family(store):
    store.save_settings({"synthetic_training_approved": True, "provider_policy_reference": "Owner-verified teacher-output contract"})
    store.put("sources", source("paper", split="heldout"))
    store.put("notes", {"id": "raw", "source_id": "paper", "text": "This might be good"})
    store.put("evidence", {"id": "e", "source_id": "paper", "passage": "stipulated pain rather than demonstrated phenomenal experience", "support_verified": True})
    store.put("notes", {"id": "supported", "source_ids": ["paper"], "evidence_ids": ["e"], "support_verified": True,
                        "question": "What does this experiment show?", "answer": "It studies stipulated pain; it does not demonstrate phenomenal experience.",
                        "generated_by": "claude", "prompt_version": "critic-v1", "review_status": "approved"})
    store.put("evidence", {"id": "invented", "source_id": "paper", "passage": "Unreported proof of consciousness", "support_verified": True})
    store.put("notes", {"id": "hallucinated", "source_ids": ["paper"], "evidence_ids": ["invented"], "support_verified": True,
                        "question": "Is it conscious?", "answer": "Yes."})
    snapshot = build_snapshot(store)
    assert len(snapshot["synthetic_sft"]) == 1
    assert snapshot["synthetic_sft"][0]["split"] == "validation"
    assert snapshot["synthetic_sft"][0]["synthetic"] is True
    rejected = {item["note_id"]: item["reasons"] for item in snapshot["manifest"]["excluded_notes"]}
    assert "raw" in rejected
    assert "evidence_passage_not_in_source" in rejected["hallucinated"]


def test_synthetic_drafts_need_policy_and_owner_approval(store):
    store.put("sources", source("paper"))
    store.put("evidence", {"id": "e", "source_id": "paper", "passage": "stipulated pain rather than demonstrated phenomenal experience", "support_verified": True})
    note = store.put("notes", {"id": "draft", "source_ids": ["paper"], "evidence_ids": ["e"], "support_verified": True,
                               "question": "What is tested?", "answer": "Stipulated pain.", "review_status": "pending"})
    first = build_snapshot(store)
    assert first["synthetic_sft"] == []
    reasons = first["manifest"]["excluded_notes"][0]["reasons"]
    assert "synthetic_output_training_policy_not_approved" in reasons
    assert "synthetic_example_needs_explicit_owner_review" in reasons
    store.save_settings({"synthetic_training_approved": True, "provider_policy_reference": "Verified contract policy"})
    assert build_snapshot(store)["synthetic_sft"] == []
    store.put("notes", {**note, "review_status": "approved"})
    assert len(build_snapshot(store)["synthetic_sft"]) == 1


def test_previous_training_family_cannot_become_new_holdout(store):
    # Choose a deterministic training-family bucket.
    import hashlib
    identifier = next(str(number) for number in range(100) if int(hashlib.sha256(f"family:{number}".encode()).hexdigest()[:8], 16) % 10 != 0)
    family = "family:" + identifier
    store.put("sources", source("original", family_id=family))
    first = build_snapshot(store)
    assert first["counts"]["train_documents"] == 1
    store.put("sources", source("new-version", family_id=family, split="heldout"))
    second = build_snapshot(store)
    assert second["counts"]["validation_documents"] == 0
    assert second["counts"]["original_documents"] == 0
    assert all("persistent_train_holdout_family_conflict" in item["reasons"] for item in second["manifest"]["excluded_sources"])


def test_near_duplicate_mirrors_share_a_holdout_group(store):
    words = [f"word{number}" for number in range(200)]
    store.put("sources", source("paper", family_id="primary-paper", text=" ".join(words)))
    store.put("sources", source("mirror", family_id="different-url", text=" ".join(words[:-5] + ["changed"] * 5), split="heldout"))
    snapshot = build_snapshot(store)
    assert snapshot["counts"]["train_documents"] == 0
    assert snapshot["counts"]["validation_documents"] == 1
    assert snapshot["original_text"][0]["source_ids"] == ["mirror", "paper"]
    assert len(snapshot["heldout_family_ids"]) == 1


def test_provenance_changes_do_not_pretend_to_be_new_corpus(store):
    initial = store.put("sources", source())
    first = build_snapshot(store)
    store.put("sources", {**initial, "provenance": {**initial["provenance"], "retrieved_at": "later"}})
    second = build_snapshot(store)
    assert first["id"] != second["id"]
    assert first["corpus_hash"] == second["corpus_hash"]


def test_cross_split_generated_example_is_quarantined(store):
    import hashlib
    train_family = next(f"train:{number}" for number in range(100) if int(hashlib.sha256(f"train:{number}".encode()).hexdigest()[:8], 16) % 10 != 0)
    text_a = "The first method compares recurrent processing with baseline classification. " * 10
    text_b = "The second study measures neural activity across conscious visual perception trials. " * 10
    store.put("sources", source("train", family_id=train_family, text=text_a))
    store.put("sources", source("heldout", family_id="heldout", text=text_b, split="heldout"))
    store.put("evidence", {"id": "mixed", "source_id": "train", "passage": "The first method compares recurrent processing", "support_verified": True})
    store.put("notes", {"id": "mix", "source_ids": ["train", "heldout"], "evidence_ids": ["mixed"], "support_verified": True,
                        "question": "Compare the studies", "answer": "They use different methods."})
    snapshot = build_snapshot(store)
    assert snapshot["synthetic_sft"] == []
    assert "instruction_spans_train_and_holdout_families" in snapshot["manifest"]["excluded_notes"][0]["reasons"]


def test_api_owner_boundary_notes_and_rights_review(store, monkeypatch):
    monkeypatch.setenv("OBSERVATORY_ADMIN_TOKEN", "owner-control-secret")
    store.put("sources", source("paper", license="unknown", license_verified=False))
    app = create_app(store, enable_runtime=False)
    with TestClient(app) as client:
        assert client.get("/api/health").json()["services"] == []
        assert client.post("/api/admin/missions/start", json={}).status_code == 401
        headers = {"Authorization": "Bearer owner-control-secret"}
        result = client.post("/api/admin/settings", headers=headers, json={"hf_token": "hf_example_secret_token"})
        assert result.status_code == 200
        assert "hf_example_secret_token" not in result.text
        assert client.post("/api/admin/missions/start", headers=headers, json={"objective": "Research consciousness"}).json()["status"] == "running"
        note = client.post("/api/admin/notes", headers=headers, json={"source_id": "paper", "text": "Investigate this source"}).json()
        assert client.patch(f"/api/admin/notes/{note['id']}", headers=headers, json={"bookmarked": True}).json()["bookmarked"]
        review = client.patch("/api/admin/sources/paper", headers=headers, json={"eligibility": "eligible"}).json()
        assert review["curation"]["eligible"] is False
        assert client.post("/api/admin/train", headers=headers, json={}).status_code == 503
        assert client.get("/api/state").status_code == 200


def test_frames_cannot_read_files_outside_frame_directory(store, tmp_path):
    outside = tmp_path / "secret.txt"
    outside.write_text("private")
    store.put("agents", {"id": "a", "frame_path": str(outside)})
    with TestClient(create_app(store, enable_runtime=False)) as client:
        assert client.get("/api/agents/a/frame").status_code == 404
        assert "secret.txt" not in client.get("/api/state").text


def test_self_hosted_fonts_load_without_exposing_other_files(store):
    import re
    with TestClient(create_app(store, enable_runtime=False)) as client:
        page = client.get("/observatory.html")
        assert 'class="grimoire-live grimoire-observatory"' in page.text
        assert 'href="observatory-fonts.css"' in page.text
        stylesheet = client.get("/observatory-fonts.css")
        assert stylesheet.status_code == 200
        fonts = re.findall(r'url\("(assets/fonts/[^\"]+)"\)', stylesheet.text)
        assert len(fonts) == 5
        for font in fonts:
            response = client.get("/" + font)
            assert response.status_code == 200
            assert response.headers["content-type"] == "font/woff2"
            assert response.content.startswith(b"wOF2")
        assert client.get("/assets/fonts/private.key").status_code == 404
        assert client.get("/assets/fonts/%2e%2e%2f%2e%2e%2fprivate.key").status_code == 404


def test_sidecar_navigation_redirects_to_public_site_without_serving_relay_files(store):
    with TestClient(create_app(store, enable_runtime=False)) as client:
        for path, destination in (("/index.html", "https://wirehead.agency/"),
                                  ("/live.html", "https://wirehead.agency/live.html")):
            response = client.get(path, follow_redirects=False)
            assert response.status_code == 307
            assert response.headers["location"] == destination
        assert client.get("/live/server.py").status_code == 404
        assert client.get("/observatory/.env").status_code == 404
