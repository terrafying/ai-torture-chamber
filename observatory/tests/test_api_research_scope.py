"""Broader source classifications preserve original text and machine evidence gates."""
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
import pytest

from observatory.app import create_app
from observatory.research_scope import DEFAULT_OBJECTIVE
from observatory.store import Store

AUTH = {"Authorization": "Bearer scope-test-owner"}


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSERVATORY_SECRET_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("OBSERVATORY_ADMIN_TOKEN", "scope-test-owner")
    store = Store(tmp_path / "scope.sqlite3")
    store.put("sources", {"id": "original", "canonical_url": "https://papers.example/mind",
        "text": "A philosophical argument considers the relation between subjective experience, selfhood and reality. " * 8,
        "source_type": "article", "provenance": {"method": "authored_fixture"},
        "license": "CC-BY-4.0", "license_verified": True, "rights_evidence": "Fixture rights recorded",
        "extraction": {"quality": "passed"}, "review_status": "pending"})
    yield store
    store.close()


def broad_review():
    return {"review_status": "approved", "quality_review": {"status": "approved",
        "reviewed_by": "operator", "rationale": "An attributed philosophical argument; no machine-sentience claim",
        "topic_relevance": "relevant", "evidence_stance": "not_applicable", "covered_stances": [],
        "source_type": "article", "topic_domains": ["philosophy_of_mind", "metaphysics_reality"],
        "evidence_kind": "philosophical_argument"}}


def test_owner_can_accept_philosophy_without_machine_stance_or_provenance_rewrite(db):
    original = db.get("sources", "original")
    with TestClient(create_app(db, enable_runtime=False)) as client:
        assert client.post("/api/admin/sources/original/review", json=broad_review()).status_code == 401
        response = client.post("/api/admin/sources/original/review", headers=AUTH, json=broad_review())
    assert response.status_code == 200 and response.json()["curation"]["eligible"]
    saved = db.get("sources", "original")
    assert saved["quality_review"]["evidence_stance"] == "not_applicable"
    assert saved["quality_review"]["topic_domains"] == ["metaphysics_reality", "philosophy_of_mind"]
    assert saved["quality_review"]["evidence_kind"] == "philosophical_argument"
    assert saved["text"] == original["text"] and saved["provenance"] == original["provenance"]


@pytest.mark.parametrize("change", [
    {"topic_domains": []}, {"topic_domains": ["anything"]}, {"topic_domains": "religion_contemplation"},
    {"evidence_kind": "proven-conscious"}, {"evidence_kind": None},
    {"covered_stances": ["supportive"]}, {"evidence_stance": "supportive"},
])
def test_invalid_or_machine_misclassified_broad_reviews_do_not_approve(db, change):
    payload = broad_review()
    payload["quality_review"].update(change)
    with TestClient(create_app(db, enable_runtime=False)) as client:
        response = client.post("/api/admin/sources/original/review", headers=AUTH, json=payload)
    assert response.status_code == 422
    assert "quality_review" not in db.get("sources", "original")


@pytest.mark.parametrize("missing", ["topic_domains", "evidence_kind"])
def test_broad_classification_requires_both_fields(db, missing):
    payload = broad_review()
    payload["quality_review"].pop(missing)
    with TestClient(create_app(db, enable_runtime=False)) as client:
        assert client.post("/api/admin/sources/original/review", headers=AUTH, json=payload).status_code == 422


def test_default_mission_is_broad_but_owner_scope_is_preserved(db):
    with TestClient(create_app(db, enable_runtime=False)) as client:
        result = client.post("/api/admin/missions/start", headers=AUTH, json={})
        assert result.status_code == 200 and result.json()["objective"] == DEFAULT_OBJECTIVE
        scoped = "Investigate human sleep and dream phenomenology only."
        result = client.post("/api/admin/missions/start", headers=AUTH, json={"objective": scoped})
        assert result.json()["objective"] == scoped
        result = client.post("/api/admin/missions/resume", headers=AUTH, json={})
        assert result.json()["objective"] == scoped
