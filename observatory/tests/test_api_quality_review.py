"""Owner quality decisions must reach the corpus without rewriting provenance."""
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
import pytest

from observatory.app import create_app
from observatory.store import Store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSERVATORY_SECRET_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("OBSERVATORY_ADMIN_TOKEN", "quality-test-owner")
    db = Store(tmp_path / "review.sqlite3")
    db.put("sources", {"id": "source-1", "canonical_url": "https://papers.example/reviewed",
                       "text": "A discussion of evidence and measurement of consciousness. " * 20,
                       "provenance": {"method": "browser_dom"}, "source_type": "article",
                       "license": "cc-by-4.0", "license_verified": True,
                       "rights_evidence": "https://creativecommons.org/licenses/by/4.0/",
                       "extraction": {"quality": "passed"}, "review_status": "pending"})
    yield db
    db.close()


def review():
    return {"review_status": "approved", "quality_review": {
        "status": "approved", "reviewed_by": "operator", "rationale": "Checked argument and document fidelity",
        "topic_relevance": "relevant", "evidence_stance": "uncertain", "source_type": "review_paper"}}


def post(client, payload):
    return client.post("/api/admin/sources/source-1/review", json=payload,
                       headers={"Authorization": "Bearer quality-test-owner"})


def test_owner_approves_quality_but_cannot_rewrite_collection(store):
    with TestClient(create_app(store, enable_runtime=False)) as client:
        page = client.get("/observatory.html")
        assert 'data-view="funding"' not in page.text
        assert "observatory-motion.js" in page.text
        assert client.get("/observatory-motion.js").status_code == 200
        assert client.post("/api/admin/sources/source-1/review", json=review()).status_code == 401
        result = post(client, {**review(), "source_type": "social", "provenance": {"method": "invented"}})
        assert result.status_code == 200
        assert result.json()["curation"]["eligible"] is True
    saved = store.get("sources", "source-1")
    assert saved["source_type"] == "article"
    assert saved["provenance"]["method"] == "browser_dom"
    assert saved["quality_review"]["source_type"] == "review_paper"


@pytest.mark.parametrize("field,value", [("contains_benchmark", "false"), ("chamber_stimulus", 1),
    ("contamination_status", {}), ("extraction_review_status", []), ("extraction_review_evidence", {})])
def test_invalid_review_values_rejected_without_server_error(store, field, value):
    with TestClient(create_app(store, enable_runtime=False)) as client:
        assert post(client, {**review(), field: value}).status_code == 422
    assert store.get("sources", "source-1").get("quality_review") is None


@pytest.mark.parametrize("field,value", [("evidence_stance", []), ("rationale", ""),
    ("reviewed_by", ""), ("topic_relevance", "certain"), ("covered_stances", ["truth"])])
def test_invalid_quality_classifications_rejected(store, field, value):
    payload = review()
    payload["quality_review"][field] = value
    with TestClient(create_app(store, enable_runtime=False)) as client:
        assert post(client, payload).status_code == 422


def test_unverified_pdf_requires_recorded_fidelity_review(store):
    source = store.get("sources", "source-1")
    source["extraction"] = {"method": "docling_pdf", "quality": "unverified"}
    store.put("sources", source)
    with TestClient(create_app(store, enable_runtime=False)) as client:
        assert post(client, review()).json()["curation"]["eligible"] is False
        result = post(client, {**review(), "extraction_review_status": "approved",
                               "extraction_review_evidence": "Compared reading order, formulas and tables against source PDF"})
        assert result.json()["curation"]["eligible"] is True
        result = post(client, {**review(), "chamber_stimulus": True})
        assert result.json()["curation"]["eligible"] is False
