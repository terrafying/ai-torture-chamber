"""Actual automatic acceptance -> sealed snapshot -> GPU input validation, offline."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from observatory.automatic_curation import AutomaticCurationWorker
from observatory.curation import build_snapshot, canonical_hash
from observatory.store import Store
from observatory.train_worker import validate_payload


def validate_packaged_worker(snapshot, changed, directory):
    """Run the actual Docker-copied validation tree with sidecar imports denied."""
    root = Path(__file__).resolve().parents[2]
    copied = []
    for line in (root / "observatory/Dockerfile.training").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts and parts[0] == "COPY" and parts[-1] in {"/app/observatory/", "/app/live/", "/app/observatory/evals"}:
            copied.extend(parts[1:-1])
    assert "observatory/curation_receipts.py" in copied
    target = directory / "worker-image"
    (target / "observatory").mkdir(parents=True)
    for source in copied:
        destination = target / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (root / source).is_dir():
            shutil.copytree(root / source, destination)
        else:
            shutil.copyfile(root / source, destination)
    (target / "valid.json").write_text(json.dumps(snapshot), encoding="utf-8")
    (target / "changed.json").write_text(json.dumps(changed), encoding="utf-8")
    script = """
import importlib.abc, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
class DenySidecar(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'browser_use','httpx','cryptography','torch','transformers','pydantic'} or fullname in {'observatory.store','observatory.research_llm','observatory.automatic_curation'}:
            raise AssertionError('GPU validation imports sidecar dependency: ' + fullname)
sys.meta_path.insert(0, DenySidecar())
from observatory.train_worker import validate_payload
from observatory.evaluation import load_eval_suite, validate_dataset_exclusion
valid = json.loads(Path(sys.argv[1], 'valid.json').read_text(encoding='utf-8'))
assert validate_payload(valid, 'cpt', require_quality=True)['train']
validate_dataset_exclusion(valid, load_eval_suite())
changed = json.loads(Path(sys.argv[1], 'changed.json').read_text(encoding='utf-8'))
try:
    validate_payload(changed, 'cpt', require_quality=True)
except ValueError as error:
    assert 'source_bound_review_receipt' in str(error)
else:
    raise AssertionError('Changed automated review accepted')
"""
    result = subprocess.run([sys.executable, "-I", "-c", script, str(target)],
                            capture_output=True, text=True, timeout=30, cwd=target)
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_accepted_original_enters_cpt_and_worker_rechecks_receipt(tmp_path):
    store = Store(tmp_path / "corpus.db")
    text = "This methodological analysis compares supportive, skeptical and uncertain perspectives on machine consciousness. " * 5
    source = {"id": "training-original", "canonical_url": "https://papers.example/methods", "text": text,
        "license": "cc-by-4.0", "license_verified": True, "rights_evidence": "Fixture recorded rights",
        "provenance": {"method": "authored_test_fixture"}, "extraction": {"quality": "passed"}, "review_status": "pending"}
    verdict = {"decision": "accept", "topic_relevance": "relevant", "evidence_stance": "methodological",
        "source_type": "theoretical_paper", "covered_stances": ["supportive", "skeptical", "uncertain"],
        "topic_domains": ["machine_consciousness"], "evidence_kind": "scientific_theory",
        "quotes": [text[:110]], "rationale": "Authored fixture covers the three relevant perspectives."}
    class Model:
        calls = 0
        async def ainvoke(self, messages, output_format=None):
            self.calls += 1
            return SimpleNamespace(completion=output_format.model_validate(verdict))
    try:
        store.save_settings({"research_provider": "openai", "research_model": "fixture-model",
            "auto_curation_enabled": True, "auto_curation_policy_ack": "originals-v2"})
        store.set_mission({"id": "fixture-mission", "status": "running"})
        # Explicit train assignment keeps this boundary fixture independent of hash buckets.
        from observatory.curation import family_id
        store.put("family_splits", {"id": canonical_hash(family_id(source)), "split": "train"})
        store.put("sources", source)
        model = Model()
        result = await AutomaticCurationWorker(store).review_source(source["id"], model=model)
        assert result["status"] == "accepted" and model.calls == 2
        store.put("sources", {"id": "validation-original", "canonical_url": "https://papers.example/measurement",
            "text": "A separate validation source describes experimental controls, calibrated instruments and uncertainty in measuring biological neural activity. " * 5,
            "license": "cc-by-4.0", "license_verified": True, "rights_evidence": "Fixture recorded rights",
            "provenance": {"method": "authored_test_fixture"}, "extraction": {"quality": "passed"}, "held_out": True,
            "quality_review": {"status": "approved", "reviewed_by": "fixture-owner", "rationale": "Fixture validation source reviewed",
                "topic_relevance": "relevant", "evidence_stance": "uncertain", "source_type": "empirical_paper"}})
        snapshot = build_snapshot(store)
        splits = validate_payload(snapshot, "cpt", require_quality=True)
        row = next(row for row in splits["train"] if row["representative_source_id"] == source["id"])
        assert row["text"] == text
        assert row["quality_review"]["reviewer_kind"] == "automated"
        assert snapshot["synthetic_sft"] == []
        # Even a newly sealed manifest cannot authorize an unrelated approval receipt.
        changed = deepcopy(snapshot)
        changed_row = next(item for item in changed["original_text"] if item["representative_source_id"] == source["id"])
        changed_row["quality_review"]["model"] = "unrelated-model"
        changed["manifest"]["original_text"] = changed["original_text"]
        changed["manifest_hash"] = canonical_hash(changed["manifest"])
        with pytest.raises(ValueError, match="source_bound_review_receipt"):
            validate_payload(changed, "cpt", require_quality=True)
        validate_packaged_worker(snapshot, changed, tmp_path)
    finally:
        store.close()


def test_application_starts_disabled_curation_without_provider_requests(tmp_path, monkeypatch):
    import httpx
    from fastapi.testclient import TestClient
    from observatory.app import create_app
    async def forbidden(*args, **kwargs):
        raise AssertionError("Disabled runtime must not contact a model or payment provider")
    monkeypatch.setattr(httpx.AsyncClient, "request", forbidden)
    store = Store(tmp_path / "application.db")
    try:
        with TestClient(create_app(store)) as client:
            health = client.get("/api/health").json()
            assert health["workers"]["automatic_curation"]["running"] is True
            worker = client.app.state.services["automatic_curation"]
            assert client.get("/api/state").json()["curation_reviews"] == []
        assert worker.closed and worker.task.done()
    finally:
        store.close()
