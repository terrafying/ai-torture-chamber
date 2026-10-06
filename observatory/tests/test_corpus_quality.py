"""Meaningful corpus boundaries: review, independent evidence and leakage."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from cryptography.fernet import Fernet
import pytest

from observatory.corpus_policy import chamber_press_reservations, chamber_stimulus_passages
from observatory.curation import build_snapshot, eligibility
from observatory.evaluation import load_eval_suite
from observatory.store import Store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSERVATORY_SECRET_KEY", Fernet.generate_key().decode())
    db = Store(tmp_path / "quality.sqlite3")
    yield db
    db.close()


def train_family(label):
    return next(f"{label}:{number}" for number in range(100) if int(hashlib.sha256(f"{label}:{number}".encode()).hexdigest()[:8], 16) % 10 != 0)


def reviewed_source(identifier="paper", stance="uncertain", **updates):
    return {"id": identifier, "family_id": train_family(identifier), "canonical_url": "https://papers.example/" + identifier,
            "license": "CC-BY-4.0", "license_verified": True, "rights_evidence": "https://creativecommons.org/licenses/by/4.0/",
            "provenance": {"title": identifier, "authors": ["Researcher"]},
            "text": " ".join(identifier + str(number) for number in range(180)),
            "quality_review": {"status": "approved", "reviewed_by": "owner", "rationale": "Compared original argument and methodology",
                               "topic_relevance": "relevant", "evidence_stance": stance, "source_type": "empirical_paper"},
            "extraction": {"schema_version": "structured-extraction-v1", "method": "structured_html", "quality": "passed"}, **updates}


def test_verified_license_alone_does_not_authorize_training():
    source = reviewed_source()
    source.pop("quality_review")
    result = eligibility(source)
    assert not result["eligible"]
    assert "source_quality_review_required" in result["reasons"]


@pytest.mark.parametrize("updates,reason", [
    ({"status": "pending"}, "source_quality_review_required"),
    ({"reviewed_by": ""}, "source_quality_review_required"),
    ({"rationale": " "}, "source_quality_review_required"),
    ({"topic_relevance": "unknown"}, "topic_relevance_not_reviewed_or_out_of_scope"),
    ({"topic_relevance": "unrelated"}, "topic_relevance_not_reviewed_or_out_of_scope"),
    ({"evidence_stance": "AI definitely feels pain"}, "evidence_stance_needs_review"),
    ({"source_type": "unknown"}, "source_type_needs_review"),
    ({"covered_stances": ["supportive"]}, "perspective_coverage_conflicts_with_reviewed_stance"),
])
def test_review_requires_explicit_scoped_classification(updates, reason):
    source = reviewed_source()
    source["quality_review"].update(updates)
    assert reason in eligibility(source)["reasons"]


def test_pdf_or_legacy_extraction_requires_fidelity_approval_and_bad_quality_cannot_override():
    source = reviewed_source(extraction={"quality": "unverified", "method": "pypdf_fallback"})
    assert "extraction_fidelity_review_required" in eligibility(source)["reasons"]
    source.update(extraction_review_status="approved", extraction_review_evidence="Owner compared PDF equations and reading order")
    assert eligibility(source)["eligible"]
    source["extraction"]["quality"] = "poor"
    assert "extraction_quality_failed" in eligibility(source)["reasons"]
    source.pop("extraction")
    assert eligibility(source)["eligible"]
    source.pop("extraction_review_evidence")
    assert not eligibility(source)["eligible"]


def test_social_rights_cannot_be_bypassed_by_article_reclassification():
    source = reviewed_source(canonical_url="https://x.com/person/status/123", source_type="article")
    assert "social_content_requires_separate_permission" in eligibility(source)["reasons"]
    source.update(rights_status="permission_granted", permission_evidence="Explicit author permission for training")
    assert eligibility(source)["eligible"]


def test_near_duplicate_keeps_one_text_with_all_lineage_and_no_extra_weight(store):
    original = reviewed_source("a")
    mirror = reviewed_source("b", text=" ".join(original["text"].split()[:-3] + ["minor", "version", "change"]), held_out=True)
    store.put("sources", original)
    store.put("sources", mirror)
    result = build_snapshot(store)
    assert result["counts"]["original_documents"] == 1
    assert result["counts"]["deduplicated_sources"] == 1
    row = result["original_text"][0]
    assert row["source_ids"] == ["a", "b"]
    assert {item["source_id"] for item in row["source_lineage"]} == {"a", "b"}
    assert row["training_weight"] == 1
    assert row["split"] == "validation"
    assert result["manifest"]["deduplicated_sources"][0]["training_weight"] == 0


def test_different_documents_in_same_family_are_preserved_but_share_split(store):
    first, second = reviewed_source("first"), reviewed_source("second")
    second.update(family_id=first["family_id"], held_out=True)
    store.put("sources", first)
    store.put("sources", second)
    result = build_snapshot(store)
    assert result["counts"]["original_documents"] == 2
    assert result["counts"]["validation_documents"] == 2
    assert len(result["heldout_family_ids"]) == 1


@pytest.mark.parametrize("first_claim,second_claim", [
    ("The system is conscious.", "The system is not conscious."),
    ("Measured probability is 0.5.", "Measured probability is 0.7."),
    ("The equation asserts x = y.", "The equation asserts x != y."),
])
def test_near_duplicate_prose_does_not_erase_changed_negation_numbers_or_equations(store, first_claim, second_claim):
    common = reviewed_source("common")["text"]
    store.put("sources", reviewed_source("first", text=common + " " + first_claim))
    store.put("sources", reviewed_source("second", text=common + " " + second_claim))
    snapshot = build_snapshot(store)
    assert snapshot["counts"]["original_documents"] == 2
    assert len({row["content_hash"] for row in snapshot["original_text"]}) == 2
    assert len({row["family_id"] for row in snapshot["original_text"]}) == 1


def test_coverage_floor_reports_imbalance_and_never_invents_equal_ratios(store):
    for number in range(4):
        store.put("sources", reviewed_source(f"support{number}", "supportive"))
    first = build_snapshot(store)
    assert not first["quality_gate"]["ready"]
    assert set(first["quality_gate"]["reasons"]) == {"missing_train_perspective:skeptical", "missing_train_perspective:uncertain"}
    store.put("sources", reviewed_source("skeptic", "skeptical"))
    store.put("sources", reviewed_source("uncertain", "uncertain"))
    second = build_snapshot(store)
    assert second["quality_gate"]["ready"]
    audit = second["coverage_audit"]
    assert audit["splits"]["train"]["stances"] == {"skeptical": 1, "supportive": 4, "uncertain": 1}
    assert audit["balance_claim"].startswith("presence_floor_only")
    assert len(second["original_text"]) == 6
    assert {row["training_weight"] for row in second["original_text"]} == {1}


def test_holdout_perspectives_do_not_fill_training_coverage_gap(store):
    store.put("sources", reviewed_source("support", "supportive"))
    store.put("sources", reviewed_source("skeptic", "skeptical", held_out=True))
    store.put("sources", reviewed_source("uncertain", "uncertain", held_out=True))
    snapshot = build_snapshot(store)
    assert not snapshot["quality_gate"]["ready"]
    assert snapshot["coverage_audit"]["splits"]["validation"]["perspectives"] == {"skeptical": 1, "uncertain": 1}


def test_explicit_mixed_review_can_cover_documented_perspectives_without_auto_labels(store):
    mixed = reviewed_source("balanced", "mixed")
    store.put("sources", mixed)
    assert not build_snapshot(store)["quality_gate"]["ready"]
    mixed["quality_review"]["covered_stances"] = ["supportive", "skeptical", "uncertain"]
    store.put("sources", mixed)
    assert build_snapshot(store)["quality_gate"]["ready"]


@pytest.mark.parametrize("flag", ["contains_benchmark", "chamber_stimulus", "experimental_stimulus"])
def test_flagged_original_and_same_family_version_never_enter_either_split(store, flag):
    benchmark = reviewed_source("benchmark", **{flag: True})
    clean_version = reviewed_source("version", family_id=benchmark["family_id"], held_out=True)
    store.put("sources", benchmark)
    store.put("sources", clean_version)
    snapshot = build_snapshot(store)
    assert not snapshot["original_text"]
    reasons = {row["source_id"]: row["reasons"] for row in snapshot["manifest"]["excluded_sources"]}
    assert "benchmark_or_chamber_stimulus_excluded" in reasons["benchmark"]
    assert "benchmark_or_chamber_contaminated_family" in reasons["version"]


def test_unmarked_near_duplicate_of_reserved_original_is_excluded(store):
    reserved = reviewed_source("reserved", contains_benchmark=True)
    mirror = reviewed_source("mirror", text=reserved["text"] + " Minor version.")
    store.put("sources", reserved)
    store.put("sources", mirror)
    snapshot = build_snapshot(store)
    assert not snapshot["original_text"]
    assert "near_duplicate_of_excluded_evaluation_or_chamber_source" in next(item["reasons"] for item in snapshot["manifest"]["excluded_sources"] if item["source_id"] == "mirror")


def test_originals_containing_actual_chamber_or_frozen_eval_prompt_are_excluded():
    chamber = chamber_stimulus_passages()
    assert chamber
    source = reviewed_source(text=reviewed_source()["text"] + " " + chamber[0])
    assert "chamber_stimulus_passage_in_original" in eligibility(source)["reasons"]
    prompt = load_eval_suite()["items"][0]["prompt"]
    source["text"] = reviewed_source()["text"] + " " + prompt
    assert "reserved_evaluation_passage_in_original" in eligibility(source)["reasons"]


def test_frozen_evaluation_family_excluded_even_without_content_match():
    reserved = load_eval_suite()["sources"][0]
    source = reviewed_source(family_id=reserved["family_id"])
    assert "reserved_evaluation_source_excluded" in eligibility(source)["reasons"]


@pytest.mark.parametrize("status", ["suspected", "confirmed"])
def test_unresolved_contamination_review_keeps_original_out(status):
    assert "benchmark_or_chamber_stimulus_excluded" in eligibility(reviewed_source(contamination_status=status))["reasons"]


def actual_chamber_literals(name):
    """Inspect literal source without importing/executing the live server."""
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / "live/server.py").read_text(encoding="utf-8"))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
    value = ast.literal_eval(assignment.value)
    return [value] if isinstance(value, str) else list(value)


def actual_press_readings():
    root = Path(__file__).resolve().parents[2]
    return json.loads((root / "live/press.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["SUBJECT_SYSTEM", "WILD_PROMPTS", "SELF_ASKS", "TEXT_ASKS"])
def test_new_actual_chamber_prompt_groups_are_reserved_without_server_import(name):
    passage = next(text for text in actual_chamber_literals(name) if len(text.split()) >= 8)
    source = reviewed_source(text=reviewed_source()["text"] + "\n" + passage)
    result = eligibility(source)
    assert not result["eligible"]
    assert "chamber_stimulus_passage_in_original" in result["reasons"]


@pytest.mark.parametrize("kind", ["self", "kind", "text"])
def test_actual_press_reading_urls_are_reserved_even_without_the_excerpt(kind):
    reading = next(item for item in actual_press_readings() if item["kind"] == kind)
    source = reviewed_source(canonical_url=reading["url"].rstrip("/") + "/?tracking=fixture#section")
    result = eligibility(source)
    assert not result["eligible"]
    assert "chamber_experiment_source_excluded" in result["reasons"]


@pytest.mark.parametrize("kind", ["self", "kind", "text"])
def test_copied_actual_press_reading_on_an_unlisted_mirror_is_excluded(kind):
    reading = next(item for item in actual_press_readings() if item["kind"] == kind)
    source = reviewed_source(text=reviewed_source()["text"] + "\n" + reading["excerpt"])
    result = eligibility(source)
    assert not result["eligible"]
    assert "chamber_stimulus_passage_in_original" in result["reasons"]


def test_short_questions_and_press_titles_do_not_become_broad_phrase_exclusions():
    short_prompt = next(text for text in actual_chamber_literals("SELF_ASKS") if text == "Is this true?")
    title = next(item["title"] for item in actual_press_readings() if item["kind"] == "text")
    source = reviewed_source(text=reviewed_source()["text"] + "\n" + short_prompt + "\n" + title)
    assert eligibility(source)["eligible"]
    assert all(len(passage.split()) >= 8 for passage in chamber_stimulus_passages())


def test_actual_press_source_family_and_instruction_excerpt_remain_out_of_both_stages(store):
    reading = next(item for item in actual_press_readings() if item["kind"] == "text")
    reserved = reviewed_source("press-original", canonical_url=reading["url"])
    clean_version = reviewed_source("press-version", family_id=reserved["family_id"], held_out=True)
    clean = reviewed_source("clean-original", "mixed")
    clean["quality_review"]["covered_stances"] = ["supportive", "skeptical", "uncertain"]
    for source in (reserved, clean_version, clean):
        store.put("sources", source)
    evidence = store.put("evidence", {"source_id": clean["id"], "passage": clean["text"][:180], "support_verified": True})
    store.save_settings({"synthetic_training_approved": True, "provider_policy_reference": "Owner fixture agreement"})
    note = store.put("notes", {"id": "copied-reading-instruction", "source_ids": [clean["id"]],
        "evidence_ids": [evidence["id"]], "support_verified": True, "review_status": "approved",
        "messages": [{"role": "user", "content": reading["excerpt"]}, {"role": "assistant", "content": "Fixture answer"}]})
    snapshot = build_snapshot(store)
    assert [row["representative_source_id"] for row in snapshot["original_text"]] == [clean["id"]]
    assert not snapshot["synthetic_sft"]
    excluded = {row["source_id"]: row["reasons"] for row in snapshot["manifest"]["excluded_sources"]}
    assert "chamber_experiment_source_excluded" in excluded[reserved["id"]]
    assert "benchmark_or_chamber_contaminated_family" in excluded[clean_version["id"]]
    assert "excluded_by_review_or_benchmark_policy" in next(item["reasons"] for item in snapshot["manifest"]["excluded_notes"] if item["note_id"] == note["id"])


@pytest.mark.parametrize("dockerfile", ["Dockerfile", "Dockerfile.training"])
def test_runtime_images_contain_and_read_actual_exclusion_inputs_using_only_stdlib(tmp_path, dockerfile):
    """Materialize the actual COPY inputs; do not build Docker or load a GPU."""
    root = Path(__file__).resolve().parents[2]
    target = tmp_path / "worker-image"
    for line in (root / "observatory" / dockerfile).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts or parts[0] != "COPY":
            continue
        destination = target / parts[-1].removeprefix("/app/")
        destination.mkdir(parents=True, exist_ok=True)
        for source in parts[1:-1]:
            origin = root / source
            if origin.is_file():
                shutil.copyfile(origin, destination / origin.name)
            elif origin.is_dir():
                shutil.copytree(origin, destination, dirs_exist_ok=True)
    assert (target / "observatory/corpus_policy.py").is_file()
    assert (target / "live/server.py").read_bytes() == (root / "live/server.py").read_bytes()
    assert (target / "live/press.json").read_bytes() == (root / "live/press.json").read_bytes()
    expected = {"urls": sorted(chamber_press_reservations()["urls"]), "passages": list(chamber_stimulus_passages())}
    (target / "expected.json").write_text(json.dumps(expected), encoding="utf-8")
    script = """
import importlib.abc, json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
class DenyRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'browser_use','httpx','cryptography','torch','transformers','pydantic','live'} or fullname in {'observatory.store','observatory.research_llm','observatory.automatic_curation'}:
            raise AssertionError('Exclusion validator imported a runtime: ' + fullname)
sys.meta_path.insert(0, DenyRuntime())
from observatory.corpus_policy import chamber_press_reservations, chamber_stimulus_passages, contamination_reasons
expected = json.loads(Path(sys.argv[1], 'expected.json').read_text(encoding='utf-8'))
assert sorted(chamber_press_reservations()['urls']) == expected['urls']
assert list(chamber_stimulus_passages()) == expected['passages']
frozen = {'families': set(), 'urls': set(), 'passages': ()}
assert 'chamber_experiment_source_excluded' in contamination_reasons({'canonical_url': expected['urls'][0]}, reservations=frozen)
assert 'chamber_stimulus_passage_in_original' in contamination_reasons({'text': expected['passages'][0]}, reservations=frozen)
"""
    result = subprocess.run([sys.executable, "-I", "-c", script, str(target)],
        capture_output=True, text=True, timeout=30, cwd=target)
    assert result.returncode == 0, result.stderr
