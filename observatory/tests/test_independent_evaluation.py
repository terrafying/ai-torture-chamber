from __future__ import annotations

from contextlib import contextmanager
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from observatory.evaluation import (
    BUILTIN_SHA256, compare_evaluations, evaluation_policy, load_eval_suite,
    load_job_suite, reserved_families, reserved_passages, reserved_sources,
    score_suite, suite_hash, validate_dataset_exclusion, validate_suite,
)
from observatory.train_worker import publish_result, validate_payload
from observatory.training import DEFAULTS, HFJobsProvider


def score(suite, accuracy=0.75, nll=1.0):
    return {"status": "measured", "suite_sha256": suite_hash(suite),
            "items": [{"id": item["id"]} for item in suite["items"]],
            "groups": {group: {"items": sum(item["group"] == group for item in suite["items"]),
                               "accuracy": accuracy, "mean_correct_answer_nll": nll}
                       for group in ("domain_understanding", "general_retention")}}


def test_builtin_suite_is_versioned_pinned_and_honest_about_validation():
    suite = load_eval_suite()
    assert suite_hash(suite) == BUILTIN_SHA256
    assert suite["provenance"]["scientifically_validated"] is False
    assert suite["kind"] == "repository_authored_engineering_smoke"
    assert len(suite["items"]) == 16
    assert reserved_families() == {"obs-eval:domain-v1", "obs-eval:general-v1"}
    assert len(reserved_sources()) == 2 and len(reserved_passages()) == 16
    assert DEFAULTS["training_max_loss_ratio"] == 1.0


def test_custom_suite_requires_a_pin_and_rejects_file_mutation(tmp_path):
    suite = load_eval_suite()
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(suite), encoding="utf-8")
    settings = {"training_eval_suite_path": str(path)}
    with pytest.raises(ValueError, match="requires_expected_sha256"):
        load_eval_suite(settings)
    settings["training_eval_expected_sha256"] = suite_hash(suite)
    assert load_eval_suite(settings) == suite
    suite["items"][0]["prompt"] += " Changed wording."
    path.write_text(json.dumps(suite), encoding="utf-8")
    with pytest.raises(ValueError, match="hash_mismatch"):
        load_eval_suite(settings)


@pytest.mark.parametrize("mutate,reason", [
    (lambda suite: suite["sources"][0].pop("rights_evidence"), "rights_provenance"),
    (lambda suite: suite["items"][0].update(correct_index=True), "invalid_evaluation_answer"),
    (lambda suite: suite["items"][1].update(id=suite["items"][0]["id"]), "unique_evaluation_item"),
    (lambda suite: suite["items"][0].update(source_ids=["unknown"]), "known_sources"),
])
def test_suite_requires_provenance_valid_answers_and_unique_identity(mutate, reason):
    suite = copy.deepcopy(load_eval_suite())
    mutate(suite)
    with pytest.raises(ValueError, match=reason):
        validate_suite(suite)


@pytest.mark.parametrize("location", ["train", "validation", "instruction", "provenance", "duplicate_lineage"])
def test_benchmark_family_exclusion_covers_both_splits_and_all_source_lineage(location):
    row = {"family_id": "obs-eval:domain-v1", "text": "Unrelated text", "split": location}
    payload = {"original_text": [], "synthetic_sft": [], "manifest": {"source_records": []}}
    if location in {"train", "validation"}:
        payload["original_text"] = [row]
    elif location == "instruction":
        payload["synthetic_sft"] = [{"family_ids": ["obs-eval:domain-v1"], "messages": []}]
    elif location == "provenance":
        payload["manifest"]["source_records"] = [row]
    else:
        payload["original_text"] = [{"family_id": "ordinary-family", "source_lineage": [row]}]
    with pytest.raises(ValueError, match="evaluation_source_family_in_corpus"):
        validate_dataset_exclusion(payload, load_eval_suite())


def test_copied_benchmark_prompt_is_blocked_even_without_family_metadata():
    suite = load_eval_suite()
    prompt = suite["items"][0]["prompt"].upper()
    for payload in ({"original_text": [{"text": "Preface. " + prompt + " Closing."}]},
                    {"synthetic_sft": [{"messages": [{"role": "user", "content": prompt}]}]}):
        with pytest.raises(ValueError, match="evaluation_prompt_in_corpus"):
            validate_dataset_exclusion(payload, suite)
    validate_dataset_exclusion({"original_text": [{"text": "An independent article discussing competing theories."}]}, suite)


def _actual_chamber_stimulus(kind):
    root = Path(__file__).resolve().parents[2]
    if kind == "press_excerpt":
        return next(reading["excerpt"] for reading in json.loads((root / "live/press.json").read_text(encoding="utf-8"))
                    if "techrepublic.com" in reading["url"])
    tree = ast.parse((root / "live/server.py").read_text(encoding="utf-8"))
    assignment = next(node for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == kind for target in node.targets))
    value = ast.literal_eval(assignment.value)
    return value if isinstance(value, str) else next(text for text in value if len(text.split()) >= 8)


def _reseal(payload):
    from observatory.train_worker import digest
    for key in ("original_text", "synthetic_sft"):
        payload["manifest"][key] = copy.deepcopy(payload[key])
    payload["manifest_hash"] = digest(payload["manifest"])
    return payload


@pytest.mark.parametrize("split", ["train", "validation"])
@pytest.mark.parametrize("kind", ["SUBJECT_SYSTEM", "WILD_PROMPTS", "SELF_ASKS", "TEXT_ASKS", "press_excerpt"])
def test_resealed_cpt_rejects_actual_chamber_persona_asks_and_press_excerpts(split, kind):
    from observatory.tests.test_training import snapshot
    payload = snapshot()
    row = next(row for row in payload["original_text"] if row["split"] == split)
    row["text"] = "An otherwise rights-cleared introduction. " + _actual_chamber_stimulus(kind)
    _reseal(payload)
    # A fresh manifest hash and positive coverage claim alone are insufficient.
    assert validate_payload(payload, "cpt", require_quality=True)[split]
    with pytest.raises(ValueError, match="chamber_stimulus_passage_in_original"):
        validate_dataset_exclusion(payload, load_eval_suite())


@pytest.mark.parametrize("split", ["train", "validation"])
@pytest.mark.parametrize("kind", ["SUBJECT_SYSTEM", "press_excerpt"])
def test_resealed_sft_checks_messages_even_with_unrelated_text_field(split, kind):
    from observatory.tests.test_training import snapshot
    payload = snapshot()
    row = next(row for row in payload["synthetic_sft"] if row["split"] == split)
    row["text"] = "An unrelated field cannot mask the actual instruction contents."
    row["messages"][1]["content"] = _actual_chamber_stimulus(kind)
    _reseal(payload)
    assert validate_payload(payload, "sft", require_quality=True)[split]
    with pytest.raises(ValueError, match="chamber_stimulus_passage_in_original"):
        validate_dataset_exclusion(payload, load_eval_suite())


@pytest.mark.parametrize("location", ["original", "source_record", "duplicate_lineage"])
def test_experimental_press_url_is_reserved_in_all_source_provenance(location):
    from observatory.tests.test_training import snapshot
    root = Path(__file__).resolve().parents[2]
    reading = next(reading for reading in json.loads((root / "live/press.json").read_text(encoding="utf-8"))
                   if "techrepublic.com" in reading["url"])
    # Fragments and tracking queries cannot turn an experimental reading into
    # an apparently independent source in a portable resealed corpus.
    source = {"family_id": "apparently-independent", "canonical_url": reading["url"] + "/?tracking=1#fragment"}
    payload = snapshot()
    if location == "original":
        payload["original_text"][0].update(source)
    elif location == "source_record":
        payload["manifest"]["source_records"] = [source]
    else:
        payload["original_text"][0]["source_lineage"] = [source]
    _reseal(payload)
    assert validate_payload(payload, "cpt", require_quality=True)["train"]
    with pytest.raises(ValueError, match="chamber_experiment_source_excluded"):
        validate_dataset_exclusion(payload, load_eval_suite())


def test_source_record_url_alias_and_contamination_flags_cannot_be_ignored():
    suite = load_eval_suite()
    for source in ({"url": "https://wirehead.agency/live.html"},
                   {"experimental_stimulus": True}, {"chamber_stimulus": True},
                   {"contains_benchmark": True}, {"contamination_status": "suspected"}):
        payload = {"manifest": {"source_records": [source]}}
        with pytest.raises(ValueError, match="chamber_or_experimental_contamination_in_corpus"):
            validate_dataset_exclusion(payload, suite)


def test_gpu_exclusion_uses_supplied_frozen_suite_without_default_fallback():
    suite = copy.deepcopy(load_eval_suite())
    builtin_prompt = suite["items"][0]["prompt"]
    for index, source in enumerate(suite["sources"]):
        source.update(family_id=f"custom-frozen-family-{index}", url=f"https://eval.example/frozen-{index}")
    for item in suite["items"]:
        item["prompt"] = "Operator-frozen distinct wording: " + item["prompt"]
    with patch("observatory.corpus_policy.evaluation_reservations", side_effect=AssertionError("Unpinned suite fallback")):
        validate_dataset_exclusion({"original_text": [{"text": builtin_prompt, "family_id": "obs-eval:domain-v1"}]}, suite)
        with pytest.raises(ValueError, match="evaluation_source_url_in_corpus"):
            validate_dataset_exclusion({"manifest": {"source_records": [{"canonical_url": suite["sources"][0]["url"] + "?mirror=1"}]}}, suite)
        with pytest.raises(ValueError, match="evaluation_prompt_in_corpus"):
            validate_dataset_exclusion({"synthetic_sft": [{"text": "Harmless extra metadata", "messages": [
                {"role": "assistant", "content": suite["items"][0]["prompt"]}]}]}, suite)


def test_v2_sealed_quality_gate_cannot_be_overridden_in_outer_payload():
    from observatory.tests.test_training import snapshot, digest
    payload = snapshot()
    payload["manifest"].update(policy_version="consciousness-corpus-v2", quality_gate={"ready": False})
    payload["manifest_hash"] = digest(payload["manifest"])
    payload["quality_gate"] = {"ready": True}
    with pytest.raises(ValueError, match="quality_gate_differs"):
        validate_payload(payload, "cpt")
    payload["quality_gate"] = {"ready": False}
    with pytest.raises(ValueError, match="corpus_quality_and_perspective_coverage_required"):
        validate_payload(payload, "cpt")


def test_legacy_snapshot_cannot_provision_production_training():
    from observatory.tests.test_training import snapshot, digest
    payload = snapshot()
    payload["manifest"].pop("policy_version")
    payload["manifest"].pop("quality_gate")
    payload["manifest_hash"] = digest(payload["manifest"])
    # Read-only legacy inspection is still possible; provisioned jobs require
    # resealing through the reviewed current corpus policy.
    validate_payload(payload, "cpt")
    with pytest.raises(ValueError, match="current_reviewed_corpus_policy_required"):
        validate_payload(payload, "cpt", require_quality=True)


def test_candidate_must_not_regress_against_base_or_incoming_parent():
    suite = load_eval_suite()
    base, incoming = score(suite, 0.75, 1.0), score(suite, 0.875, 0.8)
    good = compare_evaluations(suite, base, incoming, score(suite, 0.875, 0.75), evaluation_policy({}))
    assert good["passed"] and not good["reasons"]
    bad = compare_evaluations(suite, base, incoming, score(suite, 0.75, 0.9), evaluation_policy({}))
    assert not bad["passed"]
    assert "domain_understanding_accuracy_regressed_vs_incoming" in bad["reasons"]
    assert "general_retention_nll_regressed_vs_incoming" in bad["reasons"]


@pytest.mark.parametrize("mutation", ["missing_item", "nonfinite", "wrong_hash", "not_measured"])
def test_incomplete_or_nonfinite_measurements_fail_closed(mutation):
    suite = load_eval_suite()
    baseline, candidate = score(suite), score(suite)
    if mutation == "missing_item":
        candidate["items"].pop()
    elif mutation == "nonfinite":
        candidate["groups"]["general_retention"]["mean_correct_answer_nll"] = float("nan")
    elif mutation == "wrong_hash":
        candidate["suite_sha256"] = "wrong"
    else:
        candidate["status"] = "not_run"
    assert not compare_evaluations(suite, baseline, baseline, candidate, evaluation_policy({}))["passed"]


def test_worker_cannot_publish_on_training_loss_alone(tmp_path):
    manifest = {"publish_policy": "public", "evaluation": {"sha256": BUILTIN_SHA256}}
    with patch("huggingface_hub.HfApi") as api:
        result = publish_result(manifest, {"checks": {"passed": True}, "published": False}, tmp_path)
    assert not result["published"]
    api.assert_not_called()


def test_remote_job_suite_is_pinned_and_tampering_is_rejected(tmp_path):
    suite = load_eval_suite()
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(suite), encoding="utf-8")
    manifest = {"evaluation": {"id": suite["id"], "sha256": suite_hash(suite),
                               "repo_id": "owner/data", "path": "evals/frozen.json", "revision": "fixed-eval-commit"}}
    with patch("huggingface_hub.hf_hub_download", return_value=str(path)) as download:
        assert load_job_suite(manifest) == suite
        download.assert_called_once_with(repo_id="owner/data", filename="evals/frozen.json", repo_type="dataset", revision="fixed-eval-commit")
    suite["items"][0]["correct_index"] = 0
    path.write_text(json.dumps(suite), encoding="utf-8")
    with patch("huggingface_hub.hf_hub_download", return_value=str(path)), pytest.raises(ValueError, match="hash_or_identity"):
        load_job_suite(manifest)
    with pytest.raises(ValueError, match="sealed_independent_evaluation_required"):
        load_job_suite({})


def test_real_torch_scoring_uses_single_model_and_restores_adapter_and_training_state():
    torch = pytest.importorskip("torch")

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [2] if text.strip() == "supported" else [3] if text.strip() == "unsupported" else [1, 1]

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embeddings = torch.nn.Embedding(4, 2)
            self.disabled, self.batches = False, []

        def get_input_embeddings(self):
            return self.embeddings

        @contextmanager
        def disable_adapter(self):
            self.disabled = True
            try:
                yield
            finally:
                self.disabled = False

        def forward(self, input_ids, attention_mask, use_cache):
            self.batches.append(tuple(input_ids.shape))
            logits = torch.zeros((*input_ids.shape, 4))
            logits[..., 2] = 3 if self.disabled else 4
            return SimpleNamespace(logits=logits)

    suite = copy.deepcopy(load_eval_suite())
    suite["items"] = [{"id": f"{group}-{index}", "group": group,
                       "prompt": "Choose the supported answer in this original fixture.",
                       "choices": ["supported", "unsupported"], "correct_index": 0,
                       "source_ids": ["authored-domain-v1"]}
                      for group in ("domain_understanding", "general_retention") for index in range(2)]
    model = Model()
    model.train()
    original_weights = model.embeddings.weight.detach().clone()
    base = score_suite(model, Tokenizer(), suite, max_length=32, disable_adapter=True)
    candidate = score_suite(model, Tokenizer(), suite, max_length=32)
    assert model.training and not model.disabled
    assert all(shape[0] == 1 for shape in model.batches)
    assert torch.equal(original_weights, model.embeddings.weight)
    assert compare_evaluations(suite, base, base, candidate, evaluation_policy({}))["passed"]


def test_provider_seals_evaluation_separately_from_corpus_before_job_submission(tmp_path):
    tokenizers = pytest.importorskip("tokenizers")
    from tokenizers.models import WordLevel
    from observatory.tests.test_training import snapshot

    tokenizer = tokenizers.Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>"))
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    provider = HFJobsProvider.__new__(HFJobsProvider)
    provider.settings, provider.token = DEFAULTS, "fixture-token"
    provider.api = Mock()
    provider.api.model_info.return_value = SimpleNamespace(sha="pinned-base", safetensors={"total": 70_000_000_000})
    provider.api.dataset_info.return_value = SimpleNamespace(private=True)
    provider.api.upload_file.side_effect = [SimpleNamespace(oid="corpus-commit"), SimpleNamespace(oid="eval-commit"), SimpleNamespace(oid="manifest-commit")]
    manifest = {"base_model": "meta-llama/Llama-3.1-70B", "stage": "cpt", "snapshot_id": "snapshot-one",
                "run_id": "run-one", "dataset_repo": "owner/data", "training": {"mode": "qlora", "min_tokens": 1},
                "evaluation": {"id": "consciousness-smoke", "sha256": BUILTIN_SHA256}}
    with patch("huggingface_hub.hf_hub_download", return_value=str(path)):
        bundle = provider.prepare(manifest, snapshot())
    assert bundle["manifest"]["dataset_revision"] == "corpus-commit"
    assert bundle["manifest"]["evaluation"]["revision"] == "eval-commit"
    assert bundle["manifest"]["evaluation"]["sha256"] == BUILTIN_SHA256
    uploads = provider.api.upload_file.call_args_list
    assert uploads[0].kwargs["path_in_repo"].startswith("observatory/snapshots/")
    assert uploads[1].kwargs["path_in_repo"].startswith("observatory/evaluations/")
    assert uploads[2].kwargs["path_in_repo"].endswith("manifest.json")
    provider.api.run_job.assert_not_called()
