"""Frozen, source-separated engineering evaluations for adapter candidates.

The bundled tasks are repository-authored smoke checks, not a validated measure
of consciousness knowledge, general intelligence, or subjective experience.
No model SDK is imported until scoring is explicitly invoked inside the worker.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

GROUPS = {"domain_understanding", "general_retention"}
BUILTIN_PATH = Path(__file__).with_name("evals") / "consciousness-smoke-v1.json"
BUILTIN_SHA256 = "4c808df16f538dd07417e3de5164ba8929de0bd0d3aa1211b8c09042f678dece"
POLICY_DEFAULTS = {
    "max_accuracy_drop": 0.0, "max_nll_ratio": 1.0,
    "min_domain_accuracy": 0.5, "min_general_accuracy": 0.5,
    "max_length": 1024,
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def suite_hash(suite: dict) -> str:
    return hashlib.sha256(canonical_bytes(suite)).hexdigest()


def validate_suite(suite: dict) -> dict:
    if not isinstance(suite, dict) or suite.get("schema_version") != 1 or not suite.get("id") or not suite.get("version"):
        raise ValueError("versioned_evaluation_suite_required")
    if suite.get("scoring") != "mean_completion_log_likelihood_v1":
        raise ValueError("unsupported_evaluation_scoring")
    if not suite.get("limitations") or not suite.get("provenance"):
        raise ValueError("evaluation_provenance_and_limitations_required")
    sources, items = suite.get("sources"), suite.get("items")
    if not isinstance(sources, list) or not sources or not isinstance(items, list) or not 4 <= len(items) <= 256:
        raise ValueError("evaluation_sources_and_4_to_256_items_required")
    source_ids = set()
    for source in sources:
        if not isinstance(source, dict) or not source.get("id") or source["id"] in source_ids:
            raise ValueError("unique_evaluation_source_ids_required")
        if not source.get("family_id") or not source.get("url") or not source.get("provenance"):
            raise ValueError("evaluation_source_family_url_and_provenance_required")
        if source.get("license_verified") is not True or not source.get("rights_evidence"):
            raise ValueError("evaluation_rights_provenance_required")
        if str(source.get("license", "")).lower() not in {"cc0", "cc0-1.0", "public-domain", "cc-by", "cc-by-4.0", "cc-by-3.0", "permission-granted"}:
            raise ValueError("evaluation_license_not_allowed")
        source_ids.add(source["id"])
    identities, counts = set(), {group: 0 for group in GROUPS}
    for item in items:
        if not isinstance(item, dict) or not item.get("id") or item["id"] in identities:
            raise ValueError("unique_evaluation_item_ids_required")
        identities.add(item["id"])
        if item.get("group") not in GROUPS:
            raise ValueError("evaluation_requires_domain_and_general_groups")
        counts[item["group"]] += 1
        if not isinstance(item.get("prompt"), str) or not 20 <= len(item["prompt"].strip()) <= 16000:
            raise ValueError("invalid_evaluation_prompt")
        choices = item.get("choices")
        if not isinstance(choices, list) or not 2 <= len(choices) <= 8 or any(not isinstance(choice, str) or not choice.strip() for choice in choices):
            raise ValueError("invalid_evaluation_choices")
        if len(set(choices)) != len(choices) or type(item.get("correct_index")) is not int or not 0 <= item["correct_index"] < len(choices):
            raise ValueError("invalid_evaluation_answer")
        if not item.get("source_ids") or set(item["source_ids"]) - source_ids:
            raise ValueError("evaluation_item_requires_known_sources")
    if any(count < 2 for count in counts.values()):
        raise ValueError("at_least_two_items_per_evaluation_group_required")
    return suite


def load_eval_suite(settings: dict | None = None) -> dict:
    settings = settings or {}
    path = Path(settings.get("training_eval_suite_path") or BUILTIN_PATH)
    suite = validate_suite(json.loads(path.read_text(encoding="utf-8")))
    expected = settings.get("training_eval_expected_sha256") or (BUILTIN_SHA256 if not settings.get("training_eval_suite_path") else None)
    if not expected:
        raise ValueError("custom_evaluation_suite_requires_expected_sha256")
    if suite_hash(suite) != expected:
        raise ValueError("evaluation_suite_hash_mismatch")
    return suite


def reserved_families(settings: dict | None = None) -> set[str]:
    return {str(source["family_id"]).lower() for source in load_eval_suite(settings)["sources"]}


def reserved_sources(settings: dict | None = None) -> set[str]:
    return {str(source["url"]).rstrip("/").lower() for source in load_eval_suite(settings)["sources"]}


def reserved_passages(settings: dict | None = None) -> tuple[str, ...]:
    return tuple(item["prompt"] for item in load_eval_suite(settings)["items"])


def _normalized(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def _source_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", ""))


def validate_dataset_exclusion(payload: dict, suite: dict) -> None:
    """Reject benchmark and Chamber sources/passages in both training stages.

    Identity and exact prompt checks are conservative engineering guards. They
    cannot discover all paraphrases, translations, or pretraining contamination.
    Chamber reservations come from the packaged experiment code/readings; all
    evaluation reservations come from this job's supplied frozen suite.
    """
    from .corpus_policy import contamination_reasons

    validate_suite(suite)
    families = {str(source["family_id"]).lower() for source in suite["sources"]}
    urls = {_source_url(str(source["url"])) for source in suite["sources"]}
    prompts = tuple(_normalized(item["prompt"]) for item in suite["items"])
    reservations = {"families": families, "urls": urls, "passages": prompts}
    manifest = payload.get("manifest", {})
    originals = payload.get("original_text", []) + manifest.get("original_text", [])
    rows = originals + payload.get("synthetic_sft", []) + manifest.get("synthetic_sft", []) + manifest.get("source_records", [])
    rows += [source for row in originals for source in row.get("source_lineage", [])]
    for row in rows:
        row_families = set(row.get("family_ids", [])) | {row.get("family_id", "")}
        if {str(family).lower() for family in row_families} & families:
            raise ValueError("evaluation_source_family_in_corpus")
        if _source_url(str(row.get("canonical_url", row.get("url", "")))) in urls:
            raise ValueError("evaluation_source_url_in_corpus")
        # Check both representations: an unrelated extra text field must not
        # hide contaminated SFT messages in a portable, correctly resealed file.
        texts = [row.get("text", "")]
        if row.get("messages"):
            texts.append(" ".join(message.get("content", "") for message in row["messages"]))
        for text in texts:
            if any(prompt in _normalized(text) for prompt in prompts):
                raise ValueError("evaluation_prompt_in_corpus")
            reasons = contamination_reasons({**row, "text": text}, reservations=reservations)
            if reasons:
                raise ValueError("chamber_or_experimental_contamination_in_corpus: " + ", ".join(reasons))


def evaluation_policy(settings: dict) -> dict:
    policy = {key: settings.get("training_eval_" + key, value) for key, value in POLICY_DEFAULTS.items()}
    for key in ("max_accuracy_drop", "min_domain_accuracy", "min_general_accuracy"):
        value = float(policy[key])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("evaluation_accuracy_threshold_out_of_range")
        policy[key] = value
    ratio = float(policy["max_nll_ratio"])
    if not math.isfinite(ratio) or not 0 < ratio <= 2:
        raise ValueError("evaluation_nll_ratio_out_of_range")
    policy["max_nll_ratio"] = ratio
    length = int(policy["max_length"])
    if not 32 <= length <= 32768:
        raise ValueError("evaluation_sequence_length_out_of_range")
    policy["max_length"] = length
    return policy


def load_job_suite(manifest: dict, *, injected: bool = False) -> dict:
    spec = manifest.get("evaluation")
    if not spec:
        if injected:
            return load_eval_suite()
        raise ValueError("sealed_independent_evaluation_required")
    if injected and spec.get("suite"):
        suite = validate_suite(spec["suite"])
    else:
        from huggingface_hub import hf_hub_download
        required = ("repo_id", "path", "revision", "sha256")
        if any(not spec.get(key) for key in required):
            raise ValueError("sealed_independent_evaluation_required")
        path = hf_hub_download(repo_id=spec["repo_id"], filename=spec["path"], repo_type="dataset", revision=spec["revision"])
        suite = validate_suite(json.loads(Path(path).read_text(encoding="utf-8")))
    if suite_hash(suite) != spec.get("sha256") or suite["id"] != spec.get("id"):
        raise ValueError("evaluation_suite_hash_or_identity_mismatch")
    return suite


def score_suite(model: Any, tokenizer: Any, suite: dict, *, max_length: int, disable_adapter: bool = False) -> dict:
    """Deterministic batch-one scoring; no sampled judge or second 70B allocation.

    Answer continuations are tokenized separately with a leading space. Mean
    token log likelihood reduces the preference for short choices; option design
    and tokenizer effects remain limitations of this engineering screen.
    """
    import torch
    import torch.nn.functional as functional

    validate_suite(suite)
    if disable_adapter and not callable(getattr(model, "disable_adapter", None)):
        raise ValueError("unadapted_control_requires_disable_adapter_support")
    device = model.get_input_embeddings().weight.device
    was_training = model.training
    model.eval()
    rows = []
    context = model.disable_adapter() if disable_adapter else nullcontext()
    try:
        with context, torch.inference_mode():
            for item in suite["items"]:
                prefix = tokenizer.encode(item["prompt"].rstrip() + "\nAnswer:", add_special_tokens=False)
                if not prefix:
                    raise ValueError("evaluation_prompt_has_no_tokens")
                losses = []
                for choice in item["choices"]:
                    target = tokenizer.encode(" " + choice.strip(), add_special_tokens=False)
                    if not target or len(prefix) + len(target) > max_length:
                        raise ValueError("evaluation_item_exceeds_length_or_has_no_target_tokens")
                    ids = torch.tensor([prefix + target], device=device)
                    outputs = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                    logits = outputs.logits[:, len(prefix) - 1:ids.shape[1] - 1, :].float()
                    loss = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), torch.tensor(target, device=logits.device))
                    losses.append(float(loss.item()))
                    del outputs, logits, ids, loss
                if any(not math.isfinite(value) for value in losses):
                    raise ValueError("nonfinite_independent_evaluation_score")
                prediction = min(range(len(losses)), key=lambda index: (losses[index], index))
                rows.append({"id": item["id"], "group": item["group"], "correct": prediction == item["correct_index"],
                             "predicted_index": prediction, "correct_index": item["correct_index"],
                             "correct_answer_nll": losses[item["correct_index"]], "choice_nlls": losses})
    finally:
        model.train(was_training)
    groups = {}
    for group in sorted(GROUPS):
        selected = [row for row in rows if row["group"] == group]
        groups[group] = {"items": len(selected), "accuracy": sum(row["correct"] for row in selected) / len(selected),
                         "mean_correct_answer_nll": sum(row["correct_answer_nll"] for row in selected) / len(selected)}
    return {"status": "measured", "suite_id": suite["id"], "suite_sha256": suite_hash(suite), "groups": groups, "items": rows}


def compare_evaluations(suite: dict, base: dict, incoming: dict, candidate: dict, policy: dict) -> dict:
    """Require absolute smoke floors and non-regression versus both controls."""
    reasons = []
    expected_hash = suite_hash(validate_suite(suite))
    expected_ids = {item["id"] for item in suite["items"]}
    for name, score in (("unadapted_base", base), ("incoming_parent", incoming), ("trained_candidate", candidate)):
        if score.get("status") != "measured" or score.get("suite_sha256") != expected_hash or {item["id"] for item in score.get("items", [])} != expected_ids:
            reasons.append(name + "_evaluation_missing_or_mismatched")
            continue
        for group in GROUPS:
            metrics = score.get("groups", {}).get(group, {})
            expected_count = sum(item["group"] == group for item in suite["items"])
            accuracy, nll = metrics.get("accuracy"), metrics.get("mean_correct_answer_nll")
            if (metrics.get("items") != expected_count or not isinstance(accuracy, (float, int)) or
                    not math.isfinite(accuracy) or not 0 <= accuracy <= 1 or
                    not isinstance(nll, (float, int)) or not math.isfinite(nll) or nll < 0):
                reasons.append(name + "_invalid_" + group + "_metrics")
    if not reasons:
        for group in sorted(GROUPS):
            cand = candidate["groups"][group]
            floor = policy["min_domain_accuracy" if group == "domain_understanding" else "min_general_accuracy"]
            if cand["accuracy"] < floor:
                reasons.append(group + "_below_minimum_accuracy")
            for name, reference in (("base", base), ("incoming", incoming)):
                ref = reference["groups"][group]
                if cand["accuracy"] + policy["max_accuracy_drop"] + 1e-12 < ref["accuracy"]:
                    reasons.append(group + "_accuracy_regressed_vs_" + name)
                if cand["mean_correct_answer_nll"] > ref["mean_correct_answer_nll"] * policy["max_nll_ratio"] + 1e-8:
                    reasons.append(group + "_nll_regressed_vs_" + name)
    return {"passed": not reasons, "status": "measured", "suite_id": suite["id"], "suite_version": suite["version"],
            "suite_sha256": expected_hash, "kind": suite.get("kind", "operator_evaluation"), "policy": policy,
            "reasons": reasons, "controls": {"unadapted_base": base, "incoming_parent": incoming, "trained_candidate": candidate},
            "limitations": suite["limitations"],
            "interpretation": "Frozen task comparisons; these do not establish consciousness, pain, or scientific benchmark validity."}
