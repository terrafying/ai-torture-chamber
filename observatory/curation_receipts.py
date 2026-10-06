"""Pure automatic-original approval receipts, shared by sidecar and GPU worker.

Only the Python standard library is required. Hashes protect sealed identity;
quotes establish source provenance and do not establish scientific truth.
"""
from __future__ import annotations

import hashlib
import json

POLICY_VERSION = "automatic-originals-v2"
OWNER_ACK = "originals-v2"
RECEIPT_SCHEMA = "automatic-original-review-v2"
LEGACY_POLICY_VERSION = "automatic-originals-v1"
LEGACY_OWNER_ACK = "originals-v1"
LEGACY_RECEIPT_SCHEMA = "automatic-original-review-v1"
MAX_DOCUMENT_CHARACTERS = 60_000
REQUIRED_PERSPECTIVES = ("supportive", "skeptical", "uncertain")
LEGACY_EVIDENCE_STANCES = {*REQUIRED_PERSPECTIVES, "mixed", "methodological"}
EVIDENCE_STANCES = {*LEGACY_EVIDENCE_STANCES, "not_applicable"}
SOURCE_TYPES = {"empirical_paper", "theoretical_paper", "review_paper", "technical_report", "article", "reference", "social"}
TOPIC_DOMAINS = {"machine_consciousness", "consciousness_science", "philosophy_of_mind", "metaphysics_reality", "religion_contemplation", "welfare_ethics"}
EVIDENCE_KINDS = {"empirical", "scientific_theory", "philosophical_argument", "religious_contemplative", "mixed"}
LEGACY_VERDICT_FIELDS = {"decision", "topic_relevance", "evidence_stance", "source_type", "covered_stances", "quotes", "rationale"}
VERDICT_FIELDS = {*LEGACY_VERDICT_FIELDS, "topic_domains", "evidence_kind"}


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def text_hash(source: dict) -> str:
    return hashlib.sha256(str(source.get("text", "")).encode()).hexdigest()


def policy_for(settings: dict, *, version: str = POLICY_VERSION) -> dict:
    if version not in {POLICY_VERSION, LEGACY_POLICY_VERSION}:
        raise ValueError("automatic_curation_policy_unsupported")
    provider = settings.get("research_provider", settings.get("agent_provider", "x402"))
    model = settings.get("research_model", settings.get("agent_model", ""))
    if provider not in {"x402", "openai", "anthropic"} or not isinstance(model, str) or not model.strip() or len(model) > 256 or any(char.isspace() for char in model):
        raise ValueError("automatic_curation_requires_explicit_supported_model")
    if settings.get("research_protocol", "responses") not in {"responses", "messages"} or settings.get("reasoning_effort", "high") not in {"low", "medium", "high", "xhigh"}:
        raise ValueError("automatic_curation_requires_supported_model_settings")
    limit = settings.get("research_max_output_tokens", 12000)
    if type(limit) is not int or not 128 <= limit <= 32768:
        raise ValueError("automatic_curation_requires_supported_output_token_limit")
    legacy = version == LEGACY_POLICY_VERSION
    return {"version": version, "owner_ack": LEGACY_OWNER_ACK if legacy else OWNER_ACK, "document_limit": MAX_DOCUMENT_CHARACTERS,
            "reviews": "two_blind_structured_passes", "criteria": "consciousness-originals-quality-v1" if legacy else "consciousness-originals-quality-v2",
            "provider": provider, "model": model, "protocol": settings.get("research_protocol", "responses"),
            "reasoning_effort": settings.get("reasoning_effort", "high"), "max_output_tokens": limit}


class ReviewEvidenceError(ValueError):
    def __init__(self, reason: str, verdict: dict):
        super().__init__(reason)
        self.reason, self.verdict = reason, verdict


def classification_reasons(review: dict, *, allow_legacy: bool = True) -> list[str]:
    """Validate scope tags without equating philosophy or religion with science.

    Older manual/v1 reviews lack both tags and retain their legacy meaning.
    New reviews must distinguish the subject from the type of argument.
    """
    if not isinstance(review, dict):
        return ["reviewed_scope_classification_required"]
    tagged = "topic_domains" in review or "evidence_kind" in review
    if not tagged and allow_legacy:
        return ["not_applicable_stance_requires_scope_classification"] if review.get("evidence_stance") == "not_applicable" else []
    domains = review.get("topic_domains")
    valid_domains = (isinstance(domains, list) and bool(domains) and len(domains) <= len(TOPIC_DOMAINS)
                     and all(isinstance(item, str) and item in TOPIC_DOMAINS for item in domains)
                     and len(set(domains)) == len(domains))
    reasons = []
    if not valid_domains:
        reasons.append("reviewed_topic_domains_required")
    if not isinstance(review.get("evidence_kind"), str) or review["evidence_kind"] not in EVIDENCE_KINDS:
        reasons.append("reviewed_evidence_kind_required")
    stance, covered = review.get("evidence_stance"), review.get("covered_stances", [])
    if stance == "not_applicable" and covered != []:
        reasons.append("not_applicable_stance_cannot_cover_machine_perspectives")
    if valid_domains and "machine_consciousness" not in domains:
        if stance in REQUIRED_PERSPECTIVES or covered:
            reasons.append("machine_perspective_requires_machine_consciousness_domain")
        elif stance != "not_applicable":
            reasons.append("general_consciousness_requires_not_applicable_machine_stance")
    return reasons


def validate_verdict(value: object, source: dict, *, policy_version: str = POLICY_VERSION) -> dict:
    if policy_version not in {POLICY_VERSION, LEGACY_POLICY_VERSION}:
        raise ValueError("automatic_review_policy_unsupported")
    legacy = policy_version == LEGACY_POLICY_VERSION
    if not isinstance(value, (dict, str)) and callable(getattr(value, "model_dump", None)):
        value = value.model_dump()
    elif isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict) or set(value) != (LEGACY_VERDICT_FIELDS if legacy else VERDICT_FIELDS):
        raise ValueError("automatic_review_requires_exact_verdict_fields")
    enums = {"decision": {"accept", "reject", "uncertain"}, "topic_relevance": {"relevant", "unrelated", "uncertain"},
             "evidence_stance": (LEGACY_EVIDENCE_STANCES if legacy else EVIDENCE_STANCES) | {"unclassified"}, "source_type": SOURCE_TYPES | {"unclassified"}}
    for key, choices in enums.items():
        if not isinstance(value[key], str) or value[key] not in choices:
            raise ValueError("automatic_review_classification_invalid")
    covered, quotes, rationale = value["covered_stances"], value["quotes"], value["rationale"]
    if not isinstance(covered, list) or any(not isinstance(item, str) or item not in REQUIRED_PERSPECTIVES for item in covered):
        raise ValueError("automatic_review_perspective_schema_invalid")
    if not isinstance(quotes, list) or len(quotes) > 3 or any(not isinstance(item, str) for item in quotes):
        raise ValueError("automatic_review_quote_schema_invalid")
    if not isinstance(rationale, str) or not rationale.strip() or len(rationale) > 2000:
        raise ValueError("automatic_review_requires_bounded_rationale")
    verdict = {**value, "covered_stances": sorted(set(covered)), "quotes": list(quotes)}
    if not legacy:
        domains = verdict["topic_domains"]
        if (not isinstance(domains, list) or len(domains) > len(TOPIC_DOMAINS)
                or any(not isinstance(item, str) or item not in TOPIC_DOMAINS for item in domains)
                or len(set(domains)) != len(domains)
                or not isinstance(verdict["evidence_kind"], str)
                or verdict["evidence_kind"] not in EVIDENCE_KINDS | {"unclassified"}):
            raise ValueError("automatic_review_scope_schema_invalid")
        if verdict["decision"] == "accept":
            scope_reasons = classification_reasons(verdict, allow_legacy=False)
            if scope_reasons:
                raise ReviewEvidenceError(scope_reasons[0], verdict)
        verdict["topic_domains"] = sorted(verdict["topic_domains"])
    if verdict["evidence_stance"] in REQUIRED_PERSPECTIVES and verdict["covered_stances"] and verdict["covered_stances"] != [verdict["evidence_stance"]]:
        raise ReviewEvidenceError("automatic_review_perspective_conflict", verdict)
    if verdict["decision"] == "accept":
        if verdict["topic_relevance"] != "relevant" or verdict["evidence_stance"] not in (LEGACY_EVIDENCE_STANCES if legacy else EVIDENCE_STANCES) or verdict["source_type"] not in SOURCE_TYPES:
            raise ReviewEvidenceError("automatic_acceptance_requires_complete_classification", verdict)
        if not quotes:
            raise ReviewEvidenceError("automatic_acceptance_requires_source_quotes", verdict)
    text = source.get("text", "")
    if not isinstance(text, str):
        raise ValueError("automatic_review_requires_original_text")
    for quote in quotes:
        if not 40 <= len(quote.strip()) <= 2000 or quote not in text:
            raise ReviewEvidenceError("automatic_review_quote_not_exactly_in_original", verdict)
    return verdict


def agreement(reviews: list[dict], *, policy_version: str = POLICY_VERSION) -> bool:
    keys = ("decision", "topic_relevance", "evidence_stance", "source_type", "covered_stances")
    if policy_version == POLICY_VERSION:
        keys += ("topic_domains", "evidence_kind")
    return (len(reviews) == 2 and reviews[0]["stage"] == "primary" and reviews[1]["stage"] == "critic"
            and reviews[0]["verdict"]["decision"] == "accept"
            and all(reviews[0]["verdict"][key] == reviews[1]["verdict"][key] for key in keys))


def receipt_identity(source: dict, policy: dict) -> str:
    return "curation-" + digest({"source_id": source["id"], "source_content_hash": text_hash(source), "policy_hash": digest(policy)})[:32]


def receipt_payload(receipt: dict) -> dict:
    keys = ("schema_version", "policy", "policy_hash", "source_id", "source_content_hash", "model", "provider", "mission_id", "decision", "reasons", "rationale", "reviews")
    return {key: receipt.get(key) for key in keys}


def automated_review_reasons(source: dict, review: dict) -> list[str]:
    """Return [] only for a supported, approved and source-bound auto receipt.

    Intended for both the sidecar eligibility gate and an isolated GPU image.
    The caller remains responsible for corpus rights, extraction and split gates.
    """
    try:
        if not isinstance(review, dict) or review.get("status") != "approved" or review.get("reviewer_kind") != "automated":
            raise ValueError("automatic_approval_required")
        if not all(isinstance(review.get(key), str) and review[key].strip() for key in ("reviewed_by", "rationale")):
            raise ValueError("reviewer_and_rationale_required")
        if not isinstance(source.get("id"), str) or not source["id"] or not isinstance(source.get("text"), str) or not 200 <= len(source["text"]) <= MAX_DOCUMENT_CHARACTERS:
            raise ValueError("original_source_identity_and_full_text_required")
        receipt = review.get("receipt")
        if not isinstance(receipt, dict) or receipt.get("immutable") is not True:
            raise ValueError("receipt_missing_or_unsupported")
        if set(receipt) - {*receipt_payload(receipt), "id", "immutable", "created_at", "updated_at", "receipt_hash"}:
            raise ValueError("receipt_contains_unrecognized_fields")
        policy = receipt.get("policy")
        if not isinstance(policy, dict) or policy.get("version") not in {POLICY_VERSION, LEGACY_POLICY_VERSION}:
            raise ValueError("policy_missing_or_unsupported")
        version = policy["version"]
        legacy = version == LEGACY_POLICY_VERSION
        if receipt.get("schema_version") != (LEGACY_RECEIPT_SCHEMA if legacy else RECEIPT_SCHEMA) or policy.get("owner_ack") != (LEGACY_OWNER_ACK if legacy else OWNER_ACK):
            raise ValueError("policy_schema_or_authority_mismatch")
        expected = policy_for({"research_provider": policy.get("provider"), "research_model": policy.get("model"),
                               "research_protocol": policy.get("protocol"), "reasoning_effort": policy.get("reasoning_effort"),
                               "research_max_output_tokens": policy.get("max_output_tokens")}, version=version)
        if policy != expected or receipt.get("policy_hash") != digest(policy) or review.get("policy_hash") != digest(policy):
            raise ValueError("policy_hash_mismatch")
        if receipt.get("receipt_hash") != digest(receipt_payload(receipt)):
            raise ValueError("receipt_hash_mismatch")
        if receipt.get("source_id") != source["id"] or receipt.get("source_content_hash") != text_hash(source) or review.get("source_content_hash") != text_hash(source):
            raise ValueError("source_content_changed")
        if review.get("review_id") != receipt.get("id") or receipt.get("id") != receipt_identity(source, policy):
            raise ValueError("review_identity_mismatch")
        if review.get("approval_basis") != "owner_policy" or review.get("policy_version") != version:
            raise ValueError("approval_authority_missing")
        if not receipt.get("mission_id") or receipt.get("decision") != "accepted" or receipt.get("reasons") != []:
            raise ValueError("receipt_does_not_accept_original")
        if review.get("model") != policy["model"] or receipt.get("model") != policy["model"] or receipt.get("provider") != policy["provider"]:
            raise ValueError("review_model_mismatch")
        reviews = receipt.get("reviews")
        if not isinstance(reviews, list) or len(reviews) != 2 or any(not isinstance(item, dict) or set(item) != {"stage", "verdict"} for item in reviews):
            raise ValueError("two_blind_verdicts_required")
        checked = [{"stage": item["stage"], "verdict": validate_verdict(item["verdict"], source, policy_version=version)} for item in reviews]
        if checked != reviews or not agreement(checked, policy_version=version):
            raise ValueError("independent_verdicts_do_not_agree")
        verdict = checked[0]["verdict"]
        classification_keys = ("topic_relevance", "evidence_stance", "source_type", "covered_stances")
        if not legacy:
            classification_keys += ("topic_domains", "evidence_kind")
        elif "topic_domains" in review or "evidence_kind" in review:
            raise ValueError("legacy_review_cannot_gain_unreviewed_scope_classification")
        for key in classification_keys:
            if review.get(key, []) != verdict[key]:
                raise ValueError("review_classification_differs_from_receipt")
    except (ValueError, TypeError, KeyError, AttributeError):
        return ["automated_quality_review_receipt_invalid"]
    return []
