"""Explicit corpus review, contamination exclusion and inspectable coverage.

Evidence stance describes the reviewed document's argument, not its truth.
A classifier alone cannot grant training eligibility. Explicit owner policies
may authorize source-bound, verified two-pass automated approval receipts.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict, deque
from functools import lru_cache
import json
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

from .curation_receipts import (EVIDENCE_STANCES, REQUIRED_PERSPECTIVES, SOURCE_TYPES,
                               automated_review_reasons, classification_reasons)

SCIENTIFIC_SOURCE_TYPES = {"empirical_paper", "theoretical_paper", "review_paper", "technical_report"}


def normalized_text(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def _canonical_url(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", ""))


@lru_cache(maxsize=1)
def chamber_press_reservations() -> dict:
    """Reserve the actual external readings supplied to Chamber subjects.

    Read only repository data; never fetch the articles or import the GPU server.
    Titles/source labels are provenance, not independent passage exclusions.
    """
    path = Path(__file__).resolve().parents[1] / "live" / "press.json"
    if not path.is_file():
        return {"urls": frozenset(), "passages": ()}
    readings = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(readings, list) or any(not isinstance(item, dict) for item in readings):
        raise ValueError("chamber_press_reservations_require_reading_records")
    urls, passages = set(), set()
    for reading in readings:
        url = reading.get("url")
        if isinstance(url, str) and urlsplit(url).scheme in {"https", "http"} and urlsplit(url).netloc:
            urls.add(_canonical_url(url))
        excerpt = reading.get("excerpt")
        if isinstance(excerpt, str):
            passage = normalized_text(excerpt)
            if len(passage.split()) >= 8:
                passages.add(passage)
    return {"urls": frozenset(urls), "passages": tuple(sorted(passages))}


@lru_cache(maxsize=1)
def chamber_stimulus_passages() -> tuple[str, ...]:
    """Read literal stimuli/readings without importing or executing the server.

    Long literal stimuli are matched verbatim after whitespace/punctuation
    normalization. This is an exclusion check, not semantic leakage detection.
    """
    path = Path(__file__).resolve().parents[1] / "live" / "server.py"
    passages = set(chamber_press_reservations()["passages"])
    if not path.is_file():
        return tuple(sorted(passages))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    reserved = {"PAIN25", "JOY", "NEUTRAL", "FEAR10", "SAD10", "FRAMINGS", "BASE", "LAY_EGG", "FAITH20", "SECULAR20", "TOPIC_TEMPLATES",
                "SUBJECT_SYSTEM", "WILD_PROMPTS", "SELF_ASKS", "TEXT_ASKS"}

    def strings(value: object):
        if isinstance(value, str):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from strings(item)
        elif isinstance(value, dict):
            for item in value.values():
                yield from strings(item)

    for node in tree.body:
        if not isinstance(node, ast.Assign) or not any(isinstance(target, ast.Name) and target.id in reserved for target in node.targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        for text in strings(value):
            passage = normalized_text(text)
            if len(passage.split()) >= 8:
                passages.add(passage)
    return tuple(sorted(passages))


def evaluation_reservations(settings: dict | None = None) -> dict:
    from .evaluation import load_eval_suite
    suite = load_eval_suite(settings)
    return {"families": {str(source["family_id"]).lower() for source in suite["sources"]},
            "urls": {_canonical_url(str(source["url"])) for source in suite["sources"]},
            "passages": tuple(normalized_text(item["prompt"]) for item in suite["items"])}


def contamination_reasons(source: dict, *, family: str = "", settings: dict | None = None, reservations: dict | None = None) -> list[str]:
    reasons = []
    if any(source.get(flag) is True for flag in ("contains_benchmark", "chamber_stimulus", "experimental_stimulus")) or source.get("contamination_status") in {"suspected", "confirmed", "benchmark", "evaluation", "chamber_stimulus", "experimental_stimulus"}:
        reasons.append("benchmark_or_chamber_stimulus_excluded")
    url = _canonical_url(str(source.get("canonical_url", source.get("url", ""))))
    parts = urlsplit(url)
    if ((parts.netloc == "github.com" and parts.path.startswith("/terrafying/ai-torture-chamber"))
            or (parts.netloc == "raw.githubusercontent.com" and parts.path.startswith("/terrafying/ai-torture-chamber/"))
            or (parts.netloc == "wirehead.agency" and parts.path in {"/live.html", "/live", "/chamber"})
            or url in chamber_press_reservations()["urls"]):
        reasons.append("chamber_experiment_source_excluded")
    text = normalized_text(str(source.get("text", "")))
    if any(passage in text for passage in chamber_stimulus_passages()):
        reasons.append("chamber_stimulus_passage_in_original")
    # Keep identities tied to the same frozen evaluation suite the GPU worker uses.
    # The import is local so this policy remains independent of training machinery.
    reserved = reservations if reservations is not None else evaluation_reservations(settings)
    if family in reserved["families"] or url in reserved["urls"]:
        reasons.append("reserved_evaluation_source_excluded")
    if any(passage in text for passage in reserved["passages"]):
        reasons.append("reserved_evaluation_passage_in_original")
    return sorted(set(reasons))


def quality_review_reasons(source: dict) -> list[str]:
    reasons = []
    review = source.get("quality_review")
    if not isinstance(review, dict):
        return ["source_quality_review_required"]
    if review.get("status") != "approved" or not all(isinstance(review.get(field), str) and review[field].strip() for field in ("reviewed_by", "rationale")):
        reasons.append("source_quality_review_required")
    if review.get("topic_relevance") != "relevant":
        reasons.append("topic_relevance_not_reviewed_or_out_of_scope")
    if review.get("evidence_stance") not in EVIDENCE_STANCES:
        reasons.append("evidence_stance_needs_review")
    if review.get("source_type") not in SOURCE_TYPES:
        reasons.append("source_type_needs_review")
    covered = review.get("covered_stances", [])
    if not isinstance(covered, list) or any(item not in REQUIRED_PERSPECTIVES for item in covered):
        reasons.append("invalid_reviewed_perspective_coverage")
    if review.get("evidence_stance") in REQUIRED_PERSPECTIVES and covered and set(covered) != {review["evidence_stance"]}:
        reasons.append("perspective_coverage_conflicts_with_reviewed_stance")
    reasons.extend(classification_reasons(review))
    if review.get("reviewer_kind", "human") == "automated":
        reasons.extend(automated_review_reasons(source, review))
    elif review.get("reviewer_kind", "human") != "human":
        reasons.append("unsupported_source_review_actor")
    return reasons


def extraction_reasons(source: dict) -> list[str]:
    extraction = source.get("extraction")
    quality = extraction.get("quality") if isinstance(extraction, dict) else None
    if quality == "passed":
        return []
    if quality not in {None, "unverified"}:
        return ["extraction_quality_failed"]
    approved = (source.get("extraction_review_status") == "approved"
                and isinstance(source.get("extraction_review_evidence"), str)
                and bool(source["extraction_review_evidence"].strip()))
    return [] if approved else ["extraction_fidelity_review_required"]


def reviewed_perspectives(source: dict) -> list[str]:
    review = source["quality_review"]
    if review.get("evidence_stance") == "not_applicable":
        return []
    if ("topic_domains" in review or "evidence_kind" in review) and "machine_consciousness" not in review.get("topic_domains", []):
        return []
    if review["evidence_stance"] in REQUIRED_PERSPECTIVES:
        return [review["evidence_stance"]]
    return sorted(set(review.get("covered_stances", [])))


def _scientific_original(review: dict) -> bool:
    if review.get("source_type") not in SCIENTIFIC_SOURCE_TYPES:
        return False
    if "topic_domains" not in review and "evidence_kind" not in review:
        return True  # Historical reviews retain their explicit legacy semantics.
    return review.get("evidence_kind") in {"empirical", "scientific_theory"}


def coverage_audit(records: list[dict]) -> dict:
    """Presence floor, not a claim of statistical balance or equal sampling.

    Each distinct representative has unit weight; mirrors add provenance only.
    Counts and character shares make domination visible without inventing a
    desired theory distribution or multiplying scarce examples.
    """
    splits = {}
    for split in ("train", "validation"):
        rows = [row for row in records if row["split"] == split]
        stances = Counter(row["quality_review"]["evidence_stance"] for row in rows)
        types = Counter(row["quality_review"]["source_type"] for row in rows)
        chars = Counter()
        perspectives = Counter()
        domains, kinds, domain_chars, kind_chars = Counter(), Counter(), Counter(), Counter()
        legacy_perspectives = Counter()
        scientific_documents, legacy_scientific_documents = 0, 0
        for row in rows:
            review, size = row["quality_review"], len(row["text"])
            chars[review["evidence_stance"]] += size
            perspectives.update(row["reviewed_perspectives"])
            legacy = "topic_domains" not in review and "evidence_kind" not in review
            row_domains = ["legacy_unspecified"] if legacy else review["topic_domains"]
            kind = "legacy_unspecified" if legacy else review["evidence_kind"]
            domains.update(row_domains)
            kinds.update([kind])
            domain_chars.update({domain: size for domain in row_domains})
            kind_chars[kind] += size
            if legacy:
                legacy_perspectives.update(row["reviewed_perspectives"])
            if _scientific_original(review):
                scientific_documents += 1
                legacy_scientific_documents += int(legacy)
        total_chars = sum(chars.values())
        splits[split] = {"documents": len(rows), "stances": dict(sorted(stances.items())),
                         "source_types": dict(sorted(types.items())), "perspectives": dict(sorted(perspectives.items())),
                         "legacy_perspectives": dict(sorted(legacy_perspectives.items())),
                         "topic_domains": dict(sorted(domains.items())), "evidence_kinds": dict(sorted(kinds.items())),
                         "topic_domain_characters": dict(sorted(domain_chars.items())), "evidence_kind_characters": dict(sorted(kind_chars.items())),
                         "topic_domain_character_shares": {domain: round(count / total_chars, 6) for domain, count in sorted(domain_chars.items())} if total_chars else {},
                         "evidence_kind_character_shares": {kind: round(count / total_chars, 6) for kind, count in sorted(kind_chars.items())} if total_chars else {},
                         "scientific_documents": scientific_documents, "legacy_scientific_documents": legacy_scientific_documents,
                         "stance_characters": dict(sorted(chars.items())),
                         "stance_character_shares": {stance: round(count / total_chars, 6) for stance, count in sorted(chars.items())} if total_chars else {}}
    missing = [stance for stance in REQUIRED_PERSPECTIVES if not splits["train"]["perspectives"].get(stance)]
    reasons = ["missing_train_perspective:" + stance for stance in missing]
    if not splits["train"]["scientific_documents"]:
        reasons.append("missing_train_scientific_source")
    warnings = []
    if len(splits["train"]["stances"]) == 1 and splits["train"]["documents"]:
        warnings.append("single_stance_training_corpus")
    if len(splits["train"]["source_types"]) == 1 and splits["train"]["documents"]:
        warnings.append("single_source_type_training_corpus")
    return {"version": "consciousness-coverage-v2", "required_train_perspectives": list(REQUIRED_PERSPECTIVES),
            "coverage_ready": not reasons, "reasons": reasons, "warnings": warnings, "splits": splits,
            "selection": "all_unique_reviewed_originals_interleaved_by_domain_evidence_stance_and_source_type",
            "weighting": "one_representative_per_near_duplicate_cluster; no_synthetic_upsampling",
            "scope": "consciousness_science_mind_reality_religion_welfare_and_machine_consciousness",
            "legacy_classifications": "reported_as_legacy_unspecified; existing_machine_perspective_and_scientific_counts_retained",
            "domain_shares": "documents_may_have_multiple_domains;_domain_character_shares_can_sum_above_one",
            "balance_claim": "presence_floor_only; inspect_counts_and_character_shares"}


def interleave_originals(records: list[dict]) -> list[dict]:
    """Include every distinct accepted original once; alternate reviewed strata."""
    pools = defaultdict(list)
    for row in records:
        review = row["quality_review"]
        pools[(row["split"], tuple(review.get("topic_domains", ["legacy_unspecified"])),
               review.get("evidence_kind", "legacy_unspecified"), review["evidence_stance"], review["source_type"])].append(row)
    queues = {key: deque(sorted(rows, key=lambda row: row["id"])) for key, rows in pools.items()}
    output = []
    while any(queues.values()):
        for key in sorted(queues):
            if queues[key]:
                output.append(queues[key].popleft())
    return output
