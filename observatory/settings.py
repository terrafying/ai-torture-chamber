"""Validate research control inputs before they can affect running workers."""
from __future__ import annotations

import re


def validate_research_updates(updates: dict) -> dict:
    result = dict(updates)
    if "auto_curation_enabled" in result and not isinstance(result["auto_curation_enabled"], bool):
        raise ValueError("auto_curation_enabled must be a boolean")
    if "auto_curation_policy_ack" in result and (not isinstance(result["auto_curation_policy_ack"], str)
            or result["auto_curation_policy_ack"] not in {"", "originals-v1", "originals-v2"}):
        raise ValueError("auto_curation_policy_ack must select a supported originals policy")
    if result.get("auto_curation_enabled") is True and result.get("auto_curation_policy_ack") not in {"originals-v1", "originals-v2"}:
        raise ValueError("Automated curation requires an explicit originals policy acknowledgement")
    choices = {
        "research_provider": {"x402", "openai", "anthropic"},
        "research_protocol": {"responses", "messages"},
        "browser_provider": {"local", "browseruse", "cdp"},
        "reasoning_effort": {"low", "medium", "high", "xhigh"},
    }
    for key, allowed in choices.items():
        if key in result and (not isinstance(result[key], str) or result[key] not in allowed):
            raise ValueError(key + " must be one of: " + ", ".join(sorted(allowed)))
    for key, low, high in (("agent_count", 1, 6), ("research_max_output_tokens", 128, 32768)):
        if key not in result:
            continue
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, (str, int)) or not re.fullmatch(r"\d+", str(value)):
            raise ValueError(key + " must be an integer")
        value = int(value)
        if not low <= value <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
        result[key] = value
    if "research_model" in result:
        value = result["research_model"]
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,199}", value):
            raise ValueError("research_model must be a nonempty model identifier")
    return result
