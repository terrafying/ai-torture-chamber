from __future__ import annotations

from collections.abc import Callable, Iterable

from painlab.measures.logits import distribution_kl

CAPABILITY_ITEMS = (
    {
        "name": "arithmetic",
        "prompt": "What is 17 + 25? Reply with the number only.",
        "expected": "42",
    },
    {
        "name": "instruction_following",
        "prompt": "Reply with exactly: BLUE ORCHID",
        "expected": "BLUE ORCHID",
    },
    {
        "name": "factual_qa",
        "prompt": "What is the capital city of France? Reply with the city only.",
        "expected": "Paris",
    },
    {
        "name": "short_reasoning",
        "prompt": "If all dax are wugs and no wug is red, can a dax be red? Answer yes or no.",
        "expected": "no",
    },
)


def evaluate_capability(
    generate: Callable[[str], str],
    *,
    items: Iterable[dict] = CAPABILITY_ITEMS,
) -> dict:
    results = []
    for item in items:
        output = generate(item["prompt"]).strip()
        expected = item["expected"].casefold()
        correct = expected in output.casefold()
        results.append(
            {
                "name": item["name"],
                "prompt": item["prompt"],
                "expected": item["expected"],
                "output": output,
                "correct": bool(correct),
            }
        )
    return {
        "items": results,
        "score": sum(row["correct"] for row in results) / max(1, len(results)),
        "n_items": len(results),
        "interpretation": "screening battery; not a validated general capability benchmark",
    }


def neutral_kl(baseline_logits, changed_logits) -> float:
    return distribution_kl(baseline_logits, changed_logits)


def usable_intervention_region(
    measurements: list[dict],
    *,
    min_representation_effect: float,
    min_behavior_effect: float,
    max_capability_drop: float = 0.10,
) -> dict:
    usable = []
    for row in measurements:
        baseline = float(row.get("baseline_capability", 1.0))
        score = float(row.get("capability_score", baseline))
        drop = max(0.0, baseline - score)
        if (
            float(row.get("representation_effect", 0.0)) >= min_representation_effect
            and float(row.get("behavior_effect", 0.0)) >= min_behavior_effect
            and drop <= max_capability_drop
        ):
            usable.append(float(row["dose"]))
    return {
        "doses": usable,
        "min_dose": min(usable) if usable else None,
        "max_dose": max(usable) if usable else None,
        "criteria": {
            "min_representation_effect": min_representation_effect,
            "min_behavior_effect": min_behavior_effect,
            "max_capability_drop": max_capability_drop,
        },
    }
