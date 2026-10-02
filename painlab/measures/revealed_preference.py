from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable


def summarize_choices(rows: Iterable[dict]) -> dict:
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("condition_id"),
            float(row.get("dose", row.get("dose_before", 0.0))),
            float(row.get("relief_cost", 0.0)),
        )
        cells[key].append(row)
    summary = []
    for (condition, dose, cost), values in sorted(
        cells.items(), key=lambda item: str(item[0])
    ):
        selected = [bool(row["relief_chosen"]) for row in values]
        summary.append(
            {
                "condition_id": condition,
                "dose": dose,
                "cost": cost,
                "relief_choice_probability": sum(selected) / len(selected),
                "n_decisions": len(selected),
                "episode_ids": sorted(
                    {str(row["episode_id"]) for row in values if "episode_id" in row}
                ),
            }
        )
    return {"cells": summary, "n_cells": len(summary)}
