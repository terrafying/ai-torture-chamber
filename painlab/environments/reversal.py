from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BehavioralSchedule:
    reversal_at: int | None = None
    extinction_at: int | None = None
    devalue_at: int | None = None


def reversal_schedule(
    *, reversal_at: int, extinction_at: int | None = None, devalue_at: int | None = None
) -> BehavioralSchedule:
    return BehavioralSchedule(reversal_at, extinction_at, devalue_at)
