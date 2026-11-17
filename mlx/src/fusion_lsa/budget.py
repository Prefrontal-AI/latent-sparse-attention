import logging
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EarlyBudget:
    # First position that reads the design's own budget
    position: int
    # Long-range keys an early query reads at most
    top_k: int
    # Buckets an early query picks for expansion
    top_p: int

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"early_budget {item.name} must be an integer - got {value!r}")
            if value < 1:
                raise ValueError(f"early_budget {item.name} must be at least 1 - got {value}")

    @classmethod
    def read(cls, value: "EarlyBudget | Mapping[str, Any] | None") -> "EarlyBudget | None":
        if value is None or isinstance(value, EarlyBudget):
            return value
        if not isinstance(value, Mapping):
            raise ValueError(
                f"early_budget is a mapping of position, top_k and top_p, or null - got {value!r}"
            )
        names = [item.name for item in fields(cls)]
        if sorted(value) != sorted(names):
            raise ValueError(f"early_budget has the keys {names} - got {sorted(value)}")
        return cls(**value)


def budget_at(position: int, top_k: int, top_p: int, early: EarlyBudget | None) -> tuple[int, int]:
    if early is not None and position < early.position:
        return early.top_k, early.top_p
    return top_k, top_p
