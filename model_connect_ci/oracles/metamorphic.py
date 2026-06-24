"""Metamorphic oracle helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping


@dataclass(frozen=True)
class MetamorphicResult:
    """Outcome for a metamorphic invariant."""

    invariant: str
    passed: bool
    message: str


def check_batch_reordering(
    original: Mapping[Hashable, object],
    reordered: Mapping[Hashable, object],
) -> MetamorphicResult:
    """Validate that per-sample outputs survive batch reordering."""

    if set(original) != set(reordered):
        return MetamorphicResult(
            invariant="batch_reordering",
            passed=False,
            message="sample ids changed after batch reordering",
        )
    mismatched = [key for key in original if original[key] != reordered[key]]
    return MetamorphicResult(
        invariant="batch_reordering",
        passed=not mismatched,
        message=(
            "per-sample outputs preserved"
            if not mismatched
            else f"per-sample outputs changed for {mismatched}"
        ),
    )
