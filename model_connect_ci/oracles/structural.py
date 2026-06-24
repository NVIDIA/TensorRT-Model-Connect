"""Structural oracle helpers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Sequence


@dataclass(frozen=True)
class StructuralResult:
    """Outcome for structural output checks."""

    passed: bool
    message: str


def check_shape(actual: Sequence[int], expected: Sequence[int]) -> StructuralResult:
    """Check output shape exactly."""

    return StructuralResult(
        passed=tuple(actual) == tuple(expected),
        message=(
            "shape matches"
            if tuple(actual) == tuple(expected)
            else f"shape mismatch: expected {tuple(expected)}, got {tuple(actual)}"
        ),
    )


def check_all_finite(values: Iterable[float]) -> StructuralResult:
    """Check that numeric outputs contain no NaN or Inf."""

    for index, value in enumerate(values):
        if not isfinite(value):
            return StructuralResult(
                passed=False,
                message=f"non-finite value at index {index}: {value}",
            )
    return StructuralResult(passed=True, message="all values are finite")


def check_token_range(tokens: Iterable[int], *, vocab_size: int) -> StructuralResult:
    """Check generated token ids stay inside the model vocabulary."""

    for index, token in enumerate(tokens):
        if token < 0 or token >= vocab_size:
            return StructuralResult(
                passed=False,
                message=f"token id {token} at index {index} is outside [0, {vocab_size})",
            )
    return StructuralResult(passed=True, message="all token ids are in range")
