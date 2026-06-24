"""Reference oracle helpers for source-vs-converted output comparison."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Sequence


@dataclass(frozen=True)
class ReferenceCompareResult:
    """Outcome of a reference comparison."""

    passed: bool
    metric: str
    value: float
    threshold: float
    message: str


def compare_numeric_sequence(
    reference: Sequence[float],
    converted: Sequence[float],
    *,
    atol: float,
    rtol: float = 0.0,
) -> ReferenceCompareResult:
    """Compare numeric outputs with absolute and relative tolerance."""

    if len(reference) != len(converted):
        return ReferenceCompareResult(
            passed=False,
            metric="length",
            value=float(len(converted)),
            threshold=float(len(reference)),
            message="converted output length differs from reference",
        )
    max_abs_diff = 0.0
    for left, right in zip(reference, converted):
        max_abs_diff = max(max_abs_diff, abs(left - right))
        if not isclose(left, right, abs_tol=atol, rel_tol=rtol):
            return ReferenceCompareResult(
                passed=False,
                metric="max_abs_diff",
                value=max_abs_diff,
                threshold=atol,
                message="converted output exceeded numeric tolerance",
            )
    return ReferenceCompareResult(
        passed=True,
        metric="max_abs_diff",
        value=max_abs_diff,
        threshold=atol,
        message="converted output matches reference within tolerance",
    )


def compare_text_exact(reference: str, converted: str) -> ReferenceCompareResult:
    """Compare normalized decoded text exactly."""

    ref = " ".join(reference.split())
    got = " ".join(converted.split())
    return ReferenceCompareResult(
        passed=ref == got,
        metric="exact_text_match",
        value=1.0 if ref == got else 0.0,
        threshold=1.0,
        message="decoded text matches" if ref == got else "decoded text differs",
    )
