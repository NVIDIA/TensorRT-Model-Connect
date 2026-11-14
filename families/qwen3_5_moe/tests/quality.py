# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict new-family quality contract; no model-specific relaxed override."""


def normalized_edit_distance(left: str, right: str) -> float:
    """Apply repository default lowercase/whitespace normalization and Levenshtein distance."""
    a = " ".join(left.lower().split())
    b = " ".join(right.lower().split())
    previous = list(range(len(b) + 1))
    for index, char_a in enumerate(a, start=1):
        current = [index]
        for offset, char_b in enumerate(b, start=1):
            current.append(
                min(
                    current[-1] + 1, previous[offset] + 1, previous[offset - 1] + (char_a != char_b)
                )
            )
        previous = current
    return previous[-1] / max(len(a), len(b), 1)


def assert_quality(actual: str, expected: str) -> None:
    """Require nonempty output and predeclared NED <= 0.15 for both image and text."""
    assert actual.strip() and expected.strip()
    assert normalized_edit_distance(actual, expected) <= 0.15


def combined_evidence(gpu: dict, reference: dict) -> dict:
    """Combine independent gates without promoting failed or absent GPU evidence.

    Args:
        gpu: Recorded public/direct GPU validation evidence.
        reference: Recorded independent packed-checkpoint quality evidence.

    Returns:
        New evidence preserving profile and both independently required results.
    """
    gpu_passed = gpu.get("passed") is True
    reference_passed = reference.get("passed") is True
    return {
        **gpu,
        "gpu_passed": gpu_passed,
        "hf_quality": "passed" if reference_passed else "failed",
        "passed": gpu_passed and reference_passed,
        "hf": reference,
    }
