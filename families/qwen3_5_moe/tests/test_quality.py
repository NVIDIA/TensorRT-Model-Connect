# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict default quality gate has no short-answer or prefix relaxation."""

import pytest
from families.qwen3_5_moe.tests.quality import assert_quality, normalized_edit_distance


def test_normalization_and_fixed_threshold():
    assert normalized_edit_distance(" A  B ", "a b") == 0
    assert normalized_edit_distance("abc", "abd") == 1 / 3
    assert_quality("abcdefghij", "abcdefghiX")
    for actual, expected in (
        ("abcdefghij", "abcdefghXX"),
        ("Paris", "The answer is Paris"),
        ("", ""),
    ):
        with pytest.raises(AssertionError):
            assert_quality(actual, expected)


@pytest.mark.parametrize("gpu_passed", [True, False, None])
@pytest.mark.parametrize("reference_passed", [True, False, None])
def test_independent_quality_requires_both_gates(gpu_passed, reference_passed):
    """A passing oracle never repairs failed/missing GPU proof or mutates prior evidence."""
    from families.qwen3_5_moe.tests.quality import combined_evidence

    gpu = {"task": "vision_language_generation", "max_sequence_length": 4096}
    reference = {}
    if gpu_passed is not None:
        gpu["passed"] = gpu_passed
    if reference_passed is not None:
        reference["passed"] = reference_passed
    original_gpu = dict(gpu)
    result = combined_evidence(gpu, reference)
    assert result["passed"] is (gpu_passed is True and reference_passed is True)
    assert result["gpu_passed"] is (gpu_passed is True)
    assert result["hf_quality"] == ("passed" if reference_passed is True else "failed")
    assert result["task"] == gpu["task"]
    assert result["max_sequence_length"] == 4096
    assert gpu == original_gpu
