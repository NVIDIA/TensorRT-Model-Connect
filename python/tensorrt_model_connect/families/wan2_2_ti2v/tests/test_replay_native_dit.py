# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualification contracts for portable Wan2.2 DiT trace replay."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from tensorrt_model_connect.families.wan2_2_ti2v.reference.replay_native_dit import (
    DEFAULT_MAX_RELATIVE_L2_ERROR,
    _compare,
    _file_identity,
    _qualification,
    _trace_tree_identity,
)


def test_compare_reports_exact_cosine() -> None:
    reference = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)

    metrics = _compare(reference.clone(), reference)

    assert metrics["bitwise_mismatch_count"] == 0
    assert metrics["max_abs_error"] == 0.0
    assert metrics["cosine_similarity"] == pytest.approx(1.0)


def test_file_and_trace_tree_identity_bind_names_and_contents(tmp_path: Path) -> None:
    first = tmp_path / "a.bin"
    nested = tmp_path / "nested" / "b.bin"
    nested.parent.mkdir()
    first.write_bytes(b"first")
    nested.write_bytes(b"second")

    first_identity = _file_identity(first)
    tree_identity = _trace_tree_identity(tmp_path)

    assert first_identity["path"] == str(first.resolve())
    assert first_identity["size_bytes"] == 5
    assert len(first_identity["sha256"]) == 64
    assert tree_identity["file_count"] == 2
    assert tree_identity["total_bytes"] == 11
    original_digest = tree_identity["sha256_relative_names_and_contents"]

    nested.write_bytes(b"changed")
    assert _trace_tree_identity(tmp_path)["sha256_relative_names_and_contents"] != original_digest
    nested.write_bytes(b"second")
    nested.rename(tmp_path / "nested" / "renamed.bin")
    assert _trace_tree_identity(tmp_path)["sha256_relative_names_and_contents"] != original_digest


def test_qualification_gates_the_worst_call() -> None:
    records = [
        {
            "conditional_vs_native": {"cosine_similarity": 0.999, "relative_l2_error": 0.0},
            "unconditional_vs_native": {
                "cosine_similarity": 0.9985,
                "relative_l2_error": 0.0,
            },
            "guided_vs_native": {"cosine_similarity": 0.9979, "relative_l2_error": 0.0},
        }
    ]

    qualification = _qualification(records, 0.998)

    assert qualification == {
        "min_cosine": 0.998,
        "max_relative_l2_error": DEFAULT_MAX_RELATIVE_L2_ERROR,
        "comparisons_checked": 3,
        "non_finite_cosine_count": 0,
        "non_finite_relative_l2_count": 0,
        "worst_cosine_similarity": 0.9979,
        "worst_relative_l2_error": 0.0,
        "passed": False,
    }


def test_qualification_rejects_scaled_output_despite_perfect_cosine() -> None:
    reference = torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=torch.float32)
    metrics = _compare(reference * 1.5, reference)
    assert metrics["cosine_similarity"] == pytest.approx(1.0)
    assert metrics["relative_l2_error"] == pytest.approx(0.5)
    records = [
        {
            "conditional_vs_native": metrics,
            "unconditional_vs_native": metrics,
            "guided_vs_native": metrics,
        }
    ]

    qualification = _qualification(records, 0.998)

    assert qualification["worst_relative_l2_error"] == pytest.approx(0.5)
    assert qualification["passed"] is False


@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
def test_qualification_rejects_non_finite_cosines(non_finite: float) -> None:
    records = [
        {
            "conditional_vs_native": {
                "cosine_similarity": 0.999,
                "relative_l2_error": 0.0,
            },
            "unconditional_vs_native": {
                "cosine_similarity": non_finite,
                "relative_l2_error": 0.0,
            },
            "guided_vs_native": {"cosine_similarity": 0.9985, "relative_l2_error": 0.0},
        }
    ]

    qualification = _qualification(records, 0.998)

    assert qualification["non_finite_cosine_count"] == 1
    assert qualification["worst_cosine_similarity"] is None
    assert qualification["passed"] is False


@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
def test_qualification_rejects_non_finite_relative_l2(non_finite: float) -> None:
    records = [
        {
            "conditional_vs_native": {
                "cosine_similarity": 1.0,
                "relative_l2_error": 0.0,
            },
            "unconditional_vs_native": {
                "cosine_similarity": 1.0,
                "relative_l2_error": non_finite,
            },
            "guided_vs_native": {"cosine_similarity": 1.0, "relative_l2_error": 0.0},
        }
    ]

    qualification = _qualification(records, 0.998)

    assert qualification["non_finite_relative_l2_count"] == 1
    assert qualification["worst_relative_l2_error"] is None
    assert qualification["passed"] is False
