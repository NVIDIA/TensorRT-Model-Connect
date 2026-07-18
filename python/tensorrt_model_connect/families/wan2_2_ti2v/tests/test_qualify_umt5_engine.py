# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualification gate contracts for the Wan2.2 UMT5 engine."""

from __future__ import annotations

import math

import pytest
import torch

from tensorrt_model_connect.families.wan2_2_ti2v.reference.qualify_umt5_engine import (
    DEFAULT_MAX_RELATIVE_L2_ERROR,
    OFFICIAL_POSITIVE_PROMPT,
    _metrics,
    _qualification,
    _validate_official_positive_prompt,
)


def _prompt_report(cosine: float = 1.0, relative_l2_error: float = 0.0) -> dict:
    metrics = {
        "cosine_similarity": cosine,
        "relative_l2_error": relative_l2_error,
        "bf16_exact": True,
    }
    return {
        "full_512_rows": metrics,
        "real_token_rows": metrics,
        "layers": {},
        "attention_layers": {},
        "token_hash_matches_expected": True,
        "reference_real_hash_matches_expected": True,
    }


def test_qualification_passes_expected_exact_outputs() -> None:
    qualification = _qualification({"positive": _prompt_report()}, 0.998)

    assert qualification["comparisons_checked"] == 2
    assert qualification["worst_cosine_similarity"] == pytest.approx(1.0)
    assert qualification["passed"] is True


def test_qualification_rejects_mismatched_attached_build_report() -> None:
    qualification = _qualification(
        {"positive": _prompt_report()},
        0.998,
        build_report_engine_sha256_matches=False,
    )

    assert qualification["build_report_engine_sha256_matches"] is False
    assert qualification["passed"] is False


def test_qualification_allows_absent_or_matching_build_report_hash() -> None:
    assert _qualification(
        {"positive": _prompt_report()},
        0.998,
        build_report_engine_sha256_matches=None,
    )["passed"]
    assert _qualification(
        {"positive": _prompt_report()},
        0.998,
        build_report_engine_sha256_matches=True,
    )["passed"]


def test_official_qualification_rejects_custom_positive_prompt() -> None:
    _validate_official_positive_prompt(OFFICIAL_POSITIVE_PROMPT)

    with pytest.raises(ValueError, match="positive-prompt is fixed"):
        _validate_official_positive_prompt("a different prompt")


def test_qualification_rejects_scaled_output_despite_perfect_cosine() -> None:
    reference = torch.tensor([-2.0, -1.0, 1.0, 2.0], dtype=torch.float32)
    metrics = _metrics(reference, reference * 1.5)
    assert metrics["cosine_similarity"] == pytest.approx(1.0)
    assert metrics["relative_l2_error"] == pytest.approx(0.5)

    prompt_report = _prompt_report()
    prompt_report["real_token_rows"] = metrics
    qualification = _qualification(
        {"positive": prompt_report},
        0.998,
        DEFAULT_MAX_RELATIVE_L2_ERROR,
    )

    assert qualification["worst_relative_l2_comparison"] == "positive.real_token_rows"
    assert qualification["passed"] is False


@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
def test_qualification_rejects_non_finite_cosines(non_finite: float) -> None:
    qualification = _qualification({"positive": _prompt_report(non_finite)}, 0.998)

    assert len(qualification["non_finite_comparisons"]) == 2
    assert qualification["worst_cosine_similarity"] is None
    assert qualification["passed"] is False


@pytest.mark.parametrize("non_finite", [math.nan, math.inf, -math.inf])
def test_qualification_rejects_non_finite_relative_l2(non_finite: float) -> None:
    qualification = _qualification(
        {"positive": _prompt_report(relative_l2_error=non_finite)}, 0.998
    )

    assert len(qualification["non_finite_relative_l2_comparisons"]) == 2
    assert qualification["worst_relative_l2_error"] is None
    assert qualification["passed"] is False


@pytest.mark.parametrize(
    "key",
    ["token_hash_matches_expected", "reference_real_hash_matches_expected"],
)
def test_qualification_rejects_hash_mismatch(key: str) -> None:
    prompt_report = _prompt_report()
    prompt_report[key] = False

    qualification = _qualification({"positive": prompt_report}, 0.998)

    assert qualification["passed"] is False
