# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A failed DINOv3 feature criterion names itself and its value."""

from __future__ import annotations

import numpy as np
import pytest

from families.dinov3.validation import _assert_at_least, _assert_parity

THRESHOLDS = {
    "full_cosine": 0.999,
    "cls_cosine": 0.999,
    "pooler_cosine": 0.999,
    "register_cosine": 0.999,
    "mean_patch_cosine": 0.999,
    "p01_patch_cosine": 0.995,
    "relative_frobenius": 0.015,
}


def _features(hidden: np.ndarray) -> dict:
    return {
        "last_hidden_state": hidden,
        "last_hidden_state_shape": list(hidden.shape),
        "pooler_output": hidden[:, 0, :],
        "pooler_output_shape": [hidden.shape[0], hidden.shape[2]],
    }


def test_a_failed_criterion_reports_its_value() -> None:
    rng = np.random.default_rng(0)
    expected = rng.standard_normal((1, 1 + 4 + 200, 32))
    actual = expected.copy()
    actual[:, 5:7, :] = rng.standard_normal(
        (1, 2, 32)
    )  # two patch tokens unrelated to the reference
    reference = {**_features(expected), "num_register_tokens": 4}

    with pytest.raises(AssertionError, match=r"full cosine 0\.\d+ < 0\.999"):
        _assert_parity(
            _features(actual),
            reference,
            {"task": "image_features"},
            {"num_register_tokens": 4},
            THRESHOLDS,
        )

    # Within the error budget the patch-cosine tail is the criterion that fails, and it says so.
    loose = {**THRESHOLDS, "relative_frobenius": 1.0, "full_cosine": 0.9, "mean_patch_cosine": 0.9}
    with pytest.raises(AssertionError, match=r"1st-percentile patch cosine -?0\.\d+ < 0\.995"):
        _assert_parity(
            _features(actual),
            reference,
            {"task": "image_features"},
            {"num_register_tokens": 4},
            loose,
        )


def test_a_relative_frobenius_failure_reports_its_value() -> None:
    expected = np.random.default_rng(2).standard_normal((1, 1 + 4 + 16, 8))
    # 2% larger features keep every cosine at 1 but exceed the 1.5% relative Frobenius error budget.
    with pytest.raises(AssertionError, match=r"relative Frobenius error 0\.0200\d* > 0\.015"):
        _assert_parity(
            _features(expected * 1.02),
            {**_features(expected), "num_register_tokens": 4},
            {"task": "image_features"},
            {"num_register_tokens": 4},
            THRESHOLDS,
        )


def test_a_value_just_below_its_limit_is_reported_without_rounding() -> None:
    with pytest.raises(AssertionError, match=r"^full cosine 0\.9989999 < 0\.999$"):
        _assert_at_least("full cosine", 0.9989999, THRESHOLDS, "full_cosine")


def test_identical_features_pass() -> None:
    hidden = np.random.default_rng(1).standard_normal((1, 1 + 4 + 16, 8))
    _assert_parity(
        _features(hidden),
        {**_features(hidden), "num_register_tokens": 4},
        {"task": "image_features"},
        {"num_register_tokens": 4},
        THRESHOLDS,
    )
