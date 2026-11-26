# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The qualification's outlier gate scales the DINOv3 criteria's error budget."""

from __future__ import annotations

import numpy as np

from families.dinov3 import validation

PROFILE = (
    "dinov3-convnext-tiny-pretrain-lvd1689m"  # no register tokens; relative Frobenius limit 0.015
)


def _side(hidden: np.ndarray) -> dict:
    return {
        "last_hidden_state": hidden.reshape(-1).tolist(),
        "last_hidden_state_shape": list(hidden.shape),
        "pooler_output": hidden[:, 0, :].reshape(-1).tolist(),
        "pooler_output_shape": [hidden.shape[0], hidden.shape[2]],
    }


def test_twice_the_error_budget_admits_a_two_percent_feature_error() -> None:
    expected = np.random.default_rng(0).standard_normal((1, 1 + 49, 16))
    native = {**_side(expected), "num_register_tokens": 0}
    candidate = _side(expected * 1.02)  # every cosine 1, relative Frobenius error 0.02

    strict, reason = validation.compare(candidate, native, {"profile": PROFILE})
    assert strict is False and "relative Frobenius error" in reason
    relaxed, _ = validation.compare(candidate, native, {"profile": PROFILE, "threshold_scale": 2.0})
    assert relaxed is True
    far, _ = validation.compare(
        _side(expected * 1.04), native, {"profile": PROFILE, "threshold_scale": 2.0}
    )
    assert far is False  # 0.04 is beyond twice the budget


def test_scaling_widens_cosine_floors_and_the_frobenius_limit_only() -> None:
    scaled = validation._scaled(
        {"full_cosine": 0.999, "p01_patch_cosine": 0.995, "relative_frobenius": 0.015, "other": 3},
        2.0,
    )
    assert np.isclose(scaled["full_cosine"], 0.998) and np.isclose(scaled["p01_patch_cosine"], 0.99)
    assert np.isclose(scaled["relative_frobenius"], 0.03) and scaled["other"] == 3
