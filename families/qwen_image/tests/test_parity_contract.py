# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import numpy as np
import pytest

from .test_e2e import CASES, THRESHOLD_ROOT, _assert_pixel_statistics, _initial_latents


def _thresholds() -> dict:
    return {
        "min_pixel_mean": 0.15,
        "max_pixel_mean": 0.85,
        "min_pixel_std": 0.05,
        "reference_min_pixel_std_for_ratio": 0.08,
        "min_reference_std_ratio": 0.35,
        "temporal_consistency": 0.6,
    }


def test_qwen_image_thresholds_only_declare_active_comparisons() -> None:
    paths = sorted(THRESHOLD_ROOT.glob("*.json"))
    assert len(paths) == 4
    for path in paths:
        values = json.loads(path.read_text(encoding="utf-8"))["threshold_overrides"]
        assert "latent_cosine_per_step" not in values
        assert "lpips" not in values
        assert values["reference_min_pixel_std_for_ratio"] == 0.08
        assert values["min_reference_std_ratio"] == 0.35
        assert values["temporal_consistency"] == 0.6


def test_qwen_image_initial_latents_use_the_original_numpy_rng_contract() -> None:
    _, manifest, case = CASES["qwen-image-l0"]
    actual = _initial_latents(manifest, case)
    expected = np.random.default_rng(int(case["seed"])).standard_normal(
        actual.shape, dtype=np.float32
    )
    assert actual.dtype == np.float32
    assert actual.shape == (16, 64, 64)
    np.testing.assert_array_equal(actual, expected)


def test_qwen_image_reference_std_ratio_is_enforced() -> None:
    reference = np.array([[[0.0], [1.0]], [[0.0], [1.0]]], dtype=np.float32)
    native = np.array([[[0.44], [0.56]], [[0.44], [0.56]]], dtype=np.float32)
    with pytest.raises(AssertionError):
        _assert_pixel_statistics([native], [reference], _thresholds())


def test_qwen_image_single_frame_statistics_pass_active_contract() -> None:
    reference = np.array([[[0.0], [1.0]], [[0.0], [1.0]]], dtype=np.float32)
    native = np.array([[[0.3], [0.7]], [[0.3], [0.7]]], dtype=np.float32)
    metrics = _assert_pixel_statistics([native], [reference], _thresholds())
    assert metrics["std_ratio"] == pytest.approx(0.4)
    assert metrics["temporal_consistency"] == 1.0
