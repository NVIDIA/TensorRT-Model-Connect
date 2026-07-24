# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the Wan2.2-owned TensorRT DiT builder."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect.families.wan2_2_ti2v import dit_builder as dit


def test_dit_builder_rejects_unqualified_profiles_before_loading_weights() -> None:
    with pytest.raises(ValueError, match="not one of the qualified generation profiles"):
        dit.build_dit_engine("unused", profile=SimpleNamespace())


def _fp8_fixture(num_layers: int = 2):
    profile = SimpleNamespace(num_layers=num_layers)
    weights = {}
    scales = {}
    for name in dit._ffn_fp8_layer_names(profile):
        weights[f"{name}.weight"] = np.array(
            [[-2.0, 0.5], [1.0, 0.25]],
            dtype=np.float32,
        )
        scales[name] = {"input_scale": 8.0 / 448.0}
    return profile, weights, scales


def test_ffn_fp8_scales_require_all_and_only_ffn_projections() -> None:
    profile, weights, scales = _fp8_fixture()
    missing = dict(scales)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="missing="):
        dit._validated_ffn_fp8_scales(weights, profile, missing)

    unexpected = dict(scales)
    unexpected["blocks.0.attn1.to_q"] = {"input_scale": 1.0}
    with pytest.raises(ValueError, match="unexpected="):
        dit._validated_ffn_fp8_scales(weights, profile, unexpected)


def test_ffn_fp8_scales_derive_safe_checkpoint_weight_scales() -> None:
    profile, weights, scales = _fp8_fixture()
    result = dit._validated_ffn_fp8_scales(weights, profile, scales)

    assert result is not None
    for name, entry in result.items():
        assert entry["input_scale"] == pytest.approx(8.0 / 448.0)
        assert entry["weight_scale"] == pytest.approx(
            np.max(np.abs(weights[f"{name}.weight"])) / 448.0
        )


def test_ffn_fp8_scales_reject_weight_overflow() -> None:
    profile, weights, scales = _fp8_fixture()
    first = next(iter(scales))
    scales[first]["weight_scale"] = 1.0e-12
    with pytest.raises(ValueError, match="would overflow E4M3"):
        dit._validated_ffn_fp8_scales(weights, profile, scales)
