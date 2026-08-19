# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the Wan2.2-owned TensorRT DiT builder."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect.models.wan2_2_ti2v import dit_builder as dit


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


def _cross_qo_fp8_fixture(num_layers: int = 2):
    profile = SimpleNamespace(num_layers=num_layers)
    weights = {}
    scales = {}
    for name in dit._cross_qo_fp8_layer_names(profile):
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


def test_cross_qo_fp8_layer_names_are_exact() -> None:
    profile = SimpleNamespace(num_layers=2)

    assert dit._cross_qo_fp8_layer_names(profile) == (
        "blocks.0.attn2.to_q",
        "blocks.0.attn2.to_out.0",
        "blocks.1.attn2.to_q",
        "blocks.1.attn2.to_out.0",
    )


def test_cross_qo_fp8_scales_require_all_and_only_qualified_projections() -> None:
    profile, weights, scales = _cross_qo_fp8_fixture()
    missing = dict(scales)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="missing="):
        dit._validated_cross_qo_fp8_scales(weights, profile, missing)

    unexpected = dict(scales)
    unexpected["blocks.0.attn1.to_out.0"] = {"input_scale": 1.0}
    with pytest.raises(ValueError, match="unexpected="):
        dit._validated_cross_qo_fp8_scales(weights, profile, unexpected)


def test_cross_qo_fp8_scales_derive_safe_checkpoint_weight_scales() -> None:
    profile, weights, scales = _cross_qo_fp8_fixture()
    result = dit._validated_cross_qo_fp8_scales(weights, profile, scales)

    assert result is not None
    for name, entry in result.items():
        assert entry["input_scale"] == pytest.approx(8.0 / 448.0)
        assert entry["weight_scale"] == pytest.approx(
            np.max(np.abs(weights[f"{name}.weight"])) / 448.0
        )


def test_cross_qo_fp8_scales_reject_invalid_values() -> None:
    profile, weights, scales = _cross_qo_fp8_fixture()
    first = next(iter(scales))

    invalid_entry = dict(scales)
    invalid_entry[first] = 0.125
    with pytest.raises(TypeError, match="must be a dictionary"):
        dit._validated_cross_qo_fp8_scales(weights, profile, invalid_entry)

    invalid_input = {name: dict(entry) for name, entry in scales.items()}
    invalid_input[first]["input_scale"] = float("inf")
    with pytest.raises(ValueError, match="positive and finite"):
        dit._validated_cross_qo_fp8_scales(weights, profile, invalid_input)

    invalid_weight = {name: dict(entry) for name, entry in scales.items()}
    invalid_weight[first]["weight_scale"] = 1.0e-12
    with pytest.raises(ValueError, match="would overflow E4M3"):
        dit._validated_cross_qo_fp8_scales(weights, profile, invalid_weight)


def test_cross_qo_linear_is_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    name = "blocks.0.attn2.to_q"
    network = object()
    tensor = object()
    weights = {
        f"{name}.weight": np.ones((2, 2), dtype=np.float32),
        f"{name}.bias": np.ones((2,), dtype=np.float32),
    }
    calls = []
    monkeypatch.setattr(
        dit.op,
        "linear",
        lambda *args, **kwargs: calls.append(("bf16", args, kwargs)) or "bf16-output",
    )
    monkeypatch.setattr(
        dit.op,
        "linear_fp8_e4m3",
        lambda *args, **kwargs: calls.append(("fp8", args, kwargs)) or "fp8-output",
    )

    assert dit._cross_qo_linear(network, tensor, weights, name, None, []) == "bf16-output"
    refs = []
    scales = {name: {"input_scale": 0.125, "weight_scale": 0.25}}
    assert dit._cross_qo_linear(network, tensor, weights, name, scales, refs) == "fp8-output"

    assert [call[0] for call in calls] == ["bf16", "fp8"]
    assert calls[1][2] == {
        "input_scale": 0.125,
        "weight_scale": 0.25,
        "weight_refs": refs,
    }


def test_dit_builder_rejects_cross_qo_fp8_without_ffn_before_loading_weights() -> None:
    with pytest.raises(ValueError, match="requires the complete FFN"):
        dit.build_dit_engine(
            "unused",
            profile=dit.WAN22_TI2V_5B,
            cross_qo_fp8_scales={},
        )
