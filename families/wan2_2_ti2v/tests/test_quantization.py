# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wan2.2 TI2V NVFP4 recipe contracts: layers, scales, weight packing and rejected configs."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from families.wan2_2_ti2v import dit_builder as dit
from families.wan2_2_ti2v import fp8_profile, model, quantization, trt_builder
from families.wan2_2_ti2v.model_config import (
    WAN22_TI2V_5B,
    WAN22_TI2V_5B_L0,
    WAN22_TI2V_5B_PORTRAIT,
)


FULL = SimpleNamespace(raw={})


def _raw(profile) -> SimpleNamespace:
    return SimpleNamespace(
        raw={
            "video_width": profile.video_width,
            "video_height": profile.video_height,
            "video_num_frames": profile.video_num_frames,
            "num_inference_steps": profile.num_inference_steps,
        }
    )


# ---------------------------------------------------------------------------
# Recipe and scales
# ---------------------------------------------------------------------------


def test_recipe_is_the_fp8_projection_set() -> None:
    names = quantization.nvfp4_layer_names(WAN22_TI2V_5B)

    assert len(names) == 3 * WAN22_TI2V_5B.num_layers == 90
    assert names[:3] == ("blocks.0.ffn.net.0.proj", "blocks.0.ffn.net.2", "blocks.0.attn2.to_q")
    assert set(names) == set(fp8_profile.load_precomputed_fp8_scales(FULL))
    # Self-attention, the cross-attention key, value and output and the outer projections stay.
    assert not any(
        part in name
        for name in names
        for part in ("attn1", "attn2.to_k", "attn2.to_v", "attn2.to_out", "proj_out")
    )


def test_packaged_scales_are_the_fp8_calibration_divided_by_six() -> None:
    nvfp4 = quantization.load_precomputed_nvfp4_scales(FULL)
    fp8 = fp8_profile.load_precomputed_fp8_scales(FULL)

    assert set(nvfp4) == set(fp8)
    for name, entry in nvfp4.items():
        assert set(entry) == {"input_scale"}
        assert entry["input_scale"] == fp8[name]["input_scale"] / quantization.E2M1_MAX


@pytest.mark.parametrize("profile", [WAN22_TI2V_5B_PORTRAIT, WAN22_TI2V_5B_L0])
def test_scales_reject_unqualified_profiles(profile) -> None:
    with pytest.raises(ValueError, match="only the 1280x704, 121-frame profile"):
        quantization.load_precomputed_nvfp4_scales(_raw(profile))


def test_validated_scales_require_all_and_only_recipe_layers() -> None:
    scales = quantization.load_precomputed_nvfp4_scales(FULL)
    result = quantization.validated_nvfp4_scales(scales, WAN22_TI2V_5B)
    assert result == {name: entry["input_scale"] for name, entry in scales.items()}

    missing = dict(scales)
    missing.pop("blocks.29.attn2.to_q")
    with pytest.raises(ValueError, match="missing=\\['blocks.29.attn2.to_q'\\]"):
        quantization.validated_nvfp4_scales(missing, WAN22_TI2V_5B)

    unexpected = dict(scales)
    unexpected["blocks.0.attn2.to_out.0"] = {"input_scale": 0.01}
    with pytest.raises(ValueError, match="unexpected=\\['blocks.0.attn2.to_out.0'\\]"):
        quantization.validated_nvfp4_scales(unexpected, WAN22_TI2V_5B)


@pytest.mark.parametrize(
    "entry",
    [
        {"input_scale": 0.0},
        {"input_scale": -1.0},
        {"input_scale": float("nan")},
        {"input_scale": float("inf")},
        {"input_scale": True},
        {"input_scale": "0.1"},
        {"input_scale": 0.1, "weight_scale": 0.1},
        0.1,
    ],
)
def test_validated_scales_reject_invalid_entries(entry) -> None:
    scales = quantization.load_precomputed_nvfp4_scales(FULL)
    scales["blocks.0.ffn.net.2"] = entry
    with pytest.raises(ValueError, match="blocks.0.ffn.net.2"):
        quantization.validated_nvfp4_scales(scales, WAN22_TI2V_5B)


# ---------------------------------------------------------------------------
# Weight quantization
# ---------------------------------------------------------------------------


def test_weight_quantization_layout_and_global_scale() -> None:
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((8, 64)).astype(np.float32)
    packed, block_scale, weight_scale_2 = quantization.quantize_nvfp4_weight(weight)

    assert packed.shape == (8, 32) and packed.dtype == np.uint8
    assert block_scale.shape == (8, 4) and block_scale.dtype == np.uint8
    assert weight_scale_2 == pytest.approx(np.abs(weight).max() / (6.0 * 448.0), rel=1e-6)
    restored = quantization.dequantize_nvfp4_weight(packed, block_scale, weight_scale_2)
    blocks = np.abs(weight).reshape(8, 4, 16).max(-1, keepdims=True)
    # E2M1 keeps half a step of the largest grid gap (2 of 6) per block, plus the e4m3 scale error.
    error = np.abs(restored - weight).reshape(8, 4, 16)
    assert np.all(error <= blocks * (1.0 / 6.0 + 1.0 / 16.0))


def test_weight_quantization_is_exact_on_the_grid_and_packs_low_nibble_first() -> None:
    grid = np.asarray([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    row = np.concatenate([grid, -grid[::-1]])
    weight = np.stack([row, row * 0.0])  # second row: an all-zero block gets block scale 1
    packed, block_scale, weight_scale_2 = quantization.quantize_nvfp4_weight(weight)

    assert weight_scale_2 == pytest.approx(6.0 / (6.0 * 448.0))
    np.testing.assert_allclose(
        quantization.dequantize_nvfp4_weight(packed, block_scale, weight_scale_2),
        weight,
        rtol=1e-6,
    )
    assert packed[0, 0] == 0x10  # codes 0 (0.0, low nibble) and 1 (0.5, high nibble)
    assert block_scale[1, 0] == 0x38  # e4m3 1.0


@pytest.mark.parametrize(
    ("weight", "message"),
    [
        (np.ones((4, 24), dtype=np.float32), "multiple of 16"),
        (np.zeros((4, 16), dtype=np.float32), "positive finite"),
        (np.ones((16,), dtype=np.float32), r"\[out, in\]"),
    ],
)
def test_weight_quantization_rejects_unsupported_weights(weight, message) -> None:
    with pytest.raises(ValueError, match=message):
        quantization.quantize_nvfp4_weight(weight)


# ---------------------------------------------------------------------------
# Which projections the DiT graph builds as NVFP4
# ---------------------------------------------------------------------------


class _Weights(dict):
    """Checkpoint stand-in: every key is a mock that remembers its name."""

    def __missing__(self, key):
        value = mock.MagicMock(name=key)
        value.key = key
        self[key] = value
        return value


class _Ops:
    """Records which linear builder each DiT projection goes through."""

    def __init__(self) -> None:
        self.calls: dict[str, list[str]] = {}

    def _record(self, kind):
        def call(network, x, weight, *args, **kwargs):
            key = getattr(weight, "key", "?").removesuffix(".weight")
            self.calls.setdefault(kind, []).append(key)
            return mock.MagicMock()

        return call

    def fused_qkv_linear(self, *args, **kwargs):
        self.calls.setdefault("fused_qkv", []).append("qkv")
        return mock.MagicMock(), mock.MagicMock(), mock.MagicMock()

    def fp8_e4m3_weight_scale(self, weight):
        return 0.0

    def __getattr__(self, name):
        if name in {"linear", "linear_fp8_e4m3", "linear_nvfp4"}:
            return self._record(name)
        return lambda *args, **kwargs: mock.MagicMock()


def _graph(monkeypatch, **scales) -> dict[str, list[str]]:
    ops = _Ops()
    monkeypatch.setattr(dit, "op", ops)
    monkeypatch.setattr(dit, "_numpy_state", lambda model_dir: _Weights())
    builder = mock.MagicMock()
    builder.build_serialized_network.return_value = b"plan"
    monkeypatch.setattr(dit.trt, "Builder", lambda logger: builder)
    assert dit.build_dit_engine("unused", profile=WAN22_TI2V_5B, **scales) == b"plan"
    return ops.calls


def test_nvfp4_graph_quantizes_exactly_the_recipe_projections(monkeypatch) -> None:
    calls = _graph(monkeypatch, nvfp4_scales=quantization.load_precomputed_nvfp4_scales(FULL))

    assert sorted(calls["linear_nvfp4"]) == sorted(quantization.nvfp4_layer_names(WAN22_TI2V_5B))
    assert "linear_fp8_e4m3" not in calls
    assert not set(calls["linear"]) & set(calls["linear_nvfp4"])
    assert "blocks.0.attn2.to_out.0" in calls["linear"]


def test_fp8_and_bf16_graphs_have_no_nvfp4_projection(monkeypatch) -> None:
    ffn, cross = trt_builder._split_qualified_fp8_scales(
        fp8_profile.load_precomputed_fp8_scales(FULL), WAN22_TI2V_5B
    )
    fp8 = _graph(monkeypatch, ffn_fp8_scales=ffn, cross_qo_fp8_scales=cross)
    bf16 = _graph(monkeypatch)

    assert "linear_nvfp4" not in fp8 and "linear_nvfp4" not in bf16
    assert sorted(fp8["linear_fp8_e4m3"]) == sorted(quantization.nvfp4_layer_names(WAN22_TI2V_5B))
    assert "linear_fp8_e4m3" not in bf16


def test_cross_q_routes_to_nvfp4_with_the_packaged_scale(monkeypatch) -> None:
    name = "blocks.3.attn2.to_q"
    calls = []
    monkeypatch.setattr(
        dit.op, "linear_nvfp4", lambda *args, **kwargs: calls.append(kwargs) or "fp4-output"
    )
    weights = {f"{name}.weight": np.ones((2, 16)), f"{name}.bias": np.ones(2)}
    refs: list = []

    output = dit._cross_qo_linear(None, None, weights, name, None, refs, {name: 0.25})

    assert output == "fp4-output"
    assert calls == [{"input_scale": 0.25, "weight_refs": refs}]


# ---------------------------------------------------------------------------
# Rejected configs
# ---------------------------------------------------------------------------


def test_dit_builder_rejects_nvfp4_with_fp8_or_other_profiles_before_loading_weights() -> None:
    scales = quantization.load_precomputed_nvfp4_scales(FULL)
    with pytest.raises(ValueError, match="cannot be combined with FP8"):
        dit.build_dit_engine(
            "unused", profile=WAN22_TI2V_5B, ffn_fp8_scales={}, nvfp4_scales=scales
        )
    for profile in (WAN22_TI2V_5B_PORTRAIT, WAN22_TI2V_5B_L0):
        with pytest.raises(ValueError, match="only the 1280x704, 121-frame profile"):
            dit.build_dit_engine("unused", profile=profile, nvfp4_scales=scales)


def test_component_builder_rejects_nvfp4_with_fp8_or_other_profiles() -> None:
    scales = quantization.load_precomputed_nvfp4_scales(FULL)
    with pytest.raises(ValueError, match="cannot be combined with FP8"):
        trt_builder.build_wan22_components(
            "unused", config=FULL, weights={}, fp8_scales={}, nvfp4_scales=scales
        )
    with pytest.raises(ValueError, match="qualified only for the official"):
        trt_builder.build_wan22_components(
            "unused", config=_raw(WAN22_TI2V_5B_L0), weights={}, nvfp4_scales=scales
        )


def _request(**overrides) -> SimpleNamespace:
    values = dict(
        dynamic_kv_cache=False,
        max_sequence_length=None,
        context_parallel_size=1,
        task="image_generation",
        tensor_parallel_size=1,
        max_batch_size=1,
        fp32_layers=(),
        mtp_seq_len=None,
        precision="bf16",
        image_width=None,
        image_height=None,
        video_num_frames=None,
        model_dir="unused",
        quantization="nvfp4",
        verbose=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class _Stop(Exception):
    pass


def test_build_passes_the_nvfp4_scales_and_no_fp8_scales(monkeypatch) -> None:
    captured = {}

    def build_components(self, model_dir, config, weights, **kwargs):
        captured.update(kwargs)
        raise _Stop

    monkeypatch.setattr(model._Wan22TI2VModel, "load_weights", lambda self, d, c: {})
    monkeypatch.setattr(model._Wan22TI2VModel, "build_components", build_components)
    with pytest.raises(_Stop):
        model.build(_request(), writer=None)

    assert captured["fp8_scales"] is None
    assert captured["nvfp4_scales"] == quantization.load_precomputed_nvfp4_scales(FULL)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        (dict(image_width=704, image_height=1280), "only the 1280x704, 121-frame profile"),
        (dict(image_width=672, image_height=384, video_num_frames=5), "only the 1280x704"),
        (dict(precision="fp16"), "requires precision=bf16"),
        (dict(quantization="int4"), "does not support quantization='int4'"),
    ],
)
def test_build_rejects_unsupported_nvfp4_configs(monkeypatch, overrides, error) -> None:
    monkeypatch.setattr(model._Wan22TI2VModel, "load_weights", lambda self, d, c: {})
    with pytest.raises((ValueError, NotImplementedError), match=error):
        model.build(_request(**overrides), writer=None)
