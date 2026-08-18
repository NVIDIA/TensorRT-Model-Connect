# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side contract tests for the fixed-shape post8 sum IPluginV3."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect.families.fast_foundation_stereo import (
    native_plugin_builder,
    native_post,
)


_CHANNELS = 28
_CHANNEL_PITCH = 32
_TILE_POSITIONS = (32, 64, 128, 256)


def _named_module(name: str, **attributes):
    module = type(name, (), {})()
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def _post8_module():
    convolution = _named_module(
        "Conv3d",
        in_channels=28,
        out_channels=28,
        kernel_size=(4, 4, 4),
        stride=(4, 4, 4),
        padding=(0, 0, 0),
        dilation=(1, 1, 1),
        groups=1,
    )
    pre_sum = _named_module(
        "BasicConv",
        conv=convolution,
        bn=_named_module("SyncBatchNorm"),
        use_bn=True,
        relu=False,
    )
    attention = _named_module("CostVolumeDisparityAttention", resize_embed=False)
    resize = _named_module(
        "Upsample",
        size=None,
        scale_factor=4.0,
        mode="trilinear",
        align_corners=False,
        recompute_scale_factor=None,
    )
    return _named_module(
        "PostForwardHelper",
        op="sum",
        upsample=(pre_sum, attention, resize),
        out=(_named_module("BasicConv"), _named_module("ResnetBasicBlock3D")),
    )


def _cpu_post8_sum_oracle(linear: np.ndarray, skip_dhwc8: np.ndarray) -> np.ndarray:
    """Apply the precise TensorRT boundary: half inputs, FP32 add, half output."""

    positions = linear.shape[1]
    output = np.zeros((positions, _CHANNEL_PITCH), dtype=np.float16)
    output[:, :_CHANNELS] = (
        linear[:, :positions].T.astype(np.float32) + skip_dhwc8[:, :_CHANNELS].astype(np.float32)
    ).astype(np.float16)
    return output


def _cpu_tiled_kernel_model(
    linear: np.ndarray,
    skip_dhwc8: np.ndarray,
    *,
    tile_positions: int,
) -> np.ndarray:
    """Mirror the CUDA CTA's tiled transpose and physical DHWC8 writes."""

    positions = linear.shape[1]
    output = np.empty((positions, _CHANNEL_PITCH), dtype=np.float16)
    for tile_start in range(0, positions, tile_positions):
        valid_positions = min(tile_positions, positions - tile_start)
        transposed = np.zeros((tile_positions, _CHANNEL_PITCH), dtype=np.float16)
        transposed[:valid_positions, :_CHANNELS] = linear[
            :, tile_start : tile_start + valid_positions
        ].T
        tile = output[tile_start : tile_start + valid_positions]
        tile[:, :_CHANNELS] = (
            transposed[:valid_positions, :_CHANNELS].astype(np.float32)
            + skip_dhwc8[tile_start : tile_start + valid_positions, :_CHANNELS].astype(np.float32)
        ).astype(np.float16)
        tile[:, _CHANNELS:] = np.float16(0.0)
    return output


@pytest.mark.parametrize("tile_positions", _TILE_POSITIONS)
def test_post8_sum_tiled_cpu_model_is_bitwise_equal_to_half_boundary(
    tile_positions: int,
) -> None:
    generator = np.random.default_rng(20260817)
    positions = tile_positions * 2 + 3
    linear = generator.uniform(-32.0, 32.0, (_CHANNELS, positions)).astype(np.float16)
    skip = generator.uniform(-32.0, 32.0, (positions, _CHANNEL_PITCH)).astype(np.float16)

    edge_values = np.array(
        [0.0, -0.0, 2**-24, -(2**-24), 2**-14, -(2**-14), 65504.0, -65504.0],
        dtype=np.float16,
    )
    linear[0, : edge_values.size] = edge_values
    skip[: edge_values.size, 0] = edge_values[::-1]
    # Nonzero padded input proves that the plugin owns and clears physical lanes 28..31.
    skip[:, _CHANNELS:] = np.float16(123.0)

    with np.errstate(over="ignore", invalid="ignore"):
        expected = _cpu_post8_sum_oracle(linear, skip)
        actual = _cpu_tiled_kernel_model(linear, skip, tile_positions=tile_positions)

    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert not np.any(actual[:, _CHANNELS:].view(np.uint16))


def test_post8_sum_v3_source_owns_exact_format_rounding_and_tail_contract() -> None:
    plugin_dir = (
        Path(__file__).resolve().parents[4]
        / "python/tensorrt_model_connect/families/fast_foundation_stereo/native_plugins"
    )
    source = (plugin_dir / "post8_sum_plugin.cu").read_text(encoding="utf-8")
    header = (plugin_dir / "post8_sum_plugin.h").read_text(encoding="utf-8")
    creator = (plugin_dir / "post8_sum_plugin_creator.cpp").read_text(encoding="utf-8")
    combined = source + header + creator

    assert "FastFoundationStereoPost8Sum" in header
    assert "IPluginV3" in header
    assert "IPluginCreatorV3One" in creator
    assert "IPluginV2" not in combined
    assert "onnx" not in combined.lower()
    for fixed_contract in (
        "kBatch = 1",
        "kChannels = 28",
        "kDisparities = 48",
        "kHeight = 176",
        "kWidth = 176",
        "kChannelPitch = 32",
    ):
        assert fixed_contract in header
    assert "TensorFormat::kLINEAR" in source
    assert "TensorFormat::kDHWC8" in source
    assert "input_count == 2" in source
    assert "output_count == 1" in source
    assert "template <int32_t TilePositions>" in source
    assert "transposed[TilePositions][Plugin::kChannelPitch + 1]" in source
    assert "linear[static_cast<std::size_t>(channel) * kLogicalElements + position]" in source
    assert "static_cast<std::size_t>(position) * Plugin::kChannelPitch + channel" in source
    assert "__float2half_rn(__fadd_rn(linear_value, skip_value))" in source
    assert "channel < Plugin::kChannels" in source
    assert source.count("<<<") == 1


def test_post8_sum_v3_has_strict_tile_serialization_clone_and_registered_creator() -> None:
    plugin_dir = (
        Path(__file__).resolve().parents[4]
        / "python/tensorrt_model_connect/families/fast_foundation_stereo/native_plugins"
    )
    source = (plugin_dir / "post8_sum_plugin.cu").read_text(encoding="utf-8")
    header = (plugin_dir / "post8_sum_plugin.h").read_text(encoding="utf-8")
    creator = (plugin_dir / "post8_sum_plugin_creator.cpp").read_text(encoding="utf-8")

    assert 'kPLUGIN_VERSION = "1"' in header
    assert "fields.nbFields != 1" in source
    assert 'std::strcmp(field.name, "tile_positions") != 0' in source
    assert "field.data == nullptr" in source
    assert "field.type != nvinfer1::PluginFieldType::kINT32" in source
    assert "field.length != 1" in source
    for tile_positions in _TILE_POSITIONS:
        assert f"value == {tile_positions}" in source
        assert f"launch_post8_sum<{tile_positions}>" in source
        assert f"tile={tile_positions}" in source
        assert f"tile{tile_positions}-v1" in source
    assert "getFieldsToSerialize" in header
    assert "refreshSerializationField();" in source
    assert "return valid_ ? &serialization_collection_ : nullptr" in source
    assert '{"tile_positions", nullptr, nvinfer1::PluginFieldType::kINT32, 1}' in creator
    assert "TensorRTPhase" in creator
    assert "PluginRegistrar<trtmc::FastFoundationStereoPost8SumCreator>" in creator
    assert "tile_positions_(other.tile_positions_)" in source
    assert "std::string" not in header + source
    assert "namespace_" not in header + source
    assert 'return "";' in source
    for capability in ("kCORE", "kBUILD", "kRUNTIME"):
        assert f"PluginCapabilityType::{capability}" in source


def test_post8_sum_flag_is_default_on_with_explicit_fallback(monkeypatch) -> None:
    variable = "TRTMC_FAST_FOUNDATION_STEREO_POST8_SUM_PLUGIN"
    monkeypatch.delenv(variable, raising=False)
    assert native_post._post8_sum_plugin_enabled()
    monkeypatch.setenv(variable, "1")
    assert native_post._post8_sum_plugin_enabled()
    monkeypatch.setenv(variable, "0")
    assert not native_post._post8_sum_plugin_enabled()


def test_post8_sum_tile_positions_env_is_independent_and_strict(monkeypatch) -> None:
    variable = "TRTMC_FAST_FOUNDATION_STEREO_POST8_SUM_TILE_POSITIONS"
    monkeypatch.delenv(variable, raising=False)
    assert native_post._post8_sum_tile_positions() == 32
    for tile_positions in _TILE_POSITIONS:
        monkeypatch.setenv(variable, str(tile_positions))
        assert native_post._post8_sum_tile_positions() == tile_positions
    for invalid in ("", "31", "512", "not-an-integer"):
        monkeypatch.setenv(variable, invalid)
        with pytest.raises(RuntimeError, match="must be one of"):
            native_post._post8_sum_tile_positions()


def test_post8_sum_builder_routes_two_data_inputs_through_v3(monkeypatch) -> None:
    plugin = object()
    create_calls = []
    fields_seen = []

    class Creator:
        def create_plugin(self, name, fields, phase):
            create_calls.append((name, fields, phase))
            return plugin

    class Output:
        name = ""

    class Layer:
        name = ""

        def __init__(self) -> None:
            self.output = Output()

        def get_output(self, index: int):
            assert index == 0
            return self.output

    class Network:
        def __init__(self) -> None:
            self.inputs = None
            self.shape_inputs = None
            self.plugin = None
            self.layer = Layer()

        def add_plugin_v3(self, inputs, shape_inputs, selected_plugin):
            self.inputs = inputs
            self.shape_inputs = shape_inputs
            self.plugin = selected_plugin
            return self.layer

    def plugin_creator(_trt, plugin_name):
        assert plugin_name == "FastFoundationStereoPost8Sum"
        return Creator()

    monkeypatch.setattr(native_plugin_builder, "_plugin_v3_creator", plugin_creator)

    def plugin_field(name, data, field_type):
        fields_seen.append((name, data, field_type))
        return name

    trt_module = SimpleNamespace(
        PluginField=plugin_field,
        PluginFieldCollection=lambda fields: tuple(fields),
        PluginFieldType=SimpleNamespace(INT32="int32"),
        TensorRTPhase=SimpleNamespace(BUILD="build"),
    )
    linear = object()
    skip = object()
    network = Network()

    output = native_plugin_builder.add_post8_sum_plugin(
        network,
        linear,
        skip,
        trt_module=trt_module,
        tile_positions=128,
    )

    assert network.inputs == [linear, skip]
    assert network.shape_inputs == []
    assert network.plugin is plugin
    assert network.layer.name == "post8_to_4_sum"
    assert output.name == "post8_to_4_sum"
    assert create_calls == [("post8_to_4_sum", ("tile_positions",), "build")]
    assert len(fields_seen) == 1
    assert fields_seen[0][0] == "tile_positions"
    assert fields_seen[0][1].dtype == np.int32
    np.testing.assert_array_equal(fields_seen[0][1], np.asarray([128], dtype=np.int32))
    assert fields_seen[0][2] == "int32"


@pytest.mark.parametrize("tile_positions", (None, True, 31, 512, 32.0, "32"))
def test_post8_sum_builder_rejects_non_int32_or_unsupported_tile(
    tile_positions,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        native_plugin_builder,
        "_plugin_v3_creator",
        lambda *_args: pytest.fail("invalid tile must fail before creator lookup"),
    )

    with pytest.raises(ValueError, match="tile_positions must be one of"):
        native_plugin_builder.add_post8_sum_plugin(
            object(),
            object(),
            object(),
            trt_module=object(),
            tile_positions=tile_positions,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (lambda module: setattr(module, "op", "concat"), "post8_to_4 topology"),
        (
            lambda module: setattr(module.upsample[1], "resize_embed", True),
            "pre-sum contract",
        ),
        (
            lambda module: setattr(module.upsample[2], "scale_factor", 2.0),
            "pre-sum contract",
        ),
        (
            lambda module: setattr(module, "out", (_named_module("BasicConv"),)),
            "post8_to_4 topology",
        ),
    ),
)
def test_post8_sum_rejects_non_checkpoint_topology(mutation, match) -> None:
    module = _post8_module()
    mutation(module)

    with pytest.raises(RuntimeError, match=match):
        native_post._require_post8_sum_plugin_topology(
            module,
            tuple(module.upsample),
            tuple(module.out),
        )


@pytest.mark.parametrize(
    ("linear_shape", "skip_shape", "linear_dtype", "skip_dtype", "match"),
    (
        ((1, 27, 48, 176, 176), (1, 28, 48, 176, 176), "fp16", "fp16", "logical shapes"),
        ((1, 28, 48, 176, 176), (1, 28, 48, 176, 175), "fp16", "fp16", "logical shapes"),
        ((1, 28, 48, 176, 176), (1, 28, 48, 176, 176), "fp32", "fp16", "FP16"),
        ((1, 28, 48, 176, 176), (1, 28, 48, 176, 176), "fp16", "fp32", "FP16"),
    ),
)
def test_post8_sum_rejects_wrong_shape_or_dtype(
    linear_shape,
    skip_shape,
    linear_dtype,
    skip_dtype,
    match,
) -> None:
    graph = SimpleNamespace(network=object(), trt=SimpleNamespace(float16="fp16"))
    linear = SimpleNamespace(shape=linear_shape, dtype=linear_dtype)
    skip = SimpleNamespace(shape=skip_shape, dtype=skip_dtype)

    with pytest.raises(RuntimeError, match=match):
        native_post._post8_sum_plugin_output(
            graph,
            linear,
            skip,
            tile_positions=32,
        )


def test_post8_sum_replaces_only_sum_and_preserves_downstream_modules(monkeypatch) -> None:
    events = []
    fp16 = object()

    class Tensor:
        def __init__(self, name: str, shape=(1, 28, 48, 176, 176), dtype=fp16) -> None:
            self.name = name
            self.shape = shape
            self.dtype = dtype

    module = _post8_module()
    linear = Tensor("linear")
    skip = Tensor("skip")
    plugin_output = Tensor("plugin-output")

    class Graph:
        network = object()
        trt = SimpleNamespace(float16=fp16)

        @staticmethod
        def module(tensor, child):
            events.append(("module", tensor.name, child.__class__.__name__))
            return linear if child.__class__.__name__ == "Upsample" else Tensor("pre-resize")

        @staticmethod
        def add(*_args):
            pytest.fail("enabled post8 path must not materialize a TensorRT elementwise sum")

        @staticmethod
        def basic_conv(tensor, child, *, fold_batch_norm=False):
            events.append(("basic_conv", tensor.name, child, fold_batch_norm))
            return Tensor("after-basic")

        @staticmethod
        def resnet(tensor, child, *, fold_batch_norm=False):
            events.append(("resnet", tensor.name, child, fold_batch_norm))
            return Tensor("after-resnet")

    def cost_attention(_graph, tensor, child):
        events.append(("attention", tensor.name, child))
        return Tensor("attended")

    def add_plugin(
        network,
        plugin_linear,
        plugin_skip,
        *,
        trt_module,
        name="post8_to_4_sum",
        tile_positions=32,
    ):
        events.append(
            (
                "plugin",
                network,
                plugin_linear.name,
                plugin_skip.name,
                trt_module,
                name,
                tile_positions,
            )
        )
        return plugin_output

    monkeypatch.setattr(native_post, "_cost_attention", cost_attention)
    monkeypatch.setattr(native_plugin_builder, "add_post8_sum_plugin", add_plugin)

    output = native_post._post_forward_helper(
        Graph(),
        skip,
        Tensor("lower"),
        Tensor("feature"),
        module,
        fold_batch_norm=True,
        post8_sum_plugin=True,
        post8_sum_tile_positions=128,
    )

    assert output.name == "after-resnet"
    assert [event[0] for event in events] == [
        "module",
        "attention",
        "module",
        "plugin",
        "basic_conv",
        "resnet",
    ]
    assert events[3][2:4] == ("linear", "skip")
    assert events[3][6] == 128
    assert events[4][1] == "plugin-output"
    assert events[5][1] == "after-basic"


def test_post8_sum_shared_integration_is_v3_native_and_scoped_to_post8() -> None:
    plugin_dir = (
        Path(__file__).resolve().parents[4]
        / "python/tensorrt_model_connect/families/fast_foundation_stereo/native_plugins"
    )
    cmake = (plugin_dir / "CMakeLists.txt").read_text(encoding="utf-8")
    builder = (plugin_dir.parent / "native_plugin_builder.py").read_text(encoding="utf-8")
    post = (plugin_dir.parent / "native_post.py").read_text(encoding="utf-8")

    assert native_plugin_builder._POST8_SUM_PLUGIN_NAME == "FastFoundationStereoPost8Sum"
    assert "post8_sum_plugin.cu" in cmake
    assert "post8_sum_plugin_creator.cpp" in cmake
    assert "def add_post8_sum_plugin(" in builder
    assert "add_plugin_v3" in builder
    assert "TRTMC_FAST_FOUNDATION_STEREO_POST8_SUM_PLUGIN" in post
    assert "TRTMC_FAST_FOUNDATION_STEREO_POST8_SUM_TILE_POSITIONS" in post
    assert (
        "post8_sum_tile_positions = _post8_sum_tile_positions() if post8_sum_plugin else 32" in post
    )
    assert post.count("post8_sum_plugin=post8_sum_plugin") == 2
    assert "post8_sum_plugin=True" not in post
