# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side contract tests for the fixed-shape post8 sum IPluginV3."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect.models.fast_foundation_stereo import (
    native_plugin_builder,
    native_post,
)


_CHANNELS = 28
_CHANNEL_PITCH = 32
_TILE_POSITIONS_PER_BLOCK = 32


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
) -> np.ndarray:
    """Mirror the CUDA CTA's tiled transpose and physical DHWC8 writes."""

    positions = linear.shape[1]
    output = np.empty((positions, _CHANNEL_PITCH), dtype=np.float16)
    for tile_start in range(0, positions, _TILE_POSITIONS_PER_BLOCK):
        valid_positions = min(_TILE_POSITIONS_PER_BLOCK, positions - tile_start)
        transposed = np.zeros((_TILE_POSITIONS_PER_BLOCK, _CHANNEL_PITCH), dtype=np.float16)
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


def test_post8_sum_tiled_cpu_model_is_bitwise_equal_to_half_boundary() -> None:
    generator = np.random.default_rng(20260817)
    positions = _TILE_POSITIONS_PER_BLOCK * 2 + 3
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
        actual = _cpu_tiled_kernel_model(linear, skip)

    np.testing.assert_array_equal(actual.view(np.uint16), expected.view(np.uint16))
    assert not np.any(actual[:, _CHANNELS:].view(np.uint16))


def test_post8_sum_builder_routes_two_data_inputs_through_v3(monkeypatch) -> None:
    plugin = object()
    create_calls = []

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

    trt_module = SimpleNamespace(
        PluginFieldCollection=lambda fields: tuple(fields),
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
    )

    assert network.inputs == [linear, skip]
    assert network.shape_inputs == []
    assert network.plugin is plugin
    assert network.layer.name == "post8_to_4_sum"
    assert output.name == "post8_to_4_sum"
    assert create_calls == [("post8_to_4_sum", (), "build")]


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

    def full_volume_leaky(_graph, tensor, child, *, path, name):
        events.append(("full_volume", tensor.name, child, path, name))
        return Tensor("after-basic")

    def add_plugin(
        network,
        plugin_linear,
        plugin_skip,
        *,
        trt_module,
        name="post8_to_4_sum",
    ):
        events.append(("plugin", network, plugin_linear.name, plugin_skip.name, trt_module, name))
        return plugin_output

    monkeypatch.setattr(native_post, "_cost_attention", cost_attention)
    monkeypatch.setattr(
        native_post,
        "_folded_basic_conv_full_volume_leaky",
        full_volume_leaky,
    )
    monkeypatch.setattr(native_plugin_builder, "add_post8_sum_plugin", add_plugin)

    output = native_post._post_forward_helper(
        Graph(),
        skip,
        Tensor("lower"),
        Tensor("feature"),
        module,
        stage="post8_to_4",
    )

    assert output.name == "after-resnet"
    assert [event[0] for event in events] == [
        "module",
        "attention",
        "module",
        "plugin",
        "full_volume",
        "resnet",
    ]
    assert events[3][2:4] == ("linear", "skip")
    assert events[4][1] == "plugin-output"
    assert events[5][1] == "after-basic"
