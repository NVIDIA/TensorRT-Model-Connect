# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused host contracts for the atomic full-volume Leaky ALL3 candidate."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect.models.fast_foundation_stereo import (
    native_plugin_builder,
    native_post,
)


def _named(name: str, **attributes):
    instance = type(name, (), {})()
    for attribute, value in attributes.items():
        setattr(instance, attribute, value)
    return instance


def _basic_conv(path: str):
    specs = {
        "corr_feature_att.layers.0": (
            "Conv3d",
            32,
            28,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
            (0, 0, 0),
        ),
        "cost_agg.conv1_up": (
            "ConvTranspose3d",
            56,
            28,
            (4, 4, 4),
            (2, 2, 2),
            (1, 1, 1),
            (1, 1, 1),
            1,
            (0, 0, 0),
        ),
        "cost_agg.post8_to_4.out.0": (
            "Conv3d",
            28,
            28,
            (3, 3, 3),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            1,
            (0, 0, 0),
        ),
    }
    (
        class_name,
        input_channels,
        output_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        output_padding,
    ) = specs[path]
    attributes = {
        "in_channels": input_channels,
        "out_channels": output_channels,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "bias": None,
    }
    attributes["output_padding"] = output_padding
    return _named(
        "BasicConv",
        conv=_named(class_name, **attributes),
        bn=_named("SyncBatchNorm", num_features=28),
        use_bn=True,
        relu=True,
    )


def _model():
    corr_first = _basic_conv("corr_feature_att.layers.0")
    corr_second = _named("BasicConv")
    corr_attention = _named("FeatureAtt")
    post8_first = _basic_conv("cost_agg.post8_to_4.out.0")
    post8 = _named(
        "PostForwardHelper",
        op="sum",
        upsample=(
            _named("BasicConv"),
            _named("CostVolumeDisparityAttention"),
            _named("Upsample"),
        ),
        out=(post8_first, _named("ResnetBasicBlock3D")),
    )
    return SimpleNamespace(
        corr_feature_att=_named(
            "ForwardHelper",
            layers=(corr_first, corr_second, corr_attention),
        ),
        cost_agg=SimpleNamespace(
            conv1_up=_basic_conv("cost_agg.conv1_up"),
            post8_to_4=post8,
        ),
    )


def test_full_volume_leaky_all3_validates_all_three_checkpoint_paths() -> None:
    model = _model()
    native_post._require_full_volume_leaky_all3_topology(model)

    model.cost_agg.conv1_up.conv.stride = (1, 1, 1)
    with pytest.raises(RuntimeError, match="cost_agg.conv1_up"):
        native_post._require_full_volume_leaky_all3_topology(model)


def test_full_volume_leaky_all3_rejects_partial_container_topology() -> None:
    model = _model()
    model.corr_feature_att.layers = tuple(model.corr_feature_att.layers[:2])
    with pytest.raises(RuntimeError, match="corr_feature_att topology"):
        native_post._require_full_volume_leaky_all3_topology(model)

    model = _model()
    model.cost_agg.post8_to_4.out = (model.cost_agg.post8_to_4.out[0],)
    with pytest.raises(RuntimeError, match="post8_to_4 topology"):
        native_post._require_full_volume_leaky_all3_topology(model)


def test_full_volume_leaky_all3_builder_routes_one_input_and_empty_fields(monkeypatch) -> None:
    plugin = object()
    create_calls = []

    class Creator:
        def create_plugin(self, name, fields, phase):
            create_calls.append((name, fields, phase))
            return plugin

    class Layer:
        name = ""

        def __init__(self) -> None:
            self.output = SimpleNamespace(name="")

        def get_output(self, index: int):
            assert index == 0
            return self.output

    class Network:
        def __init__(self) -> None:
            self.arguments = None
            self.layer = Layer()

        def add_plugin_v3(self, inputs, shape_inputs, selected_plugin):
            self.arguments = (inputs, shape_inputs, selected_plugin)
            return self.layer

    def creator(_trt, plugin_name):
        assert plugin_name == "FastFoundationStereoFullVolumeLeaky"
        return Creator()

    monkeypatch.setattr(native_plugin_builder, "_plugin_v3_creator", creator)
    trt_module = SimpleNamespace(
        PluginFieldCollection=lambda fields: tuple(fields),
        TensorRTPhase=SimpleNamespace(BUILD="build"),
    )
    network = Network()
    tensor = object()
    output = native_plugin_builder.add_full_volume_leaky_plugin(
        network,
        tensor,
        trt_module=trt_module,
        name="full_volume_leaky_0",
    )

    assert network.arguments == ([tensor], [], plugin)
    assert network.layer.name == "full_volume_leaky_0"
    assert output.name == "full_volume_leaky_0"
    assert create_calls == [("full_volume_leaky_0", (), "build")]


def test_full_volume_leaky_all3_folded_routes_have_three_unique_names(monkeypatch) -> None:
    fp16 = object()
    events = []

    class Tensor:
        shape = (1, 28, 48, 176, 176)
        dtype = fp16

    class Graph:
        network = object()
        trt = SimpleNamespace(float16=fp16)

        @staticmethod
        def _convolution_batch_norm(tensor, convolution, batch_norm, *, dimensions, deconv):
            events.append(("fold", tensor, convolution, batch_norm, dimensions, deconv))
            return Tensor()

    def add_plugin(network, tensor, *, trt_module, name):
        events.append(("plugin", network, tensor, trt_module, name))
        return Tensor()

    monkeypatch.setattr(native_plugin_builder, "add_full_volume_leaky_plugin", add_plugin)
    graph = Graph()
    names = (
        ("corr_feature_att.layers.0", "corr_feature_att_layers_0_leaky"),
        ("cost_agg.conv1_up", "cost_agg_conv1_up_leaky"),
        ("cost_agg.post8_to_4.out.0", "cost_agg_post8_to_4_out_0_leaky"),
    )
    for path, name in names:
        native_post._folded_basic_conv_full_volume_leaky(
            graph,
            Tensor(),
            _basic_conv(path),
            path=path,
            name=name,
        )

    assert [event[-1] for event in events if event[0] == "plugin"] == [name for _, name in names]
    assert [event[-1] for event in events if event[0] == "fold"] == [False, True, False]


def test_full_volume_leaky_exhaustive_half_model_locks_nan_and_signed_zero() -> None:
    bits = np.arange(1 << 16, dtype=np.uint16)
    values = bits.view(np.float16)
    with np.errstate(invalid="ignore", over="ignore"):
        scaled = (values.astype(np.float32) * np.float32(0.01)).astype(np.float16)
    expected = scaled.view(np.uint16).copy()
    nonnegative = values >= np.float16(0.0)
    expected[nonnegative] = bits[nonnegative]

    assert expected[0x0000] == 0x0000
    assert expected[0x8000] == 0x8000
    assert np.isnan(expected.view(np.float16)[np.isnan(values)]).all()
    assert np.count_nonzero(np.isnan(values)) == 2046
