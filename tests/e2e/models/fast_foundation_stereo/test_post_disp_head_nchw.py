# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_model_connect.families.fast_foundation_stereo import native_post
from tensorrt_model_connect.families.fast_foundation_stereo.native_graph import NativeGraph


_ENV = "TRTMC_FAST_FOUNDATION_STEREO_DISP_HEAD_NCHW_POINTWISE"
_FOLD_ENV = "TRTMC_FAST_FOUNDATION_STEREO_DISP_HEAD_FOLD_GAMMA"
_TANH_ENV = "TRTMC_FAST_FOUNDATION_STEREO_DISP_HEAD_SECOND_GELU_TANH"


class _Tensor:
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype


def _conv(in_channels, out_channels, kernel_size, padding, *, groups=1):
    Conv2d = type("Conv2d", (), {})
    module = Conv2d()
    module.in_channels = in_channels
    module.out_channels = out_channels
    module.kernel_size = kernel_size
    module.stride = (1, 1)
    module.padding = padding
    module.dilation = (1, 1)
    module.groups = groups
    module.weight = np.zeros((out_channels, in_channels // groups, *kernel_size), dtype=np.float32)
    module.bias = np.zeros((out_channels,), dtype=np.float32)
    return module


def _linear(in_features, out_features):
    Linear = type("Linear", (), {})
    module = Linear()
    module.in_features = in_features
    module.out_features = out_features
    module.weight = np.zeros((out_features, in_features), dtype=np.float32)
    module.bias = np.zeros((out_features,), dtype=np.float32)
    return module


def _block(hidden_width):
    EdgeNextConvEncoder = type("EdgeNextConvEncoder", (), {})
    Identity = type("Identity", (), {})
    GELU = type("GELU", (), {})
    block = EdgeNextConvEncoder()
    block.dwconv = _conv(36, 36, (7, 7), (3, 3), groups=36)
    block.norm = Identity()
    block.pwconv1 = _linear(36, hidden_width)
    block.act = GELU()
    block.act.approximate = "none"
    block.pwconv2 = _linear(hidden_width, 36)
    block.gamma = np.zeros((36,), dtype=np.float32)
    return block


def _disp_head():
    Sequential = type(
        "Sequential",
        (),
        {"__iter__": lambda self: iter(self.layers), "__len__": lambda self: len(self.layers)},
    )
    ReLU = type("ReLU", (), {})
    module = Sequential()
    module.layers = (
        _conv(60, 36, (3, 3), (1, 1)),
        ReLU(),
        _block(212),
        _block(244),
        _conv(36, 1, (3, 3), (1, 1)),
    )
    return module


def test_disp_head_nchw_gate_defaults_on_and_requires_exact_one(monkeypatch) -> None:
    monkeypatch.delenv(_ENV, raising=False)
    assert native_post._disp_head_nchw_pointwise_enabled()
    for value in ("0", "true"):
        monkeypatch.setenv(_ENV, value)
        assert not native_post._disp_head_nchw_pointwise_enabled()
    monkeypatch.setenv(_ENV, "1")
    assert native_post._disp_head_nchw_pointwise_enabled()


def test_disp_head_gamma_fold_gate_defaults_on_and_requires_exact_one(monkeypatch) -> None:
    monkeypatch.delenv(_FOLD_ENV, raising=False)
    assert native_post._disp_head_fold_gamma_enabled()
    for value in ("0", "true"):
        monkeypatch.setenv(_FOLD_ENV, value)
        assert not native_post._disp_head_fold_gamma_enabled()
    monkeypatch.setenv(_FOLD_ENV, "1")
    assert native_post._disp_head_fold_gamma_enabled()


def test_disp_head_second_gelu_tanh_gate_defaults_on_and_requires_exact_one(monkeypatch) -> None:
    monkeypatch.delenv(_TANH_ENV, raising=False)
    assert native_post._disp_head_second_gelu_tanh_enabled()
    for value in ("0", "true"):
        monkeypatch.setenv(_TANH_ENV, value)
        assert not native_post._disp_head_second_gelu_tanh_enabled()
    monkeypatch.setenv(_TANH_ENV, "1")
    assert native_post._disp_head_second_gelu_tanh_enabled()


def test_disp_head_gamma_fold_requires_nchw_gate_before_graph_layers(monkeypatch) -> None:
    monkeypatch.setenv(_FOLD_ENV, "1")
    monkeypatch.setenv(_ENV, "0")
    with pytest.raises(RuntimeError, match="requires.*DISP_HEAD_NCHW_POINTWISE=1"):
        native_post.add_post_graph(
            object(),
            object(),
            {},
            max_disparity=192,
            valid_iters=8,
        )


def test_disp_head_second_gelu_tanh_requires_nchw_gate_before_graph_layers(monkeypatch) -> None:
    monkeypatch.setenv(_FOLD_ENV, "0")
    monkeypatch.setenv(_TANH_ENV, "1")
    monkeypatch.setenv(_ENV, "0")
    with pytest.raises(RuntimeError, match="requires.*DISP_HEAD_NCHW_POINTWISE=1"):
        native_post.add_post_graph(
            object(),
            object(),
            {},
            max_disparity=192,
            valid_iters=8,
        )


def test_disp_head_nchw_routes_only_two_scoped_encoders() -> None:
    module = _disp_head()

    class _Graph:
        def __init__(self):
            self.trt = SimpleNamespace(float16="fp16", float32="fp32")
            self.work_trt_dtype = "fp16"
            self.events = []

        def sequential(self, tensor, target):
            self.events.append(("sequential", tensor, target))
            return "legacy"

        def module(self, tensor, layer):
            self.events.append(("module", layer.__class__.__name__))
            if layer is module.layers[0] or layer is module.layers[1]:
                return _Tensor((1, 36, 176, 176), "fp16")
            return _Tensor((1, 1, 176, 176), "fp16")

        def edge_next_encoder(
            self,
            tensor,
            block,
            *,
            nchw_pointwise,
            fold_gamma=False,
            gelu_approximate="none",
        ):
            self.events.append(
                (
                    "edge",
                    block.pwconv1.out_features,
                    nchw_pointwise,
                    fold_gamma,
                    gelu_approximate,
                )
            )
            return _Tensor((1, 36, 176, 176), "fp32")

    graph = _Graph()
    hidden = _Tensor((1, 60, 176, 176), "fp16")
    assert native_post._disp_head_delta(graph, hidden, module, nchw_pointwise=False) == "legacy"
    assert graph.events == [("sequential", hidden, module)]

    graph.events.clear()
    output = native_post._disp_head_delta(graph, hidden, module, nchw_pointwise=True)
    assert output.shape == (1, 1, 176, 176)
    assert graph.events == [
        ("module", "Conv2d"),
        ("module", "ReLU"),
        ("edge", 212, True, False, "none"),
        ("edge", 244, True, False, "none"),
        ("module", "Conv2d"),
    ]

    graph.events.clear()
    output = native_post._disp_head_delta(
        graph,
        hidden,
        module,
        nchw_pointwise=True,
        second_gelu_tanh=True,
    )
    assert output.shape == (1, 1, 176, 176)
    assert graph.events == [
        ("module", "Conv2d"),
        ("module", "ReLU"),
        ("edge", 212, True, False, "none"),
        ("edge", 244, True, False, "tanh"),
        ("module", "Conv2d"),
    ]


def test_disp_head_gamma_fold_routes_exactly_sixteen_static_block_calls() -> None:
    module = _disp_head()

    class _Graph:
        def __init__(self):
            self.trt = SimpleNamespace(float16="fp16", float32="fp32")
            self.work_trt_dtype = "fp16"
            self.folded_widths = []

        def module(self, tensor, layer):
            if layer is module.layers[0] or layer is module.layers[1]:
                return _Tensor((1, 36, 176, 176), "fp16")
            return _Tensor((1, 1, 176, 176), "fp16")

        def edge_next_encoder(
            self,
            tensor,
            block,
            *,
            nchw_pointwise,
            fold_gamma=False,
            gelu_approximate="none",
        ):
            assert nchw_pointwise
            assert fold_gamma
            assert gelu_approximate == "none"
            self.folded_widths.append(block.pwconv1.out_features)
            return _Tensor((1, 36, 176, 176), "fp32")

    graph = _Graph()
    hidden = _Tensor((1, 60, 176, 176), "fp16")
    for _ in range(8):
        native_post._disp_head_delta(
            graph,
            hidden,
            module,
            nchw_pointwise=True,
            fold_gamma=True,
        )
    assert graph.folded_widths == [212, 244] * 8


def test_disp_head_nchw_rejects_topology_drift_before_layers() -> None:
    module = _disp_head()
    module.layers[2].norm = type("LayerNorm", (), {})()
    graph = SimpleNamespace(events=[])
    with pytest.raises(RuntimeError, match="Identity normalization"):
        native_post._disp_head_delta(
            graph,
            _Tensor((1, 60, 176, 176), "fp16"),
            module,
            nchw_pointwise=True,
        )
    assert graph.events == []


def test_edge_next_encoder_nchw_keeps_fp16_pointwise_and_fp32_residual() -> None:
    block = _block(212)
    events = []
    graph = object.__new__(NativeGraph)
    graph.trt = SimpleNamespace(float16="fp16", float32="fp32")
    graph.work_trt_dtype = "fp16"

    def cast(tensor, dtype):
        events.append(("cast", dtype))
        return _Tensor(tensor.shape, dtype)

    graph.cast = cast
    graph.conv2d = lambda tensor, module: (
        events.append(("depthwise", module.groups)) or _Tensor((1, 36, 176, 176), "fp16")
    )
    graph.linear_as_conv2d = lambda tensor, module: (
        events.append(("pointwise", module.out_features))
        or _Tensor((1, module.out_features, 176, 176), "fp16")
    )
    graph.gelu = lambda tensor: events.append(("gelu",)) or tensor

    def constant(values, shape, *, dtype=None, target_dtype=None):
        events.append(("gamma", values.shape, shape, dtype, target_dtype))
        return _Tensor(shape, target_dtype)

    graph.constant = constant
    graph.mul = lambda lhs, rhs: events.append(("mul",)) or lhs
    graph.add = lambda lhs, rhs: events.append(("add",)) or lhs
    graph._array = NativeGraph._array
    graph.transpose = lambda *_args, **_kwargs: pytest.fail("NCHW path must not transpose")
    graph.linear = lambda *_args, **_kwargs: pytest.fail("NCHW path must not use MatMul Linear")

    output = NativeGraph.edge_next_encoder(
        graph,
        _Tensor((1, 36, 176, 176), "fp16"),
        block,
        nchw_pointwise=True,
    )
    assert output.dtype == "fp32"
    assert ("pointwise", 212) in events
    assert ("pointwise", 36) in events
    assert ("gamma", (1, 36, 1, 1), (1, 36, 1, 1), np.float32, "fp32") in events


def test_native_gelu_defaults_exact_and_uses_native_tanh_only_when_requested() -> None:
    events = []

    class _Layer:
        @staticmethod
        def get_output(index):
            assert index == 0
            return "output"

    graph = object.__new__(NativeGraph)
    graph.trt = SimpleNamespace(
        ActivationType=SimpleNamespace(GELU_ERF="gelu-erf", GELU_TANH="gelu-tanh")
    )
    graph.network = SimpleNamespace(
        add_activation=lambda tensor, activation: events.append((tensor, activation)) or _Layer()
    )

    assert NativeGraph.gelu(graph, "input") == "output"
    assert NativeGraph.gelu(graph, "input", approximate="tanh") == "output"
    assert events == [("input", "gelu-erf"), ("input", "gelu-tanh")]
    with pytest.raises(ValueError, match="must be 'none' or 'tanh'"):
        NativeGraph.gelu(graph, "input", approximate="fast")
    assert len(events) == 2


def test_native_gelu_tanh_rejects_missing_tensorrt_api_before_layers() -> None:
    graph = object.__new__(NativeGraph)
    graph.trt = SimpleNamespace(ActivationType=SimpleNamespace(GELU_ERF="gelu-erf"))
    graph.network = SimpleNamespace(
        add_activation=lambda *_args: pytest.fail("missing GELU_TANH must add no layers")
    )
    with pytest.raises(RuntimeError, match="does not expose native GELU_TANH"):
        NativeGraph.gelu(graph, "input", approximate="tanh")


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("first_block_width", "second validated"),
        ("non_identity_norm", "second validated"),
        ("wrong_shape", "second validated"),
        ("non_nchw", "second validated"),
        ("non_fp16", "second validated"),
    ),
)
def test_edge_next_encoder_gelu_tanh_rejects_out_of_scope_before_layers(
    mutation: str,
    match: str,
) -> None:
    block = _block(244)
    tensor = _Tensor((1, 36, 176, 176), "fp16")
    nchw_pointwise = True
    graph = object.__new__(NativeGraph)
    graph.trt = SimpleNamespace(float16="fp16", float32="fp32")
    graph.work_trt_dtype = "fp16"
    if mutation == "first_block_width":
        block = _block(212)
    elif mutation == "non_identity_norm":
        block.norm = type("LayerNorm", (), {})()
    elif mutation == "wrong_shape":
        tensor = _Tensor((1, 36, 88, 88), "fp16")
    elif mutation == "non_nchw":
        nchw_pointwise = False
    else:
        graph.work_trt_dtype = "fp32"
    graph.cast = lambda *_args, **_kwargs: pytest.fail("invalid scope must add no layers")
    graph.conv2d = lambda *_args, **_kwargs: pytest.fail("invalid scope must add no layers")

    with pytest.raises(RuntimeError, match=match):
        NativeGraph.edge_next_encoder(
            graph,
            tensor,
            block,
            nchw_pointwise=nchw_pointwise,
            gelu_approximate="tanh",
        )


def test_edge_next_encoder_gamma_fold_quantizes_before_fp32_channel_scale() -> None:
    block = _block(212)
    generator = np.random.default_rng(170817)
    block.pwconv2.weight = generator.normal(
        0.0,
        0.07,
        size=block.pwconv2.weight.shape,
    ).astype(np.float32)
    block.pwconv2.bias = generator.normal(0.0, 0.07, size=(36,)).astype(np.float32)
    block.gamma = np.linspace(0.173, 1.317, 36, dtype=np.float32)
    original_weight = block.pwconv2.weight.copy()
    original_bias = block.pwconv2.bias.copy()
    original_gamma = block.gamma.copy()
    pointwise_modules = []

    graph = object.__new__(NativeGraph)
    graph.trt = SimpleNamespace(float16="fp16", float32="fp32")
    graph.work_trt_dtype = "fp16"
    graph._array = NativeGraph._array
    graph.cast = lambda tensor, dtype: _Tensor(tensor.shape, dtype)
    graph.conv2d = lambda _tensor, _module: _Tensor((1, 36, 176, 176), "fp16")

    def linear_as_conv2d(_tensor, module):
        pointwise_modules.append(module)
        return _Tensor((1, module.out_features, 176, 176), "fp16")

    graph.linear_as_conv2d = linear_as_conv2d
    graph.gelu = lambda tensor: tensor
    graph.constant = lambda *_args, **_kwargs: pytest.fail("folded gamma must not add a constant")
    graph.mul = lambda *_args, **_kwargs: pytest.fail("folded gamma must not add a multiply")
    graph.add = lambda _lhs, rhs: rhs

    output = NativeGraph.edge_next_encoder(
        graph,
        _Tensor((1, 36, 176, 176), "fp16"),
        block,
        nchw_pointwise=True,
        fold_gamma=True,
    )

    assert output.dtype == "fp32"
    assert pointwise_modules[0] is block.pwconv1
    folded = pointwise_modules[1]
    assert folded is not block.pwconv2
    expected_weight = (
        original_weight.astype(np.float16).astype(np.float32) * original_gamma[:, None]
    ).astype(np.float16)
    expected_bias = (original_bias.astype(np.float16).astype(np.float32) * original_gamma).astype(
        np.float16
    )
    np.testing.assert_array_equal(folded.weight, expected_weight)
    np.testing.assert_array_equal(folded.bias, expected_bias)
    assert folded.weight.dtype == np.float16
    assert folded.bias.dtype == np.float16
    assert folded.weight.flags.c_contiguous
    assert folded.bias.flags.c_contiguous

    direct_weight_fold = (original_weight * original_gamma[:, None]).astype(np.float16)
    direct_bias_fold = (original_bias * original_gamma).astype(np.float16)
    assert np.any(folded.weight != direct_weight_fold)
    assert np.any(folded.bias != direct_bias_fold)
    np.testing.assert_array_equal(block.pwconv2.weight, original_weight)
    np.testing.assert_array_equal(block.pwconv2.bias, original_bias)
    np.testing.assert_array_equal(block.gamma, original_gamma)


@pytest.mark.parametrize(
    ("mutation", "nchw_pointwise", "match"),
    (
        ("missing_gamma", True, "requires a gamma"),
        ("weight_shape", True, "shape drift"),
        ("non_nchw", False, "requires NCHW"),
    ),
)
def test_edge_next_encoder_gamma_fold_rejects_invalid_scope_before_layers(
    mutation,
    nchw_pointwise,
    match,
) -> None:
    block = _block(212)
    if mutation == "missing_gamma":
        block.gamma = None
    elif mutation == "weight_shape":
        block.pwconv2.weight = np.zeros((35, 212), dtype=np.float32)

    graph = object.__new__(NativeGraph)
    graph.trt = SimpleNamespace(float16="fp16", float32="fp32")
    graph.work_trt_dtype = "fp16"
    graph._array = NativeGraph._array
    graph.cast = lambda *_args, **_kwargs: pytest.fail("invalid folding must add no layers")
    graph.conv2d = lambda *_args, **_kwargs: pytest.fail("invalid folding must add no layers")

    with pytest.raises(RuntimeError, match=match):
        NativeGraph.edge_next_encoder(
            graph,
            _Tensor((1, 36, 176, 176), "fp16"),
            block,
            nchw_pointwise=nchw_pointwise,
            fold_gamma=True,
        )


def test_disp_head_nchw_is_native_only() -> None:
    family = Path(native_post.__file__).resolve().parent
    source = (
        (family / "native_post.py").read_text(encoding="utf-8")
        + (family / "native_graph.py").read_text(encoding="utf-8")
    ).lower()
    assert "disp_head_nchw_pointwise" in source
    assert "disp_head_fold_gamma" in source
    assert "onnx" not in source
