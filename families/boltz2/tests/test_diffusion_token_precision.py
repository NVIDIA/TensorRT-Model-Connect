# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precision of the diffusion token-transformer projections (bf16 default, fp32 upstream)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from families.boltz2 import cli
from families.boltz2.diffusion_token_builder import (
    TOKEN_CHANNELS,
    define_diffusion_token_network,
)
from families.boltz2.graph_ops import Graph
from tensorrt_model_connect import BuildRequest as LegacyRequest
from tensorrt_model_connect import family_cli


# Projections per token layer: adaln.s_scale/s_bias, proj_q/k/v/g/o, output_projection.0,
# transition.adaln.s_scale/s_bias, swish_gate.0, a_to_b, b_to_a, output_projection.0.
PROJECTIONS_PER_LAYER = 14


class _Tensor:
    def __init__(self, shape, dtype, layer=None):
        self._shape = None if shape is None else tuple(int(dim) for dim in shape)
        self.dtype = dtype
        self.layer = layer
        self.name = ""

    @property
    def shape(self):
        return self._shape if self._shape is not None else self.layer.output_shape()


class _Layer:
    def __init__(self, kind, inputs, shape=None, dtype=None, **attributes):
        self.kind = kind
        self.inputs = inputs
        self._shape = shape
        self.__dict__.update(attributes)
        self.output = _Tensor(shape, dtype if dtype is not None else inputs[0].dtype, self)

    def output_shape(self):
        source = self.inputs[0].shape
        if getattr(self, "reshape_dims", None) is not None:
            return tuple(self.reshape_dims)
        if getattr(self, "first_transpose", None) is not None:
            return tuple(source[axis] for axis in self.first_transpose)
        return source

    def get_output(self, _index):
        return self.output


def _broadcast(lhs, rhs):
    return tuple(max(a, b) for a, b in zip(lhs, rhs, strict=True))


class _Network:
    """Strongly typed stub: every binary op requires equal input dtypes."""

    def __init__(self, trt):
        self.trt = trt
        self.layers = []
        self.outputs = []

    def _add(self, *args, **kwargs):
        layer = _Layer(*args, **kwargs)
        self.layers.append(layer)
        return layer

    def add_input(self, name, dtype, shape):
        tensor = _Tensor(shape, dtype)
        tensor.name = name
        return tensor

    def add_constant(self, shape, _weights):
        return self._add("constant", [], shape=shape, dtype=self.trt.float32)

    def add_cast(self, tensor, dtype):
        return self._add("cast", [tensor], shape=tensor.shape, dtype=dtype)

    def add_matrix_multiply(self, lhs, lhs_op, rhs, rhs_op):
        assert lhs.dtype == rhs.dtype, "matmul input dtypes differ"
        assert lhs_op == self.trt.MatrixOperation.NONE
        rows, inner = lhs.shape[-2:]
        rhs_rows, rhs_cols = rhs.shape[-2:]
        if rhs_op == self.trt.MatrixOperation.TRANSPOSE:
            rhs_rows, rhs_cols = rhs_cols, rhs_rows
        assert inner == rhs_rows, (lhs.shape, rhs.shape)
        batch = _broadcast(lhs.shape[:-2], rhs.shape[:-2])
        return self._add("matmul", [lhs, rhs], shape=(*batch, rows, rhs_cols))

    def add_elementwise(self, lhs, rhs, _operation):
        assert lhs.dtype == rhs.dtype, "elementwise input dtypes differ"
        return self._add("elementwise", [lhs, rhs], shape=_broadcast(lhs.shape, rhs.shape))

    def add_shuffle(self, tensor):
        return self._add("shuffle", [tensor], reshape_dims=None, first_transpose=None)

    def add_slice(self, tensor, _start, shape, _stride):
        return self._add("slice", [tensor], shape=shape)

    def add_normalization_v2(self, tensor, gamma, beta, _axes):
        assert tensor.dtype == gamma.dtype == beta.dtype
        return self._add("normalization", [tensor, gamma, beta], shape=tensor.shape)

    def add_activation(self, tensor, _kind):
        return self._add("activation", [tensor], shape=tensor.shape)

    def add_softmax(self, tensor):
        return self._add("softmax", [tensor], shape=tensor.shape)

    def mark_output(self, tensor):
        self.outputs.append(tensor)


def _stub_trt():
    return SimpleNamespace(
        float32="float32",
        bfloat16="bfloat16",
        Weights=lambda array: array,
        MatrixOperation=SimpleNamespace(NONE="none", TRANSPOSE="transpose"),
        ElementWiseOperation=SimpleNamespace(SUM="sum", SUB="sub", PROD="prod", DIV="div"),
        ActivationType=SimpleNamespace(SIGMOID="sigmoid"),
    )


def _token_weights(layer: int) -> dict[str, np.ndarray]:
    prefix = f"structure_module.score_model.token_transformer.layers.{layer}"
    width, hidden = TOKEN_CHANNELS, 2 * TOKEN_CHANNELS
    shapes = {
        "adaln.s_norm.weight": (width,),
        "adaln.s_scale.weight": (width, width),
        "adaln.s_scale.bias": (width,),
        "adaln.s_bias.weight": (width, width),
        **{f"pair_bias_attn.proj_{name}.weight": (width, width) for name in "qkvgo"},
        "pair_bias_attn.proj_q.bias": (width,),
        "output_projection.0.weight": (width, width),
        "output_projection.0.bias": (width,),
        "transition.adaln.s_norm.weight": (width,),
        "transition.adaln.s_scale.weight": (width, width),
        "transition.adaln.s_scale.bias": (width,),
        "transition.adaln.s_bias.weight": (width, width),
        "transition.swish_gate.0.weight": (2 * hidden, width),
        "transition.a_to_b.weight": (hidden, width),
        "transition.b_to_a.weight": (width, hidden),
        "transition.output_projection.0.weight": (width, width),
        "transition.output_projection.0.bias": (width,),
    }
    return {f"{prefix}.{name}": np.zeros(shape, np.float32) for name, shape in shapes.items()}


def _define(projection_precision: str, layers: int = 2, token_count: int = 5):
    trt = _stub_trt()
    network = _Network(trt)
    weights = {}
    for layer in range(layers):
        weights.update(_token_weights(layer))
    output = define_diffusion_token_network(
        network,
        trt,
        weights,
        token_count=token_count,
        first_layer=0,
        layer_count=layers,
        projection_precision=projection_precision,
    )
    return network, output


def _matmuls(network):
    return [layer for layer in network.layers if layer.kind == "matmul"]


def test_fp32_projection_precision_keeps_the_upstream_fp32_graph():
    network, output = _define("fp32")
    assert output.dtype == "float32" and output.shape == (1, 5, TOKEN_CHANNELS)
    assert {layer.output.dtype for layer in network.layers} == {"float32"}
    # Only the fp32 weight / bias / scalar constants are cast (no-op casts are skipped).
    assert not [layer for layer in network.layers if layer.kind == "cast"]
    assert len(_matmuls(network)) == 2 * (PROJECTIONS_PER_LAYER + 2)


def test_bf16_projection_precision_casts_only_the_projections():
    network, output = _define("bf16")
    assert output.dtype == "float32" and output.shape == (1, 5, TOKEN_CHANNELS)
    matmuls = _matmuls(network)
    projections = [layer for layer in matmuls if layer.output.dtype == "bfloat16"]
    attention = [layer for layer in matmuls if layer.output.dtype == "float32"]
    assert len(projections) == 2 * PROJECTIONS_PER_LAYER
    # Attention scores and the probability-weighted sum stay fp32.
    assert len(attention) == 2 * 2
    for layer in projections:
        assert [tensor.dtype for tensor in layer.inputs] == ["bfloat16", "bfloat16"]
    # Norms, softmax, activations and every elementwise op (bias, gates, residual) stay fp32.
    for layer in network.layers:
        if layer.kind in {"normalization", "softmax", "activation", "elementwise"}:
            assert layer.output.dtype == "float32", layer.kind
    bf16_casts = [
        layer
        for layer in network.layers
        if layer.kind == "cast" and layer.output.dtype == "bfloat16"
    ]
    back_casts = [
        layer
        for layer in network.layers
        if layer.kind == "cast" and layer.output.dtype == "float32"
    ]
    assert len(bf16_casts) == 2 * 2 * PROJECTIONS_PER_LAYER  # activation and weight per projection
    assert len(back_casts) == 2 * PROJECTIONS_PER_LAYER
    assert all(layer.inputs[0].layer.kind == "matmul" for layer in back_casts)


def test_unknown_projection_precision_is_rejected():
    with pytest.raises(ValueError, match="projection precision"):
        _define("fp16")


def test_graph_linear_without_dtype_adds_no_cast():
    trt = _stub_trt()
    network = _Network(trt)
    graph = Graph(
        network, trt, {"p.weight": np.ones((3, 4), np.float32), "p.bias": np.ones(3, np.float32)}
    )
    output = graph.linear(network.add_input("x", trt.float32, (1, 2, 4)), "p")
    assert output.dtype == "float32" and output.shape == (1, 2, 3)
    assert [layer.kind for layer in network.layers] == [
        "constant",
        "matmul",
        "constant",
        "elementwise",
    ]


def test_graph_linear_with_bf16_adds_bias_in_the_input_dtype():
    trt = _stub_trt()
    network = _Network(trt)
    graph = Graph(
        network,
        trt,
        {"p.weight": np.ones((3, 4), np.float32), "p.bias": np.ones(3, np.float32)},
        linear_dtype=trt.bfloat16,
    )
    output = graph.linear(network.add_input("x", trt.float32, (1, 2, 4)), "p")
    assert output.dtype == "float32" and output.shape == (1, 2, 3)
    kinds = [(layer.kind, layer.output.dtype) for layer in network.layers]
    assert kinds == [
        ("cast", "bfloat16"),
        ("constant", "float32"),
        ("cast", "bfloat16"),
        ("matmul", "bfloat16"),
        ("cast", "float32"),
        ("constant", "float32"),
        ("elementwise", "float32"),
    ]


def test_build_defaults_to_bf16_token_projections(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli, "build_bundle", lambda request, output: calls.append(request))
    output = str(tmp_path / "model.bundle")
    assert family_cli.main(["boltz2", "build", str(tmp_path), "-o", output]) == 0
    assert (
        family_cli.main(
            ["boltz2", "build", str(tmp_path), "-o", output, "--diffusion-token-precision", "fp32"]
        )
        == 0
    )
    assert [request.diffusion_token_precision for request in calls] == ["bf16", "fp32"]
    with pytest.raises(SystemExit):
        family_cli.main(
            ["boltz2", "build", str(tmp_path), "-o", output, "--diffusion-token-precision", "fp16"]
        )


def test_request_validates_token_precision_and_legacy_request_uses_default(tmp_path):
    assert cli.BuildRequest(tmp_path).diffusion_token_precision == "bf16"
    assert (
        cli.BuildRequest(tmp_path, diffusion_token_precision="fp32").diffusion_token_precision
        == "fp32"
    )
    with pytest.raises(ValueError, match="diffusion_token_precision"):
        cli.BuildRequest(tmp_path, diffusion_token_precision="fp16")
    legacy = LegacyRequest(tmp_path, tmp_path / "out", "boltz2", "structure_prediction", "bf16")
    assert cli.coerce_request(legacy).diffusion_token_precision == "bf16"


def test_model_build_passes_token_precision_to_the_token_plans(monkeypatch, tmp_path):
    model = pytest.importorskip("families.boltz2.model")
    seen = {}

    def extra(self, _config, _weights, _cache, **kwargs):
        seen.update(kwargs)
        return {}

    monkeypatch.setattr(model.Boltz2Plugin, "load_weights", lambda self, *args, **kwargs: {})
    monkeypatch.setattr(model.Boltz2Plugin, "build_engine", lambda self, *args, **kwargs: b"plan")
    monkeypatch.setattr(model.Boltz2Plugin, "build_extra_engines", extra)
    writer = SimpleNamespace(
        set_header=lambda **_: None, add_bytes=lambda *_: None, add_json=lambda *_: None
    )
    for precision in ("fp32", "bf16"):
        model.build(cli.BuildRequest(tmp_path, diffusion_token_precision=precision), writer)
        assert seen["diffusion_token_precision"] == precision
