# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest


class _Tensor:
    def __init__(self, values, dtype):
        storage = {
            "fp32": np.float32,
            "fp16": np.float16,
            "bf16": ml_dtypes.bfloat16,
            "int32": np.int32,
        }
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            self.values = np.asarray(values).astype(storage[dtype]).astype(np.float32)
        self.dtype = dtype
        self.shape = self.values.shape


def _layer(tensor):
    return SimpleNamespace(get_output=lambda index: tensor)


class _Network:
    def __init__(self):
        self.constants = []
        self.gathers = []
        self.output = None

    def add_input(self, name, dtype, shape):
        values = np.arange(shape[0]) if name == "input_ids" else np.ones(shape)
        return _Tensor(values, dtype)

    def add_constant(self, shape, weights):
        values = weights.reshape(shape)
        self.constants.append(values.copy())
        return _layer(
            _Tensor(
                values,
                "fp16"
                if values.dtype == np.float16
                else "int32"
                if values.dtype == np.int32
                else "fp32",
            )
        )

    def add_cast(self, tensor, dtype):
        return _layer(_Tensor(tensor.values, dtype))

    def add_elementwise(self, lhs, rhs, operation):
        assert lhs.dtype == rhs.dtype, "mixed elementwise dtypes"
        function = {"prod": np.multiply, "sum": np.add, "sub": np.subtract}[operation]
        return _layer(_Tensor(function(lhs.values, rhs.values), lhs.dtype))

    def add_gather(self, table, indices, axis):
        self.gathers.append(table)
        return _layer(_Tensor(np.take(table.values, indices.values.astype(int), axis), table.dtype))

    def add_shuffle(self, tensor):
        layer = SimpleNamespace(reshape_dims=tensor.shape)
        layer.get_output = lambda index: _Tensor(
            tensor.values.reshape(layer.reshape_dims), tensor.dtype
        )
        return layer

    def add_matrix_multiply(self, lhs, lhs_op, rhs, rhs_op):
        assert lhs.dtype == rhs.dtype, "mixed matmul dtypes"
        return _layer(_Tensor(lhs.values @ rhs.values, lhs.dtype))

    def add_normalization_v2(self, tensor, gamma, beta, axes):
        assert tensor.dtype == gamma.dtype == beta.dtype, "mixed normalization dtypes"
        layer = SimpleNamespace(epsilon=1e-5)

        def output(index):
            values = tensor.values
            normalized = (values - values.mean(-1, keepdims=True)) / np.sqrt(
                values.var(-1, keepdims=True) + layer.epsilon
            )
            return _Tensor(normalized * gamma.values + beta.values, tensor.dtype)

        layer.get_output = output
        return layer

    def add_activation(self, tensor, operation):
        assert operation == "tanh"
        return _layer(_Tensor(np.tanh(tensor.values), tensor.dtype))

    def mark_output(self, tensor):
        self.output = tensor


def _source_functions(path, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *[node for node in tree.body if isinstance(node, ast.FunctionDef)],
    ]
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture
def graph():
    network = _Network()
    logger = type("Logger", (), {"VERBOSE": 1, "WARNING": 2, "__init__": lambda self, level: None})
    builder = SimpleNamespace(
        create_network=lambda flags: network,
        create_builder_config=lambda: SimpleNamespace(clear_flag=lambda flag: None),
        build_serialized_network=lambda net, config: b"typed-graph",
    )
    trt = SimpleNamespace(
        float16="fp16",
        float32="fp32",
        bfloat16="bf16",
        int32="int32",
        Weights=lambda values: values,
        Logger=logger,
        Builder=lambda logger: builder,
        NetworkDefinitionCreationFlag=SimpleNamespace(STRONGLY_TYPED=1),
        BuilderFlag=SimpleNamespace(TF32=1),
        ElementWiseOperation=SimpleNamespace(PROD="prod", SUM="sum", SUB="sub"),
        MatrixOperation=SimpleNamespace(NONE="none"),
        ActivationType=SimpleNamespace(TANH="tanh"),
    )
    owner = Path(__file__).resolve().parents[1]
    ops = _source_functions(owner / "graph_ops.py", {"np": np, "trt": trt})
    ops.add_attention_from_rows = lambda net, q, k, v, **kwargs: q
    encoder = _source_functions(
        owner / "encoder_builder.py", {"np": np, "trt": trt, "sys": sys, "graph_ops": ops}
    )
    return network, encoder


def _weights(values):
    matrix = np.tile(values, (4, 1)).astype(np.float32)
    weights = {
        "embedding": matrix,
        "position_embedding": np.zeros((4, 4), np.float32),
        "token_type_embedding": np.zeros((1, 4), np.float32),
        "embed_norm": np.ones(4, np.float32),
        "embed_norm_beta": np.zeros(4, np.float32),
    }
    for index in range(2):
        prefix = f"layer.{index}."
        for name in ("w_q", "w_k", "w_v", "w_o", "w_fc1", "w_fc2"):
            weights[prefix + name] = np.eye(4, dtype=np.float32)
        for name in (
            "q_bias",
            "k_bias",
            "v_bias",
            "o_bias",
            "fc1_bias",
            "fc2_bias",
            "post_attn_norm_beta",
            "output_norm_beta",
        ):
            weights[prefix + name] = np.zeros(4, np.float32)
        for name in ("post_attn_norm", "output_norm"):
            weights[prefix + name] = np.ones(4, np.float32)
    return weights


@pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("fp32_layers", [[], [1]])
def test_encoder_constants_match_requested_graph_precision(graph, precision, fp32_layers):
    network, encoder = graph
    config = SimpleNamespace(
        hidden_size=4,
        num_hidden_layers=2,
        num_attention_heads=1,
        intermediate_size=4,
        rms_norm_eps=1e-5,
        hidden_act="gelu_new",
        raw={"type_vocab_size": 1, "_fp32_layers": fp32_layers},
    )
    values = [2**20, -(2**20), 2**-30, -(2**-30)] if precision != "fp16" else [1, -1, 0.5, -0.5]
    weights = _weights(values)
    assert encoder.build_encoder_engine(config, weights, 4, precision=precision) == b"typed-graph"
    table = network.gathers[0]
    assert table.dtype == precision
    expected = np.asarray(
        values,
        dtype={"fp32": np.float32, "fp16": np.float16, "bf16": ml_dtypes.bfloat16}[precision],
    ).astype(np.float32)
    np.testing.assert_array_equal(table.values, np.tile(expected, (4, 1)))
    assert np.isfinite(network.output.values).all()
    assert network.output.dtype == "fp32"
    assert network.output.shape == (4, 4)
    if precision == "bf16":
        assert not any(values.dtype == np.float16 for values in network.constants)
