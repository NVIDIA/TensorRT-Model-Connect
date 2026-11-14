# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import math
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

from families.modernbert.config import ModelConfig, resolve_attention_contract


class _Tensor:
    def __init__(self, values, dtype):
        storage = {
            "fp32": np.float32,
            "fp16": np.float16,
            "bf16": ml_dtypes.bfloat16,
            "int32": np.int32,
            "bool": np.bool_,
        }
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            self.values = np.asarray(values).astype(storage[dtype]).astype(np.float32)
        self.dtype = dtype
        self.shape = self.values.shape


def _layer(tensor):
    return SimpleNamespace(get_output=lambda index: tensor)


class _Network:
    def __init__(self, sequence_length):
        self.sequence_length = sequence_length
        self.ropes = []
        self.attention_masks = []
        self.window_selections = 0
        self.constants = []
        self.gathers = []
        self.output = None

    def add_input(self, name, dtype, shape):
        values = (
            np.arange(self.sequence_length) % 4
            if name == "input_ids"
            else np.asarray(
                [1] * (self.sequence_length - 1) + [0] if self.sequence_length > 1 else [1]
            )
        )
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
        function = {"prod": np.multiply, "sum": np.add, "sub": np.subtract, "less": np.less}[
            operation
        ]
        return _layer(
            _Tensor(function(lhs.values, rhs.values), "bool" if operation == "less" else lhs.dtype)
        )

    def add_gather(self, table, indices, axis):
        self.gathers.append(table)
        return _layer(_Tensor(np.take(table.values, indices.values.astype(int), axis), table.dtype))

    def add_shuffle(self, tensor):
        layer = SimpleNamespace(
            reshape_dims=tensor.shape, first_transpose=None, second_transpose=None
        )

        def output(index):
            values = tensor.values
            if layer.first_transpose is not None:
                values = values.transpose(layer.first_transpose)
            values = values.reshape(layer.reshape_dims)
            if layer.second_transpose is not None:
                values = values.transpose(layer.second_transpose)
            return _Tensor(values, tensor.dtype)

        layer.get_output = output
        return layer

    def add_shape(self, tensor):
        return _layer(_Tensor(tensor.shape, "int32"))

    def add_slice(self, tensor, start, shape, stride):
        layer = SimpleNamespace(length=shape[0])
        layer.set_input = lambda index, value: setattr(layer, "length", int(value.values[0]))
        layer.get_output = lambda index: _Tensor(
            tensor.values[start[0] : start[0] + layer.length : stride[0]], tensor.dtype
        )
        return layer

    def add_rotary_embedding(self, tensor, cos, sin, interleaved, rotary_dim):
        assert tensor.dtype == cos.dtype == sin.dtype, "mixed RoPE dtypes"
        self.ropes.append((tensor, cos, sin))
        layer = _layer(tensor)
        layer.set_input = lambda index, value: None
        return layer

    def add_select(self, condition, lhs, rhs):
        self.window_selections += 1
        assert condition.dtype == "bool" and lhs.dtype == rhs.dtype
        return _layer(
            _Tensor(np.where(condition.values.astype(bool), lhs.values, rhs.values), lhs.dtype)
        )

    def add_dist_collective(self, tensor, *args):
        return _layer(tensor)

    def add_attention(self, q, k, v, normalization, causal):
        assert normalization == "softmax" and not causal
        layer = SimpleNamespace(mask=None)

        def output(index):
            scores = q.values @ k.values.swapaxes(-1, -2)
            if layer.mask is not None:
                assert q.dtype == k.dtype == v.dtype == layer.mask.dtype
                scores += layer.mask.values
            probabilities = np.exp(scores - scores.max(-1, keepdims=True))
            probabilities /= probabilities.sum(-1, keepdims=True)
            return _Tensor(probabilities @ v.values, q.dtype)

        layer.get_output = output
        return layer

    def add_unary(self, tensor, operation):
        if operation == "abs":
            return _layer(_Tensor(np.abs(tensor.values), tensor.dtype))
        assert operation == "erf"
        return _layer(_Tensor(np.vectorize(math.erf)(tensor.values), tensor.dtype))

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

    def mark_output(self, tensor):
        self.output = tensor


def _source_functions(path, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
        *[node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))],
    ]
    ast.fix_missing_locations(tree)
    exec(compile(tree, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.fixture
def graph():
    def create(sequence_length, native_attention=False):
        network = _Network(sequence_length)
        logger = type(
            "Logger", (), {"VERBOSE": 1, "WARNING": 2, "__init__": lambda self, level: None}
        )
        profile = SimpleNamespace(set_shape=lambda *args: None)
        builder = SimpleNamespace(
            create_network=lambda flags: network,
            create_builder_config=lambda: SimpleNamespace(
                clear_flag=lambda flag: None, add_optimization_profile=lambda profile: None
            ),
            create_optimization_profile=lambda: profile,
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
            Permutation=tuple,
            NetworkDefinitionCreationFlag=SimpleNamespace(STRONGLY_TYPED=1),
            BuilderFlag=SimpleNamespace(TF32=1),
            ElementWiseOperation=SimpleNamespace(PROD="prod", SUM="sum", SUB="sub", LESS="less"),
            MatrixOperation=SimpleNamespace(NONE="none"),
            CollectiveOperation=SimpleNamespace(ALL_REDUCE="allreduce"),
            ReduceOperation=SimpleNamespace(SUM="sum"),
            UnaryOperation=SimpleNamespace(ERF="erf", ABS="abs"),
            AttentionNormalizationOp=SimpleNamespace(SOFTMAX="softmax"),
        )
        owner = Path(__file__).resolve().parents[1]
        ops = _source_functions(owner / "graph/model.py", {"np": np, "trt": trt})
        parallel = _source_functions(
            owner / "parallel.py",
            {
                "np": np,
                "dataclass": dataclass,
                "replace": replace,
                "__name__": __name__,
                "trt": trt,
                "sys": sys,
                "graph_ops": ops,
                "ModelConfig": ModelConfig,
                "resolve_attention_contract": resolve_attention_contract,
            },
        )

        def attention(net, q, k, v, **kwargs):
            network.attention_masks.append(kwargs["mask"].values.copy())
            assert q.dtype == k.dtype == v.dtype == kwargs["mask"].dtype
            return q

        if not native_attention:
            ops.add_attention_from_rows = attention
        encoder = _source_functions(
            owner / "model.py",
            {
                "np": np,
                "trt": trt,
                "sys": sys,
                "Path": Path,
                "graph_ops": ops,
                "ModelConfig": ModelConfig,
                "resolve_attention_contract": resolve_attention_contract,
                "normalize_parallel_config": parallel.normalize_parallel_config,
                "ParallelConfig": parallel.ParallelConfig,
                "__package__": "families.modernbert",
            },
        )
        return network, encoder, parallel, ops

    return create


def _weights(values):
    weights = {
        "embedding": np.tile(values, (4, 1)).astype(np.float32),
        "embed_norm": np.ones(4, np.float32),
        "final_norm": np.ones(4, np.float32),
    }
    for index in range(4):
        prefix = f"layer.{index}."
        for name in ("w_q", "w_k", "w_v", "w_o", "w_mlp_input", "w_mlp_gate", "w_down"):
            weights[prefix + name] = np.eye(4, dtype=np.float32)
        weights[prefix + "mlp_norm"] = np.ones(4, np.float32)
        if index:
            weights[prefix + "attn_norm"] = np.ones(4, np.float32)
    return weights


@pytest.mark.parametrize("mode", ["fp32", "fp16", "tp2"])
@pytest.mark.parametrize("sequence_length", [1, 2, 7])
@pytest.mark.parametrize("layout", ["legacy", "materialized", "full_only"])
def test_local_layers_apply_bidirectional_window_and_padding(graph, mode, sequence_length, layout):
    network, encoder, parallel, _ = graph(sequence_length)
    raw = {
        "model_type": "modernbert",
        "hidden_size": 4,
        "vocab_size": 4,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "intermediate_size": 4,
        "global_attn_every_n_layers": 2,
        "local_attention": 4,
    }
    if layout == "materialized":
        raw.update(
            layer_types=[
                "full_attention",
                "sliding_attention",
                "global_attention",
                "local_attention",
            ],
            sliding_window=2,
            local_attention=10,
        )
    elif layout == "full_only":
        raw["layer_types"] = ["full_attention"] * 4
    config = ModelConfig.from_json(json.dumps(raw))
    weights = _weights([1, -1, 0.5, -0.5])
    if mode == "tp2":
        parallel.build_tp_modernbert_engine(
            config,
            weights,
            sequence_length,
            parallel_config=parallel.ParallelConfig(tp_size=2, rank=0),
        )
    else:
        encoder._ModernBertModel().build_engine(config, weights, 8, precision=mode)

    assert len(network.attention_masks) == 4
    assert network.window_selections == (0 if layout == "full_only" else 1)
    q_positions = np.arange(sequence_length)[:, None]
    k_positions = np.arange(sequence_length)[None, :]
    padding = np.arange(sequence_length) < (sequence_length - 1 if sequence_length > 1 else 1)
    local_allowed = (np.abs(q_positions - k_positions) <= 2) & padding
    for layer, mask in enumerate(network.attention_masks):
        actual = np.broadcast_to(mask, (1, 1, sequence_length, sequence_length))[0, 0] == 0
        expected = (
            local_allowed
            if layer % 2 and layout != "full_only"
            else np.broadcast_to(padding, actual.shape)
        )
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("window", [0, 2])
def test_window_mask_dtype_and_inclusive_boundary(graph, dtype, window):
    network, _, _, ops = graph(7)
    positions = _Tensor(np.arange(7), "int32")
    padding = _Tensor([[[[0, 0, 0, 0, 0, 0, -10000]]]], dtype)
    mask = ops.add_bidirectional_window_mask(network, positions, padding, window)
    assert mask.dtype == dtype
    allowed = mask.values[0, 0] == 0
    expected = np.array([[abs(q - k) <= window and k != 6 for k in range(7)] for q in range(7)])
    np.testing.assert_array_equal(allowed, expected)


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("sequence_length", [None, 7])
def test_attention_values_exclude_distant_and_padded_tokens(graph, dtype, sequence_length):
    network, _, _, ops = graph(7, native_attention=True)
    q = k = _Tensor(np.ones((7, 4)), dtype)
    values = np.arange(28, dtype=np.float32).reshape(7, 4)
    padding = _Tensor([[[[0, 0, 0, 0, 0, 0, -10000]]]], dtype)
    local_mask = ops.add_bidirectional_window_mask(
        network, _Tensor(np.arange(7), "int32"), padding, 2
    )

    def context(v, mask):
        return ops.add_attention_from_rows(
            network,
            q,
            k,
            _Tensor(v, dtype),
            num_heads=2,
            head_dim=2,
            q_seq=sequence_length,
            kv_seq=sequence_length,
            mask=mask,
        ).values

    local = context(values, local_mask)
    full = context(values, padding)
    for query in range(7):
        neighbors = [key for key in range(6) if abs(query - key) <= 2]
        np.testing.assert_allclose(local[query], values[neighbors].mean(0), atol=0.02)
        np.testing.assert_allclose(full[query], values[:6].mean(0), atol=0.02)
    distant = values.copy()
    distant[5] += 12
    np.testing.assert_array_equal(context(distant, local_mask)[0], local[0])
    np.testing.assert_allclose(context(distant, padding)[0], full[0] + 2, atol=0.02)
    assert not np.array_equal(context(distant, local_mask)[4], local[4])
    padded = values.copy()
    padded[6] += 100
    np.testing.assert_array_equal(context(padded, local_mask), local)
    np.testing.assert_array_equal(context(padded, padding), full)
