# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU graph composition checks, without TensorRT execution or BF16 numerics."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


class Tensor:
    def __init__(self, values, dtype="fp32"):
        self.values = np.asarray(values, dtype=np.float32)
        self.dtype = dtype
        self.shape = self.values.shape


class Layer:
    def __init__(self, tensor):
        self.tensor = tensor

    def get_output(self, index):
        assert index == 0
        return self.tensor


class Shuffle:
    def __init__(self, tensor):
        self.tensor = tensor
        self.reshape_dims = tensor.shape
        self.second_transpose = None

    def get_output(self, index):
        assert index == 0
        values = self.tensor.values.reshape(self.reshape_dims)
        if self.second_transpose is not None:
            values = values.transpose(self.second_transpose)
        return Tensor(values, self.tensor.dtype)


class Concatenation:
    def __init__(self, tensors):
        self.tensors = tensors
        self.axis = 0

    def get_output(self, index):
        assert index == 0
        assert len({tensor.dtype for tensor in self.tensors}) == 1, "mixed concatenation dtypes"
        return Tensor(
            np.concatenate([tensor.values for tensor in self.tensors], axis=self.axis),
            self.tensors[0].dtype,
        )


class Network:
    def __init__(self):
        self.casts = []

    def add_constant(self, shape, values):
        dtype = "fp16" if values.dtype == np.float16 else "fp32"
        return Layer(Tensor(values.reshape(shape), dtype))

    def add_cast(self, tensor, dtype):
        self.casts.append((tensor.dtype, dtype))
        return Layer(Tensor(tensor.values, dtype))

    def add_matrix_multiply(self, left, _left_op, right, right_op):
        assert left.dtype == right.dtype, "mixed matrix multiply dtypes"
        values = right.values.swapaxes(-1, -2) if right_op == "transpose" else right.values
        return Layer(Tensor(left.values @ values, left.dtype))

    def add_elementwise(self, left, right, operation):
        assert left.dtype == right.dtype, "mixed elementwise dtypes"
        function = {
            "min": np.minimum,
            "max": np.maximum,
            "prod": np.multiply,
            "sum": np.add,
            "sub": np.subtract,
        }[operation]
        return Layer(Tensor(function(left.values, right.values), left.dtype))

    def add_activation(self, tensor, operation):
        assert operation == "sigmoid"
        return Layer(Tensor(1 / (1 + np.exp(-tensor.values)), tensor.dtype))

    def add_shuffle(self, tensor):
        return Shuffle(tensor)

    def add_concatenation(self, tensors):
        return Concatenation(tensors)

    def add_reduce(self, tensor, operation, axes, keep_dims):
        assert operation == "max" and axes == 4
        return Layer(Tensor(tensor.values.max(axis=2, keepdims=keep_dims), tensor.dtype))

    def add_softmax(self, tensor):
        values = np.exp(tensor.values - tensor.values.max(axis=-1, keepdims=True))
        return Layer(Tensor(values / values.sum(axis=-1, keepdims=True), tensor.dtype))

    def add_slice(self, tensor, start, shape, stride):
        slices = tuple(slice(a, a + n * s, s) for a, n, s in zip(start, shape, stride))
        return Layer(Tensor(tensor.values[slices], tensor.dtype))


@pytest.fixture
def family(monkeypatch):
    trt = SimpleNamespace(
        float16="fp16",
        float32="fp32",
        bfloat16="bf16",
        int32="int32",
        Weights=lambda values: values,
        Permutation=tuple,
        MatrixOperation=SimpleNamespace(NONE="none", TRANSPOSE="transpose"),
        ElementWiseOperation=SimpleNamespace(
            MIN="min", MAX="max", PROD="prod", SUM="sum", SUB="sub"
        ),
        ActivationType=SimpleNamespace(SIGMOID="sigmoid"),
        ReduceOperation=SimpleNamespace(MAX="max"),
    )
    name = "_gpt_oss_constant_test"
    package = ModuleType(name)
    package.__path__ = [str(Path(__file__).resolve().parents[1])]
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    monkeypatch.setitem(sys.modules, name, package)
    try:
        yield importlib.import_module(name + ".model")
    finally:
        for key in tuple(sys.modules):
            if key.startswith(name + "."):
                del sys.modules[key]


@pytest.mark.parametrize(
    "dtype,storage", [("fp32", np.float32), ("fp16", np.float16), ("bf16", np.float16)]
)
def test_expert_constants_match_activation_dtype(family, dtype, storage):
    network = Network()
    inputs = Tensor([[10.0, -10.0], [-2.0, 3.0]], dtype)
    weights = np.eye(2, dtype=storage)
    output = family._add_gpt_oss_expert(
        network, inputs, 2, 2, weights, weights, weights, dtype=storage
    )
    assert output.dtype == dtype
    if dtype == "fp32":
        gate = np.minimum(inputs.values, 7)
        up = np.clip(inputs.values, -7, 7)
        expected = (up + 1) * gate / (1 + np.exp(-gate * np.float32(1.702)))
        np.testing.assert_allclose(output.values, expected, rtol=1e-6, atol=1e-7)
    if dtype == "bf16":
        assert ("fp16", "bf16") in network.casts
    else:
        assert not network.casts


@pytest.mark.parametrize(
    "dtype,storage", [("fp32", np.float32), ("fp16", np.float16), ("bf16", np.float16)]
)
@pytest.mark.parametrize("heads,kv_heads", [(1, 1), (2, 2), (4, 1)])
def test_sink_constant_matches_attention_dtype(
    family, monkeypatch, dtype, storage, heads, kv_heads
):
    monkeypatch.setattr(
        family.graph_ops, "add_apply_rope_native", lambda network, tensor, *args: tensor
    )
    network = Network()
    inputs = Tensor([[0.5, 1.0]], dtype)
    cache = Tensor(np.zeros((1, kv_heads * 2)), dtype)
    weights = {
        "layer.0.w_q": np.tile(np.eye(2, dtype=storage), (1, heads)),
        "layer.0.w_k": np.tile(np.eye(2, dtype=storage), (1, kv_heads)),
        "layer.0.w_v": np.tile(np.eye(2, dtype=storage), (1, kv_heads)),
        "layer.0.w_o": np.tile(np.eye(2, dtype=storage), (heads, 1)) / heads,
        "layer.0.sinks": np.full(heads, 0.25, dtype=storage),
    }
    result = family._add_gpt_oss_attention(
        network,
        inputs,
        cache,
        cache,
        Tensor([[0.0, 0.0]], dtype),
        Tensor([1], "int32"),
        weights=weights,
        prefix="layer.0",
        hidden_size=2,
        attention_size=heads * 2,
        kv_attention_size=kv_heads * 2,
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=2,
        max_cache_length=1,
        cos_half_tensor=None,
        sin_half_tensor=None,
        attn_scale_tensor=Tensor([[[1.0]]], dtype),
        dtype=storage,
    )
    assert all(tensor.dtype == dtype for tensor in result.values())
    if dtype != "bf16":
        assert not network.casts
    if dtype == "fp32":
        logits = np.array([0.0, 1.25, 0.25], dtype=np.float32)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        np.testing.assert_allclose(
            result["attn_out"].values, inputs.values * probabilities[1], rtol=1e-6
        )
