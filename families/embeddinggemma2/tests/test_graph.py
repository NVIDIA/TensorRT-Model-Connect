# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FP32 NumPy graph double: checks topology and dtypes, not GPU execution."""

import sys
from importlib.machinery import ModuleSpec
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from families.embeddinggemma2.config import Config
from families.embeddinggemma2.tests.test_config import tiny_config


class Tensor:
    def __init__(self, values, dtype="fp32"):
        self.values = np.asarray(values)
        self.dtype = dtype
        self.shape = self.values.shape


class Layer:
    def __init__(self, tensor):
        self.tensor = tensor

    def get_output(self, index):
        assert index == 0
        return self.tensor


class Shuffle(Layer):
    def __init__(self, tensor):
        super().__init__(tensor)
        self.reshape_dims = tensor.shape
        self.second_transpose = None

    def get_output(self, index):
        dims = tuple(self.tensor.shape[i] if d == 0 else d for i, d in enumerate(self.reshape_dims))
        values = self.tensor.values.reshape(dims)
        if self.second_transpose is not None:
            values = values.transpose(self.second_transpose)
        return Tensor(values, self.tensor.dtype)


class Network:
    def add_constant(self, shape, values):
        return Layer(Tensor(values.reshape(shape), "int32" if values.dtype == np.int32 else "fp32"))

    def add_cast(self, tensor, dtype):
        return Layer(Tensor(tensor.values, dtype))

    def add_shuffle(self, tensor):
        return Shuffle(tensor)

    def add_gather(self, data, indices, axis):
        assert indices.dtype == "int32"
        return Layer(Tensor(np.take(data.values, indices.values, axis), data.dtype))

    def add_matrix_multiply(self, a, a_op, b, b_op):
        assert a.dtype == b.dtype and a_op == "none"
        rhs = b.values.swapaxes(-1, -2) if b_op == "transpose" else b.values
        return Layer(Tensor(a.values @ rhs, a.dtype))

    def add_elementwise(self, a, b, op):
        assert a.dtype == b.dtype
        funcs = {
            "sum": np.add,
            "sub": np.subtract,
            "prod": np.multiply,
            "pow": np.power,
            "less": np.less,
        }
        return Layer(Tensor(funcs[op](a.values, b.values), "bool" if op == "less" else a.dtype))

    def add_unary(self, tensor, op):
        assert op == "abs"
        return Layer(Tensor(np.abs(tensor.values), tensor.dtype))

    def add_activation(self, tensor, op):
        assert op == "tanh"
        return Layer(Tensor(np.tanh(tensor.values), tensor.dtype))

    def add_reduce(self, tensor, op, axes, keep):
        assert op == "avg"
        dims = tuple(i for i in range(len(tensor.shape)) if axes & (1 << i))
        return Layer(Tensor(tensor.values.mean(axis=dims, keepdims=keep), tensor.dtype))

    def add_select(self, condition, a, b):
        assert condition.dtype == "bool" and a.dtype == b.dtype
        return Layer(Tensor(np.where(condition.values, a.values, b.values), a.dtype))

    def add_softmax(self, tensor):
        values = np.exp(tensor.values - tensor.values.max(axis=-1, keepdims=True))
        return Layer(Tensor(values / values.sum(axis=-1, keepdims=True), tensor.dtype))


@pytest.fixture
def graph(monkeypatch):
    trt = ModuleType("tensorrt")
    trt.__spec__ = ModuleSpec("tensorrt", loader=None)
    trt.__dict__.update(
        dict(
            float32="fp32",
            bfloat16="bf16",
            int32="int32",
            MatrixOperation=SimpleNamespace(NONE="none", TRANSPOSE="transpose"),
            ElementWiseOperation=SimpleNamespace(
                SUM="sum", SUB="sub", PROD="prod", POW="pow", LESS="less"
            ),
            UnaryOperation=SimpleNamespace(ABS="abs"),
            ActivationType=SimpleNamespace(TANH="tanh"),
            ReduceOperation=SimpleNamespace(AVG="avg"),
        )
    )
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    from families.embeddinggemma2 import graph

    return graph


def weights(config):
    rng = np.random.default_rng(31)
    result = {
        key: rng.normal(0, 0.2, shape).astype(np.float32)
        for key, shape in config.weight_shapes().items()
    }
    for key in result:
        if "norm.weight" in key or "layernorm.weight" in key or key.endswith("layer_scalar"):
            result[key].fill(1)
    return result


@pytest.mark.parametrize("dtype", ["fp32", "bf16"])
def test_encoder_dtype_and_future_token_dependence(graph, dtype):
    config = Config.from_dict(tiny_config())
    params = weights(config)
    positions = Tensor(np.arange(4, dtype=np.int32), "int32")
    first = graph.encode(
        Network(),
        Tensor(np.array([2, 5, 6, 1], dtype=np.int32), "int32"),
        positions,
        config,
        params,
        dtype,
        32,
    )
    second = graph.encode(
        Network(),
        Tensor(np.array([2, 5, 7, 1], dtype=np.int32), "int32"),
        positions,
        config,
        params,
        dtype,
        32,
    )
    assert first.dtype == "fp32" and first.shape == (4, 6)
    assert np.isfinite(first.values).all()
    assert not np.allclose(first.values[0], second.values[0])


def test_inclusive_bidirectional_local_mask(graph):
    positions = Tensor(np.arange(5, dtype=np.int32), "int32")
    mask = graph.local_mask(Network(), positions, 1)
    expected = np.abs(np.arange(5)[:, None] - np.arange(5)[None, :]) <= 1
    np.testing.assert_array_equal(mask.values == 0, expected[None, :, :])
    assert np.isneginf(mask.values[~expected[None, :, :]]).all()


def test_fp32_graph_matches_pinned_official_tiny_model(graph):
    reference = pytest.importorskip(
        "transformers.models.embedding_gemma2.modeling_embedding_gemma2"
    )
    from transformers import EmbeddingGemma2TextConfig
    import torch

    raw = tiny_config()["text_config"]
    params = weights(Config.from_dict(tiny_config()))
    model = reference.EmbeddingGemma2TextModel(EmbeddingGemma2TextConfig(**raw)).eval()
    model.config._attn_implementation = "eager"
    model.load_state_dict(
        {key: torch.from_numpy(value) for key, value in params.items()}, strict=True
    )
    ids = torch.tensor([[2, 5, 6, 1]])
    with torch.no_grad():
        expected = (
            model(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state[0].numpy()
        )
    actual = graph.encode(
        Network(),
        Tensor(ids[0].numpy().astype(np.int32), "int32"),
        Tensor(np.arange(4, dtype=np.int32), "int32"),
        Config.from_dict(tiny_config()),
        params,
        "fp32",
        32,
    )
    np.testing.assert_allclose(actual.values, expected, rtol=2e-5, atol=2e-6)
