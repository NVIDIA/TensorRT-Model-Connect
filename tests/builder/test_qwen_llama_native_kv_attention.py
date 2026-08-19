# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT topology checks for the qualified BF16 native-KV path."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

_FAMILIES = (
    "tensorrt_model_connect.models.qwen.graph_ops",
    "tensorrt_model_connect.models.llama.graph_ops",
)


class _FakeTensor:
    def __init__(self, name, dtype):
        self.name = name
        self.dtype = dtype


class _FakeLayer:
    def __init__(self, output):
        self._output = output

    def get_output(self, index):
        assert index == 0
        return self._output


class _FakeNetwork:
    def __init__(self):
        self.calls = []

    def _layer(self, operation, dtype):
        return _FakeLayer(_FakeTensor(f"{operation}_{len(self.calls)}", dtype))

    def add_kv_cache_update(self, cache, update, indices, mode):
        layer = self._layer("kv_cache_update", cache.dtype)
        self.calls.append(("kv_cache_update", cache, update, indices, mode, layer))
        return layer

    def add_cast(self, tensor, dtype):
        layer = self._layer("cast", dtype)
        self.calls.append(("cast", tensor, dtype, layer))
        return layer

    def add_elementwise(self, lhs, rhs, operation):
        layer = self._layer("elementwise", lhs.dtype)
        self.calls.append(("elementwise", lhs, rhs, operation, layer))
        return layer

    def add_attention_v2(self, q, k, v, normalization, causal_mask):
        layer = self._layer("attention", q.dtype)
        self.calls.append(("attention", q, k, v, normalization, causal_mask, layer))
        return layer


def _add_native_attention(monkeypatch, graph_ops, dtype):
    network = _FakeNetwork()
    heads, kv_heads, head_dim = 4, 2, 128
    tensors = {
        "q": _FakeTensor("q", dtype),
        "k": _FakeTensor("k", dtype),
        "v": _FakeTensor("v", dtype),
        "cache_k": _FakeTensor("cache_k", dtype),
        "cache_v": _FakeTensor("cache_v", dtype),
        "write_indices": _FakeTensor("cache_write_indices", trt.int32),
        "lengths": _FakeTensor("key_value_lengths", trt.int32),
    }
    monkeypatch.setattr(
        graph_ops,
        "reshape_rows_to_heads_4d",
        lambda _network, tensor, *_args, **_kwargs: tensor,
    )
    monkeypatch.setattr(
        graph_ops,
        "reshape_heads_4d_to_rows",
        lambda _network, tensor, *_args, **_kwargs: tensor,
    )
    result = graph_ops.add_native_kv_cache_attention_from_rows(
        network,
        tensors["q"],
        tensors["k"],
        tensors["v"],
        tensors["cache_k"],
        tensors["cache_v"],
        tensors["write_indices"],
        tensors["lengths"],
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        q_seq=1,
    )
    return network, tensors, result


@pytest.mark.parametrize("module_name", _FAMILIES)
def test_bf16_uses_exact_fp32_scale_before_fused_attention(
    monkeypatch,
    module_name,
):
    graph_ops = importlib.import_module(module_name)
    constants: list[tuple[tuple[int, ...], np.ndarray, np.dtype]] = []

    def _record(network, shape, values, dtype=np.float32):
        constants.append((shape, np.asarray(values).copy(), np.dtype(dtype)))
        tensor = _FakeTensor("scale", trt.float32)
        network.calls.append(("constant", tensor))
        return tensor

    monkeypatch.setattr(graph_ops, "add_constant", _record)
    network, tensors, result = _add_native_attention(monkeypatch, graph_ops, trt.bfloat16)
    relevant = [
        call
        for call in network.calls
        if call[0] in {"cast", "constant", "elementwise", "attention"}
    ]

    assert len(constants) == 1
    shape, values, dtype = constants[0]
    assert shape == (1, 1, 1, 1)
    assert dtype == np.dtype(np.float32)
    assert float(values.item()) == pytest.approx(1.0 / np.sqrt(128), rel=0.0, abs=0.0)
    assert [call[0] for call in relevant] == [
        "cast",
        "constant",
        "elementwise",
        "cast",
        "attention",
    ]
    upcast, constant, product, downcast, attention = relevant
    assert upcast[1] is tensors["q"]
    assert upcast[2] == trt.float32
    assert upcast[3].get_output(0).dtype == trt.float32
    assert constant[1].dtype == trt.float32
    assert product[1] is upcast[3].get_output(0)
    assert product[2] is constant[1]
    assert product[3] == trt.ElementWiseOperation.PROD
    assert downcast[1] is product[4].get_output(0)
    assert downcast[2] == trt.bfloat16
    assert attention[1] is downcast[3].get_output(0)
    assert attention[2] is result["present_k"]
    assert attention[3] is result["present_v"]
    assert attention[4] == trt.AttentionNormalizationOp.SOFTMAX
    assert attention[5] == trt.CausalMaskKind.LOWER_RIGHT
    assert attention[6].decomposable is False
    assert attention[6].key_value_lengths is tensors["lengths"]
    assert result["context"] is attention[6].get_output(0)


@pytest.mark.parametrize("module_name", _FAMILIES)
def test_unqualified_fp16_graph_fails_closed(monkeypatch, module_name):
    graph_ops = importlib.import_module(module_name)

    with pytest.raises(ValueError, match="requires BF16"):
        _add_native_attention(monkeypatch, graph_ops, trt.float16)
