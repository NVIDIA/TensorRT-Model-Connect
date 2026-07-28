# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT topology checks for the qualified BF16 native-KV path."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

_FAMILIES = (
    "tensorrt_model_connect.families.qwen.graph_ops",
    "tensorrt_model_connect.families.llama.graph_ops",
)


def _add_native_attention(graph_ops, dtype):
    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    heads, kv_heads, head_dim, capacity = 4, 2, 128, 8
    q = network.add_input("q", dtype, (1, heads * head_dim))
    k = network.add_input("k", dtype, (1, kv_heads * head_dim))
    v = network.add_input("v", dtype, (1, kv_heads * head_dim))
    cache_shape = (1, kv_heads, capacity, head_dim)
    cache_k = network.add_input("cache_k", dtype, cache_shape)
    cache_v = network.add_input("cache_v", dtype, cache_shape)
    write_indices = network.add_input("cache_write_indices", trt.int32, (1,))
    lengths = network.add_input("key_value_lengths", trt.int32, (1,))
    graph_ops.add_native_kv_cache_attention_from_rows(
        network,
        q,
        k,
        v,
        cache_k,
        cache_v,
        write_indices,
        lengths,
        num_heads=heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        q_seq=1,
    )
    return network, lengths


@pytest.mark.parametrize("module_name", _FAMILIES)
def test_bf16_uses_exact_fp32_scale_before_fused_attention(
    monkeypatch, module_name,
):
    graph_ops = importlib.import_module(module_name)
    constants: list[tuple[tuple[int, ...], np.ndarray, np.dtype]] = []
    real_add_constant = graph_ops.add_constant

    def _record(network, shape, values, dtype=np.float32):
        constants.append((shape, np.asarray(values).copy(), np.dtype(dtype)))
        return real_add_constant(network, shape, values, dtype=dtype)

    monkeypatch.setattr(graph_ops, "add_constant", _record)
    network, lengths = _add_native_attention(graph_ops, trt.bfloat16)
    relevant = {
        trt.LayerType.CAST,
        trt.LayerType.CONSTANT,
        trt.LayerType.ELEMENTWISE,
        trt.LayerType.ATTENTION_INPUT,
    }
    layers = [
        network.get_layer(index)
        for index in range(network.num_layers)
        if network.get_layer(index).type in relevant
    ]

    assert len(constants) == 1
    shape, values, dtype = constants[0]
    assert shape == (1, 1, 1, 1)
    assert dtype == np.dtype(np.float32)
    assert float(values.item()) == pytest.approx(
        1.0 / np.sqrt(128), rel=0.0, abs=0.0
    )
    assert [layer.type for layer in layers] == [
        trt.LayerType.CAST,
        trt.LayerType.CONSTANT,
        trt.LayerType.ELEMENTWISE,
        trt.LayerType.CAST,
        trt.LayerType.ATTENTION_INPUT,
    ]
    upcast, constant, product, downcast, attention = layers
    assert upcast.get_output(0).dtype == trt.float32
    assert constant.get_output(0).dtype == trt.float32
    assert product.get_input(0) is upcast.get_output(0)
    assert downcast.get_input(0) is product.get_output(0)
    assert downcast.get_output(0).dtype == trt.bfloat16
    assert attention.get_input(0) is downcast.get_output(0)
    assert attention.get_input(6) is lengths


@pytest.mark.parametrize("module_name", _FAMILIES)
def test_unqualified_fp16_graph_fails_closed(module_name):
    graph_ops = importlib.import_module(module_name)

    with pytest.raises(ValueError, match="requires BF16"):
        _add_native_attention(graph_ops, trt.float16)
