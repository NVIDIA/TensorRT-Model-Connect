# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precision-contract tests for Mixtral expert routing."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np


plugin = importlib.import_module("tensorrt_model_connect.families.mixtral.plugin")


class _Tensor:
    def __init__(self, name: str, dtype: str) -> None:
        self.name = name
        self.dtype = dtype


class _Layer:
    def __init__(self, *outputs: _Tensor) -> None:
        self._outputs = outputs
        self.axes = 0
        self.axis = 0
        self.reshape_dims = ()

    def get_output(self, index: int) -> _Tensor:
        return self._outputs[index]


class _Network:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def add_cast(self, tensor: _Tensor, dtype: str) -> _Layer:
        output = _Tensor(f"cast({tensor.name})", dtype)
        self.events.append(("cast", tensor, dtype, output))
        return _Layer(output)

    def add_softmax(self, tensor: _Tensor) -> _Layer:
        self.events.append(("softmax", tensor))
        return _Layer(_Tensor("router_probabilities", tensor.dtype))

    def add_topk(self, tensor: _Tensor, operation, top_k: int, axes: int) -> _Layer:
        self.events.append(("topk", tensor, operation, top_k, axes))
        return _Layer(
            _Tensor("top_values", tensor.dtype),
            _Tensor("top_indices", "int32"),
        )

    def add_reduce(self, tensor: _Tensor, operation, axes: int, keep_dims: bool) -> _Layer:
        self.events.append(("reduce", tensor, operation, axes, keep_dims))
        return _Layer(_Tensor("top_values_sum", tensor.dtype))

    def add_elementwise(self, lhs: _Tensor, rhs: _Tensor, operation) -> _Layer:
        assert lhs.dtype == rhs.dtype
        output = _Tensor(f"{operation}({lhs.name},{rhs.name})", lhs.dtype)
        self.events.append(("elementwise", lhs, rhs, operation, output))
        return _Layer(output)

    def add_concatenation(self, tensors: list[_Tensor]) -> _Layer:
        self.events.append(("concatenation", tensors))
        return _Layer(_Tensor("experts", tensors[0].dtype))

    def add_slice(self, tensor: _Tensor, *, start, shape, stride) -> _Layer:
        self.events.append(("slice", tensor, start, shape, stride))
        return _Layer(_Tensor(f"slice({tensor.name})", tensor.dtype))

    def add_shuffle(self, tensor: _Tensor) -> _Layer:
        self.events.append(("shuffle", tensor))
        return _Layer(_Tensor(f"shuffle({tensor.name})", tensor.dtype))

    def add_gather(self, tensor: _Tensor, indices: _Tensor, axis: int) -> _Layer:
        self.events.append(("gather", tensor, indices, axis))
        return _Layer(_Tensor("expert_output", tensor.dtype))


def test_fp16_mixtral_quantizes_router_logits_before_fp32_routing(monkeypatch) -> None:
    """Match HF's FP16 router projection and FP32 routing boundaries."""
    fake_trt = SimpleNamespace(
        float32="fp32",
        TopKOperation=SimpleNamespace(MAX="max"),
        ReduceOperation=SimpleNamespace(SUM="sum"),
        ElementWiseOperation=SimpleNamespace(DIV="div", PROD="prod", SUM="sum"),
    )
    monkeypatch.setattr(plugin, "trt", fake_trt)

    def fake_matmul(network, inp, _in_features, _out_features, weight, *, dtype):
        assert inp.dtype == "fp16"
        assert dtype == np.float16
        assert weight.dtype == np.float32
        return _Tensor("router_logits", "fp16")

    monkeypatch.setattr(plugin.graph_ops, "add_matmul_rhs_constant", fake_matmul)
    monkeypatch.setattr(
        plugin,
        "_add_swiglu_expert",
        lambda *args, **kwargs: _Tensor("expert", "fp16"),
    )

    network = _Network()
    result = plugin._add_mixtral_moe_block(
        network=network,
        inp=_Tensor("hidden_states", "fp16"),
        weights={
            "layer.0.router": np.array(
                [[0.1, -0.2], [0.3, -0.4]], dtype=np.float32),
            **{
                f"layer.0.expert.{expert}.{name}": np.zeros((2, 2), dtype=np.float16)
                for expert in range(2)
                for name in ("w_gate", "w_up", "w_down")
            },
        },
        prefix="layer.0",
        hidden_size=2,
        num_experts=2,
        moe_intermediate=2,
        top_k=2,
        dtype=np.float16,
    )

    softmax_input = next(
        event[1] for event in network.events if event[0] == "softmax"
    )
    topk_input = next(event[1] for event in network.events if event[0] == "topk")
    router_cast = next(
        event for event in network.events
        if event[0] == "cast" and event[1].name == "router_logits"
    )
    assert router_cast[1].dtype == "fp16"
    assert router_cast[2] == "fp32"
    assert softmax_input.dtype == "fp32"
    assert topk_input.dtype == "fp32"

    products = [
        event
        for event in network.events
        if event[0] == "elementwise" and event[3] == "prod"
    ]
    assert products
    assert all(lhs.dtype == rhs.dtype == "fp32" for _, lhs, rhs, _, _ in products)
    assert result.dtype == "fp16"
