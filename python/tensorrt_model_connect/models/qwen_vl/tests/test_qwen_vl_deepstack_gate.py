# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL DeepStack gating regression tests."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass


plugin = importlib.import_module(
    "tensorrt_model_connect.models.qwen_vl.model")


@dataclass
class _Tensor:
    name: str
    dtype: object


class _Layer:
    def __init__(self, output: _Tensor) -> None:
        self.output = output

    def get_output(self, index: int) -> _Tensor:
        assert index == 0
        return self.output


class _Network:
    def __init__(self) -> None:
        self.operations: list[tuple] = []

    def add_cast(self, tensor: _Tensor, dtype: object) -> _Layer:
        self.operations.append(("cast", tensor.name, dtype))
        return _Layer(_Tensor("zero_cast", dtype))

    def add_elementwise(
        self, lhs: _Tensor, rhs: _Tensor, operation: object,
    ) -> _Layer:
        self.operations.append(("elementwise", lhs.name, rhs.name, operation))
        return _Layer(_Tensor("condition", plugin.trt.bool))

    def add_select(
        self, condition: _Tensor, when_true: _Tensor, when_false: _Tensor,
    ) -> _Layer:
        self.operations.append(
            ("select", condition.name, when_true.name, when_false.name))
        return _Layer(_Tensor("gated", when_true.dtype))


def test_deepstack_gate_uses_hard_zero_select(monkeypatch) -> None:
    network = _Network()
    embed = _Tensor("deepstack_embed", plugin.trt.float16)
    active = _Tensor("deepstack_active", plugin.trt.float16)

    monkeypatch.setattr(
        plugin.graph_ops,
        "add_constant",
        lambda *_args, **_kwargs: _Tensor("zero", plugin.trt.float32),
    )

    gated = plugin._add_nan_safe_deepstack_gate(network, embed, active)

    assert gated.name == "gated"
    assert network.operations == [
        ("cast", "zero", plugin.trt.float16),
        (
            "elementwise",
            "deepstack_active",
            "zero_cast",
            plugin.trt.ElementWiseOperation.GREATER,
        ),
        ("select", "condition", "deepstack_embed", "zero_cast"),
    ]
    assert all(
        operation[-1] != plugin.trt.ElementWiseOperation.PROD
        for operation in network.operations
        if operation[0] == "elementwise"
    )


def test_qwen3_deepstack_is_after_complete_decoder_layer() -> None:
    source = inspect.getsource(plugin._build_qwen3_vl_decoder)

    final_residual = source.index(
        "residual2 = network.add_elementwise(")
    deepstack_sum = source.index(
        "deepstack_sum = network.add_elementwise(")

    assert final_residual < deepstack_sum
    assert "post_attn_ds" not in source
