# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precision-contract tests for Gemma graph operations."""

from __future__ import annotations

import pytest


trt = pytest.importorskip("tensorrt")

from families.gemma import graph_ops  # noqa: E402


class _Tensor:
    def __init__(self, dtype, shape=(1, 8)):
        self.dtype = dtype
        self.shape = shape


class _Layer:
    def __init__(self, output):
        self._output = output

    def get_output(self, index):
        assert index == 0
        return self._output


class _Network:
    def __init__(self):
        self.activation_type = None
        self.activation_dtype = None
        self.casts = []

    def add_cast(self, tensor, dtype):
        self.casts.append((tensor.dtype, dtype))
        return _Layer(_Tensor(dtype, tensor.shape))

    def add_activation(self, tensor, activation_type):
        self.activation_type = activation_type
        self.activation_dtype = tensor.dtype
        return _Layer(_Tensor(tensor.dtype, tensor.shape))


def test_gelu_tanh_uses_fused_trt_activation_for_fp16() -> None:
    network = _Network()

    output = graph_ops.add_gelu_new(network, _Tensor(trt.float16))

    assert network.activation_type == trt.ActivationType.GELU_TANH
    assert network.activation_dtype == trt.float32
    assert network.casts == [
        (trt.float16, trt.float32),
        (trt.float32, trt.float16),
    ]
    assert output.dtype == trt.float16


def test_gelu_tanh_uses_fused_trt_activation_for_fp32() -> None:
    network = _Network()

    output = graph_ops.add_gelu_new(network, _Tensor(trt.float32))

    assert network.activation_type == trt.ActivationType.GELU_TANH
    assert network.activation_dtype == trt.float32
    assert network.casts == []
    assert output.dtype == trt.float32
