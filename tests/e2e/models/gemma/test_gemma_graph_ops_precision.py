# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precision-contract tests for Gemma graph operations."""

from __future__ import annotations

import numpy as np
import pytest


trt = pytest.importorskip("tensorrt")

from tensorrt_model_connect.families.gemma import graph_ops  # noqa: E402
from tests.builder.conftest import requires_trt, run_trt_graph  # noqa: E402


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


def test_gelu_tanh_uses_fused_trt_activation_for_fp16():
    network = _Network()

    output = graph_ops.add_gelu_new(network, _Tensor(trt.float16))

    assert network.activation_type == trt.ActivationType.GELU_TANH
    assert network.activation_dtype == trt.float32
    assert network.casts == [
        (trt.float16, trt.float32),
        (trt.float32, trt.float16),
    ]
    assert output.dtype == trt.float16


def test_gelu_tanh_uses_fused_trt_activation_for_fp32():
    network = _Network()

    output = graph_ops.add_gelu_new(network, _Tensor(trt.float32))

    assert network.activation_type == trt.ActivationType.GELU_TANH
    assert network.activation_dtype == trt.float32
    assert network.casts == []
    assert output.dtype == trt.float32


@requires_trt
def test_gelu_tanh_fp16_matches_fp32_reference():
    values = np.array(
        [-3.1015625, -2.5234375, -2.142578125,
         2.142578125, 2.521484375, 3.1015625],
        dtype=np.float16,
    ).reshape(1, -1)

    def build(network, inputs):
        activated = graph_ops.add_gelu_new(
            network, inputs["x"], dtype=np.float16)
        output = network.add_cast(activated, trt.float32).get_output(0)
        return {"out": output}

    result = run_trt_graph(build, {"x": values})["out"]
    x = values.astype(np.float32)
    reference = (
        0.5 * x * (
            1.0 + np.tanh(
                np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)
            )
        )
    ).astype(np.float16).astype(np.float32)
    np.testing.assert_allclose(result, reference, atol=5e-4, rtol=0.0)
