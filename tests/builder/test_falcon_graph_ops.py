# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused graph-operation tests for the Falcon family."""

from __future__ import annotations

import pytest


trt = pytest.importorskip(
    "tensorrt", reason="TensorRT is required for Falcon graph tests")

from tensorrt_model_connect.families.falcon import graph_ops  # noqa: E402


class _ActivationLayer:
    def __init__(self, output):
        self._output = output

    def get_output(self, index: int):
        assert index == 0
        return self._output


class _ActivationNetwork:
    def __init__(self):
        self.activation_type = None

    def add_activation(self, tensor, activation_type):
        self.activation_type = activation_type
        return _ActivationLayer(tensor)


class _Tensor:
    def __init__(self, dtype):
        self.dtype = dtype


class _GraphLayer:
    def __init__(self, output):
        self._output = output
        self.axis = None
        self.reshape_dims = None

    def get_output(self, index: int):
        assert index == 0
        return self._output

    def set_input(self, index: int, _tensor):
        assert index == 1


class _AlibiNetwork:
    def __init__(self):
        self.casts = []
        self.elementwise = []

    def add_cast(self, _tensor, dtype):
        self.casts.append(dtype)
        return _GraphLayer(_Tensor(dtype))

    def add_concatenation(self, tensors):
        return _GraphLayer(_Tensor(tensors[0].dtype))

    def add_shape(self, _tensor):
        return _GraphLayer(_Tensor(trt.int64))

    def add_constant(self, _shape, weights):
        return _GraphLayer(_Tensor(weights.dtype))

    def add_slice(self, tensor, **_kwargs):
        return _GraphLayer(_Tensor(tensor.dtype))

    def add_shuffle(self, tensor):
        return _GraphLayer(_Tensor(tensor.dtype))

    def add_elementwise(self, lhs, rhs, operation):
        self.elementwise.append((operation, lhs.dtype, rhs.dtype))
        return _GraphLayer(_Tensor(lhs.dtype))


def test_falcon_gelu_uses_exact_erf_variant():
    network = _ActivationNetwork()
    tensor = object()

    output = graph_ops.add_activation(network, tensor, "gelu")

    assert output is tensor
    assert network.activation_type == trt.ActivationType.GELU_ERF


def test_falcon_alibi_matches_reference_bfloat16_rounding_path():
    network = _AlibiNetwork()

    output = graph_ops.add_alibi_mask_4d(
        network,
        _Tensor(trt.float16),
        _Tensor(trt.int32),
        _Tensor(trt.float32),
        _Tensor(trt.float32),
        num_heads=32,
        target_dtype=trt.float16,
        bias_scale=0.125,
    )

    products = [
        operands
        for operation, *operands in network.elementwise
        if operation == trt.ElementWiseOperation.PROD
    ]
    assert products == [
        [trt.bfloat16, trt.bfloat16],
        [trt.float16, trt.float16],
    ]
    assert all(
        operation != trt.ElementWiseOperation.SUB
        for operation, *_operands in network.elementwise
    )
    assert network.casts.count(trt.bfloat16) == 2
    assert output.dtype == trt.float16
