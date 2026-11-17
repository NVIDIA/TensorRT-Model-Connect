# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ViT attention softmax runs in FP32 inside FP16 engines."""

from __future__ import annotations

import numpy as np
import pytest

trt = pytest.importorskip("tensorrt")

from families.dinov3 import graph_ops  # noqa: E402


class _Tensor:
    def __init__(self, dtype) -> None:
        self.dtype = dtype


class _Layer:
    def __init__(self, output) -> None:
        self.output = output
        self.axes = 0

    def get_output(self, index: int):
        assert index == 0
        return self.output


class _Network:
    """Records the dtypes the attention graph hands to each layer."""

    def __init__(self) -> None:
        self.softmax_inputs = []
        self.matmul_inputs = []

    def add_constant(self, _shape, weights):
        return _Layer(_Tensor(trt.float16 if weights.dtype == np.float16 else trt.float32))

    def add_cast(self, _tensor, dtype):
        return _Layer(_Tensor(dtype))

    def add_elementwise(self, lhs, _rhs, _op):
        return _Layer(_Tensor(lhs.dtype))

    def add_matrix_multiply(self, lhs, _op_lhs, rhs, _op_rhs):
        self.matmul_inputs.append((lhs.dtype, rhs.dtype))
        return _Layer(_Tensor(lhs.dtype))

    def add_softmax(self, tensor):
        self.softmax_inputs.append(tensor.dtype)
        return _Layer(_Tensor(tensor.dtype))


def test_fp16_attention_takes_the_softmax_in_fp32() -> None:
    network = _Network()
    q, k, v = (_Tensor(trt.float16) for _ in range(3))

    output = graph_ops.attention(network, q, k, v, head_dim=64, dtype=np.float16)

    assert network.softmax_inputs == [trt.float32]
    # Scores in FP16, then probabilities cast back to FP16 for the value matmul.
    assert network.matmul_inputs == [(trt.float16, trt.float16), (trt.float16, trt.float16)]
    assert output.dtype == trt.float16
