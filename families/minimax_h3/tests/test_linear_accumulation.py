# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-only typed graph contracts, not TensorRT kernel parity evidence."""

from types import SimpleNamespace

import ml_dtypes
import numpy as np
import pytest

from families.minimax_h3 import graph_ops as op


_STORAGE = {
    op.trt.float16: np.float16,
    op.trt.bfloat16: ml_dtypes.bfloat16,
    op.trt.float32: np.float32,
}


def _tensor(value, dtype):
    value = np.asarray(value, dtype=_STORAGE[dtype])
    return SimpleNamespace(shape=value.shape, dtype=dtype, value=value)


def _layer(tensor):
    return SimpleNamespace(get_output=lambda _index: tensor)


class _Network:
    def __init__(self):
        self.casts = []
        self.products = []
        self.sums = []

    def add_constant(self, shape, value):
        assert tuple(shape) == value.shape and value.dtype == np.float32
        return _layer(_tensor(value, op.trt.float32))

    def add_cast(self, value, dtype):
        self.casts.append((value.dtype, dtype))
        return _layer(_tensor(value.value, dtype))

    def add_matrix_multiply(self, left, left_op, right, right_op):
        assert left_op == op.trt.MatrixOperation.NONE
        assert right_op == op.trt.MatrixOperation.TRANSPOSE
        assert left.dtype == right.dtype
        self.products.append((left, right))
        result = left.value.astype(np.float32) @ right.value.astype(np.float32).swapaxes(-1, -2)
        return _layer(_tensor(result, left.dtype))

    def add_elementwise(self, left, right, operation):
        assert operation == op.trt.ElementWiseOperation.SUM
        assert left.dtype == right.dtype
        self.sums.append((left, right))
        result = left.value.astype(np.float32) + right.value.astype(np.float32)
        return _layer(_tensor(result, left.dtype))


@pytest.mark.parametrize("rank", [2, 3])
@pytest.mark.parametrize("source_dtype", [op.trt.float16, op.trt.float32])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("round_before_bias", [False, True])
def test_linear_float_accumulation_preserves_half_operands_and_bias_boundary(
    rank, source_dtype, with_bias, round_before_bias
):
    network = _Network()
    shape = (1,) * (rank - 1) + (2,)
    source = _tensor(np.array([1.0001, 0.0004883]).reshape(shape), source_dtype)
    weight = np.array([[1.0001, 1.0001], [0.3334, -0.1429]], np.float32)
    bias = np.array([1.2e-7, 0.00017], np.float32) if with_bias else None
    half_input = source.value.astype(np.float16).astype(np.float32)
    half_weight = weight.astype(np.float16).astype(np.float32)
    expected = half_input @ half_weight.T
    if with_bias:
        if round_before_bias:
            expected = expected.astype(np.float16).astype(np.float32)
        expected += bias.astype(np.float16).astype(np.float32)
    try:
        actual = op.linear(
            network,
            source,
            weight,
            bias,
            compute_dtype=op.trt.float16,
            accumulation_dtype=op.trt.float32,
            round_before_bias=round_before_bias,
        )
        assert actual.dtype == op.trt.float16
        np.testing.assert_array_equal(actual.value, expected.astype(np.float16))
        left, right = network.products[0]
        assert left.dtype == right.dtype == op.trt.float32
        assert right.shape == (1,) * (rank - 2) + weight.shape
        np.testing.assert_array_equal(left.value, half_input)
        np.testing.assert_array_equal(right.value.reshape(weight.shape), half_weight)
        assert (op.trt.float32, op.trt.float16) in network.casts
        assert (op.trt.float16, op.trt.float32) in network.casts
        if with_bias:
            assert len(network.sums) == 1
            assert all(value.dtype == op.trt.float32 for value in network.sums[0])
            np.testing.assert_array_equal(
                network.sums[0][1].value.reshape(bias.shape),
                bias.astype(np.float16).astype(np.float32),
            )
            # The separate-bias path must retain its first half rounding.
            assert actual.value.reshape(-1)[0] == np.float16(
                1.0 if round_before_bias else 1.0009765625
            )
        else:
            assert not network.sums
    finally:
        op.release_weight_buffers(network)


@pytest.mark.parametrize("compute_dtype", [op.trt.float16, op.trt.bfloat16, op.trt.float32])
@pytest.mark.parametrize("with_bias", [False, True])
def test_linear_default_accumulation_keeps_existing_graph(compute_dtype, with_bias):
    network = _Network()
    source = _tensor([[1.0001, 0.0004883]], op.trt.float32)
    weight = np.array([[1.0001, 1.0001]], np.float32)
    bias = np.array([1.2e-7], np.float32) if with_bias else None
    try:
        actual = op.linear(network, source, weight, bias, compute_dtype=compute_dtype)
        assert actual.dtype == compute_dtype
        assert all(value.dtype == compute_dtype for value in network.products[0])
        if with_bias:
            assert all(value.dtype == compute_dtype for value in network.sums[0])
        else:
            assert not network.sums
        assert (compute_dtype, op.trt.float32) not in network.casts
    finally:
        op.release_weight_buffers(network)


@pytest.mark.parametrize("accumulation_dtype", [op.trt.float16, op.trt.bfloat16, "float32"])
def test_linear_rejects_invalid_accumulation_before_graph_calls(accumulation_dtype):
    with pytest.raises(ValueError, match="FP32 accumulation"):
        op.linear(
            object(),
            _tensor([[1.0]], op.trt.float16),
            np.ones((1, 1), np.float32),
            accumulation_dtype=accumulation_dtype,
        )


@pytest.mark.parametrize("round_before_bias", [True, 1, "false"])
def test_linear_rejects_invalid_bias_rounding_before_graph_calls(round_before_bias):
    with pytest.raises(ValueError, match="round_before_bias"):
        op.linear(
            object(),
            _tensor([[1.0]], op.trt.float16),
            np.ones((1, 1), np.float32),
            round_before_bias=round_before_bias,
        )
