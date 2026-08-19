# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for Wan2.2-owned TensorRT graph operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from tensorrt_model_connect.models.wan2_2_ti2v import trt_ops as op


@dataclass
class _Tensor:
    shape: tuple[int, int]
    dtype: str = "bfloat16"


class _Layer:
    def __init__(self, output: _Tensor) -> None:
        self._output = output

    def get_output(self, index: int) -> _Tensor:
        assert index == 0
        return self._output


class _Network:
    def __init__(self) -> None:
        self.slices: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []

    def add_slice(
        self,
        tensor: _Tensor,
        start: tuple[int, int],
        shape: tuple[int, int],
        stride: tuple[int, int],
    ) -> _Layer:
        assert tensor.dtype == "bfloat16"
        self.slices.append((start, shape, stride))
        return _Layer(_Tensor(shape, dtype=tensor.dtype))


def test_fused_qkv_linear_packs_qkv_in_order_and_preserves_bf16_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows, hidden_size = 7, 4
    x = _Tensor((rows, hidden_size))
    weights = tuple(
        np.full((hidden_size, hidden_size), fill_value, dtype=np.float32)
        for fill_value in (1.0, 2.0, 3.0)
    )
    biases = tuple(
        np.full((hidden_size,), fill_value, dtype=np.float32) for fill_value in (4.0, 5.0, 6.0)
    )
    calls = []

    def _linear(network, actual_x, weight, bias):
        calls.append((network, actual_x, weight, bias))
        return _Tensor((rows, 3 * hidden_size))

    monkeypatch.setattr(op, "linear", _linear)
    network = _Network()

    q, k, v = op.fused_qkv_linear(
        network,
        x,
        weights[0],
        biases[0],
        weights[1],
        biases[1],
        weights[2],
        biases[2],
        rows=rows,
        hidden_size=hidden_size,
    )

    assert len(calls) == 1
    _, actual_x, packed_weight, packed_bias = calls[0]
    assert actual_x is x
    assert packed_weight.shape == (3 * hidden_size, hidden_size)
    assert packed_bias.shape == (3 * hidden_size,)
    for index in range(3):
        np.testing.assert_array_equal(
            packed_weight[index * hidden_size : (index + 1) * hidden_size],
            weights[index],
        )
        np.testing.assert_array_equal(
            packed_bias[index * hidden_size : (index + 1) * hidden_size],
            biases[index],
        )
    reference_x = np.arange(2 * hidden_size, dtype=np.float32).reshape(2, hidden_size)
    separate_qkv = np.concatenate(
        [reference_x @ weight.T + bias for weight, bias in zip(weights, biases)],
        axis=1,
    )
    packed_qkv = reference_x @ packed_weight.T + packed_bias
    np.testing.assert_array_equal(packed_qkv, separate_qkv)
    assert network.slices == [
        ((0, 0), (rows, hidden_size), (1, 1)),
        ((0, hidden_size), (rows, hidden_size), (1, 1)),
        ((0, 2 * hidden_size), (rows, hidden_size), (1, 1)),
    ]
    assert (q.shape, k.shape, v.shape) == ((rows, hidden_size),) * 3
    assert (q.dtype, k.dtype, v.dtype) == ("bfloat16",) * 3


@pytest.mark.parametrize(("bad_weight", "bad_bias"), [(True, False), (False, True)])
def test_fused_qkv_linear_rejects_shape_drift(bad_weight: bool, bad_bias: bool) -> None:
    hidden_size = 4
    weight_shape = (hidden_size, hidden_size - 1) if bad_weight else (hidden_size, hidden_size)
    bias_shape = (hidden_size - 1,) if bad_bias else (hidden_size,)
    weights = [np.zeros((hidden_size, hidden_size), dtype=np.float32) for _ in range(3)]
    biases = [np.zeros((hidden_size,), dtype=np.float32) for _ in range(3)]
    weights[1] = np.zeros(weight_shape, dtype=np.float32)
    biases[2] = np.zeros(bias_shape, dtype=np.float32)

    with pytest.raises(ValueError, match="Q/K/V (weights|biases) must all have shape"):
        op.fused_qkv_linear(
            _Network(),
            _Tensor((2, hidden_size)),
            weights[0],
            biases[0],
            weights[1],
            biases[1],
            weights[2],
            biases[2],
            rows=2,
            hidden_size=hidden_size,
        )
