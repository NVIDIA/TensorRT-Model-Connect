# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The constant sharing of ``graph.Net`` on a recording stand-in for the TensorRT network (no GPU needed)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from families.esmfold2.graph import Net


class _Tensor:
    def __init__(self, shape: tuple[int, ...], dtype: Any, values: np.ndarray | None = None) -> None:
        self.shape = tuple(shape)
        self.dtype = dtype
        self.values = values


class _Layer:
    def __init__(self, output: _Tensor) -> None:
        self._output = output

    def get_output(self, index: int) -> _Tensor:
        assert index == 0
        return self._output


class _Weights:
    def __init__(self, *arguments: Any) -> None:
        assert len(arguments) == 1, "the test builds float32 constants only"
        self.array = arguments[0]


class _Network:
    def __init__(self, trt: Any) -> None:
        self.trt = trt
        self.norms: list[tuple[_Tensor, _Tensor]] = []
        self.matmul_rhs: list[_Tensor] = []

    def add_constant(self, shape: tuple[int, ...], weights: _Weights) -> _Layer:
        return _Layer(_Tensor(shape, self.trt.float32, np.array(weights.array)))

    def add_cast(self, tensor: _Tensor, dtype: Any) -> _Layer:
        return _Layer(_Tensor(tensor.shape, dtype))

    def add_normalization_v2(self, tensor: _Tensor, gamma: _Tensor, beta: _Tensor, axes: int) -> _Layer:
        self.norms.append((gamma, beta))
        return _Layer(_Tensor(tensor.shape, tensor.dtype))

    def add_matrix_multiply(self, lhs: _Tensor, lhs_op: Any, rhs: _Tensor, rhs_op: Any) -> _Layer:
        self.matmul_rhs.append(rhs)
        return _Layer(_Tensor(lhs.shape[:-1] + rhs.shape[-1:], lhs.dtype))

    def add_elementwise(self, lhs: _Tensor, rhs: _Tensor, op: Any) -> _Layer:
        return _Layer(_Tensor(max(lhs.shape, rhs.shape, key=len), lhs.dtype))


def _net(weights: dict[str, np.ndarray], reads: list[str]) -> tuple[Net, _Network]:
    trt = SimpleNamespace(
        float32="float32", bfloat16="bfloat16", int32="int32", int64="int64", bool="bool", Weights=_Weights,
        ElementWiseOperation=SimpleNamespace(SUM="sum", PROD="prod"),
        MatrixOperation=SimpleNamespace(NONE="none", TRANSPOSE="transpose"),
    )
    network = _Network(trt)

    def weight(name: str) -> np.ndarray:
        reads.append(name)
        return weights[name]

    return Net(network, trt, trt.float32, weight), network


def test_weightless_layer_norm_constants_follow_the_input_width() -> None:
    net, network = _net({}, [])
    net.layer_norm(_Tensor((1, 5, 4), "float32"), None)
    net.layer_norm(_Tensor((1, 5, 8), "float32"), None)
    net.layer_norm(_Tensor((1, 7, 8), "float32"), None)
    (gamma4, beta4), (gamma8, beta8), (gamma8_again, _) = network.norms
    assert gamma4.shape == beta4.shape == (1, 1, 4)
    assert gamma8.shape == beta8.shape == (1, 1, 8)
    np.testing.assert_array_equal(gamma8.values, np.ones((1, 1, 8), np.float32))
    np.testing.assert_array_equal(beta8.values, np.zeros((1, 1, 8), np.float32))
    assert gamma8_again is gamma8  # the same width shares its constants


def test_an_explicit_weight_is_not_shared_under_the_checkpoint_prefix() -> None:
    checkpoint = np.arange(12, dtype=np.float32).reshape(3, 4)
    reads: list[str] = []
    net, network = _net({"proj.weight": checkpoint}, reads)
    x = _Tensor((1, 5, 4), "float32")
    net.linear(x, "proj", weight=np.ones((2, 4), np.float32))
    output = net.linear(x, "proj")
    assert reads == ["proj.weight"]
    assert output.shape == (1, 5, 3)
    np.testing.assert_array_equal(network.matmul_rhs[1].values, checkpoint.T.reshape(1, 4, 3))
    assert net.linear(x, "proj") is not None and reads == ["proj.weight"]  # the checkpoint weight is shared


def test_linear_without_a_weight_source_is_rejected() -> None:
    net, _ = _net({}, [])
    with pytest.raises(ValueError, match="checkpoint prefix or an explicit weight"):
        net.linear(_Tensor((1, 5, 4), "float32"))
