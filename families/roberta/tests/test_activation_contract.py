# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the actual activation builder against both GELU definitions on CPU."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("tensorrt")
from families.roberta import graph_ops


class _Tensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.shape = self.values.shape
        self.dtype = graph_ops.trt.float32


def _layer(values):
    return SimpleNamespace(get_output=lambda index: _Tensor(values))


class _Network:
    def add_constant(self, shape, weights):
        return _layer(weights.numpy().reshape(shape))

    def add_elementwise(self, lhs, rhs, operation):
        if operation == graph_ops.trt.ElementWiseOperation.PROD:
            return _layer(lhs.values * rhs.values)
        assert operation == graph_ops.trt.ElementWiseOperation.SUM
        return _layer(lhs.values + rhs.values)

    def add_activation(self, tensor, operation):
        if operation == graph_ops.trt.ActivationType.TANH:
            return _layer(np.tanh(tensor.values))
        assert operation == graph_ops.trt.ActivationType.GELU_ERF
        return _layer([0.5 * x * (1 + math.erf(x / math.sqrt(2))) for x in tensor.values])


@pytest.mark.parametrize("activation", ["gelu", "gelu_new"])
def test_checkpoint_activation_preserves_its_gelu_definition(activation):
    x = np.asarray([-3.0, -1.0, 0.0, 1.0, 3.0], dtype=np.float32)
    actual = graph_ops.add_activation(_Network(), _Tensor(x), activation).values
    if activation == "gelu":
        expected = [0.5 * v * (1 + math.erf(v / math.sqrt(2))) for v in x]
    else:
        expected = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
