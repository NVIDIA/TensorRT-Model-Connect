# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import __future__
import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from families.electra.config import ModelConfig


class _Tensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.shape = self.values.shape
        self.dtype = "float32"


def _layer(values):
    return SimpleNamespace(get_output=lambda index: _Tensor(values))


class _Network:
    def add_constant(self, shape, weights):
        return _layer(weights.reshape(shape))

    def add_elementwise(self, lhs, rhs, operation):
        if operation == "PROD":
            return _layer(lhs.values * rhs.values)
        assert operation == "SUM"
        return _layer(lhs.values + rhs.values)

    def add_activation(self, tensor, operation):
        if operation == "TANH":
            return _layer(np.tanh(tensor.values))
        assert operation == "GELU_ERF"
        values = tensor.values
        erf = np.asarray([math.erf(float(x) / math.sqrt(2)) for x in values.flat])
        return _layer(0.5 * values * (1 + erf.reshape(values.shape)))


@pytest.fixture
def add_activation():
    path = Path(__file__).resolve().parents[1] / "graph_ops.py"
    names = {"_cast_back_to_trt_dtype", "add_constant", "add_gelu_new", "add_activation"}
    functions = [
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    namespace = {
        "np": np,
        "trt": SimpleNamespace(
            Weights=lambda values: values,
            ActivationType=SimpleNamespace(TANH="TANH", GELU_ERF="GELU_ERF"),
            ElementWiseOperation=SimpleNamespace(PROD="PROD", SUM="SUM"),
        ),
    }
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace["add_activation"]


@pytest.mark.parametrize("activation", ["gelu", "gelu_new"])
@pytest.mark.parametrize("shape", [(7,), (1, 7), (7, 1)])
def test_checkpoint_gelu_definition(add_activation, activation, shape):
    values = np.asarray([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32).reshape(shape)
    config = ModelConfig.create_tiny("electra", hidden_act=activation)
    result = add_activation(_Network(), _Tensor(values), config.hidden_act)
    if activation == "gelu":
        expected = np.asarray(
            [0.5 * x * (1 + math.erf(x / math.sqrt(2))) for x in values.flat]
        ).reshape(shape)
    else:
        expected = (
            0.5 * values * (1 + np.tanh(np.sqrt(2 / np.pi) * (values + 0.044715 * values**3)))
        )
    assert result.shape == shape
    np.testing.assert_allclose(result.values, expected, rtol=1e-6, atol=1e-7)
