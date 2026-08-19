# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PersonaPlex-owned graph operation dependency and behavior contracts."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


OWNER_ROOT = Path(__file__).resolve().parents[1]


def _module_symbols(path: Path) -> set[str]:
    symbols: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.Assign):
            symbols.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            symbols.add(node.target.id)
    return symbols


def test_personaplex_builders_only_call_concrete_graph_ops() -> None:
    available = _module_symbols(OWNER_ROOT / "graph_ops.py")
    missing: list[tuple[str, int, str]] = []

    for path in sorted(OWNER_ROOT.glob("*.py")):
        if path.name == "graph_ops.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        aliases = {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if alias.name == "graph_ops"
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in aliases
                and node.attr not in available
            ):
                missing.append((path.name, node.lineno, node.attr))

    assert missing == []


def test_personaplex_activation_dispatch_preserves_specialized_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("tensorrt")
    from tensorrt_model_connect.models.personaplex import graph_ops

    class Tensor:
        pass

    class Layer:
        def __init__(self, output):
            self.output = output

        def get_output(self, index):
            assert index == 0
            return self.output

    class Network:
        def __init__(self):
            self.activations = []
            self.elementwise = []

        def add_activation(self, tensor, activation):
            output = Tensor()
            self.activations.append((tensor, activation, output))
            return Layer(output)

        def add_elementwise(self, left, right, operation):
            output = Tensor()
            self.elementwise.append((left, right, operation, output))
            return Layer(output)

    input_tensor = Tensor()
    network = Network()
    gelu_output = Tensor()
    monkeypatch.setattr(graph_ops, "add_gelu_new", lambda *_args, **_kwargs: gelu_output)

    assert graph_ops.add_activation(network, input_tensor, "gelu") is gelu_output
    relu_output = graph_ops.add_activation(network, input_tensor, "relu")
    squared_output = graph_ops.add_activation(network, input_tensor, "squared_relu")
    silu_output = graph_ops.add_activation(network, input_tensor, "silu")

    assert relu_output is network.activations[0][2]
    assert squared_output is network.elementwise[0][3]
    assert silu_output is network.elementwise[1][3]
    assert network.activations[0][1] == graph_ops.trt.ActivationType.RELU
    assert network.activations[1][1] == graph_ops.trt.ActivationType.RELU
    assert network.activations[2][1] == graph_ops.trt.ActivationType.SIGMOID
    assert network.elementwise[0][0] is network.elementwise[0][1]
    assert network.elementwise[0][2] == graph_ops.trt.ElementWiseOperation.PROD
    assert network.elementwise[1][0] is input_tensor
    assert network.elementwise[1][2] == graph_ops.trt.ElementWiseOperation.PROD
    with pytest.raises(ValueError, match="Unsupported activation"):
        graph_ops.add_activation(network, input_tensor, "unsupported")
