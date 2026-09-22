# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_source(names, optimize, **namespace):
    source = Path(__file__).parents[1] / "model.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names
    ]
    assert {node.name for node in nodes} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    env = {"np": np, "Path": Path, **namespace}
    exec(compile(module, str(source), "exec", optimize=optimize), env)
    return env


@pytest.mark.parametrize("optimize", [0, 2])
@pytest.mark.parametrize("pattern", ["M?*", "M-x*-M", " M", "m", "M\n"])
def test_unknown_hybrid_symbols_are_rejected(optimize, pattern):
    parser = _load_source(["_parse_layer_types"], optimize)["_parse_layer_types"]
    with pytest.raises(ValueError, match="pattern"):
        parser(pattern)


@pytest.mark.parametrize("optimize", [0, 2])
def test_valid_hybrid_pattern_preserves_every_layer(optimize):
    parser = _load_source(["_parse_layer_types"], optimize)["_parse_layer_types"]
    assert parser("M-*M") == ["mamba2", "mlp", "attention", "mamba2"]


@pytest.mark.parametrize("optimize", [0, 2])
@pytest.mark.parametrize("pattern,layers", [("M-", 3), ("M?*", 2), ("M?", 2)])
def test_loader_rejects_bad_raw_pattern_before_tensor_access(optimize, pattern, layers):
    def unexpected_tensor(*_):
        pytest.fail("malformed pattern reached checkpoint tensor loading")

    env = _load_source(
        ["_parse_layer_types", "_NemotronHModel"],
        optimize,
        _open_safetensors=lambda _: [],
        _load_tensor=unexpected_tensor,
    )
    config = SimpleNamespace(
        hidden_size=8,
        vocab_size=4,
        num_hidden_layers=layers,
        num_attention_heads=2,
        head_dim=4,
        raw={"hybrid_override_pattern": pattern},
    )
    with pytest.raises(ValueError, match="[Pp]attern"):
        env["_NemotronHModel"]().load_weights("unused", config)
