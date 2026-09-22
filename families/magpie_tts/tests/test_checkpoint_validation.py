# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path

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
@pytest.mark.parametrize("shape", [(), (6,), (5, 2), (6, 3), (6, 2, 1)])
def test_fused_qkv_rejects_malformed_shapes(optimize, shape):
    env = _load_source(["_to_np", "_split_fused_qkv"], optimize)
    with pytest.raises(ValueError, match="Expected fused QKV"):
        env["_split_fused_qkv"](np.zeros(shape), 2)


@pytest.mark.parametrize("optimize", [0, 2])
@pytest.mark.parametrize("shape", [(), (4,), (3, 6), (4, 6, 1)])
def test_fused_kv_rejects_malformed_shapes(optimize, shape):
    env = _load_source(["_to_np", "_split_fused_kv"], optimize)
    with pytest.raises(ValueError, match="Expected fused KV"):
        env["_split_fused_kv"](np.zeros(shape), 2)


@pytest.mark.parametrize("optimize", [0, 2])
def test_valid_fused_weights_keep_order_and_layout(optimize):
    env = _load_source(["_to_np", "_split_fused_qkv", "_split_fused_kv"], optimize)
    for name, rows, chunks in [("_split_fused_qkv", 6, 3), ("_split_fused_kv", 4, 2)]:
        weights = np.arange(rows * 2, dtype=np.float32).reshape(rows, 2)
        outputs = env[name](weights, 2)
        assert len(outputs) == chunks
        for index, output in enumerate(outputs):
            np.testing.assert_array_equal(output, weights[index * 2 : (index + 1) * 2].T)
            assert output.flags.c_contiguous
