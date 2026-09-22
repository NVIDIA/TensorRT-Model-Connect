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
@pytest.mark.parametrize("shape", [(), (4,), (3, 8), (4, 7), (4, 8, 1)])
def test_loader_rejects_complete_malformed_embedding_shape(optimize, shape):
    embedding = np.zeros(shape, dtype=np.float32)
    env = _load_source(
        ["_Olmo2Model"],
        optimize,
        WeightDict=dict,
        _open_safetensors=lambda _: [],
        _load_tensor=lambda *_: embedding,
    )
    config = SimpleNamespace(
        hidden_size=8, vocab_size=4, num_hidden_layers=0, num_attention_heads=2
    )
    with pytest.raises(ValueError, match="Embedding shape") as error:
        env["_Olmo2Model"]().load_weights("unused", config)
    assert " != " in str(error.value)
    assert "!==" not in str(error.value)


@pytest.mark.parametrize("optimize", [0, 2])
def test_valid_embedding_mapping_is_preserved(optimize):
    embedding = np.arange(32, dtype=np.float32).reshape(4, 8)
    tensors = {
        "model.embed_tokens.weight": embedding,
        "model.norm.weight": np.ones(8, dtype=np.float32),
        "lm_head.weight": embedding.copy(),
    }
    env = _load_source(
        ["_Olmo2Model"],
        optimize,
        WeightDict=dict,
        _open_safetensors=lambda _: [],
        _load_tensor=lambda _, name: tensors[name],
        _has_tensor=lambda _, name: name in tensors,
        _transpose_2d=lambda tensor, _: np.ascontiguousarray(tensor.T),
    )
    config = SimpleNamespace(
        hidden_size=8,
        vocab_size=4,
        num_hidden_layers=0,
        num_attention_heads=2,
        max_position_embeddings=4,
        raw={},
    )
    weights = env["_Olmo2Model"]().load_weights("unused", config)
    np.testing.assert_array_equal(weights["embedding"], embedding)
    np.testing.assert_array_equal(weights["w_out"], embedding.T)
