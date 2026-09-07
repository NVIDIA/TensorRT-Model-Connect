# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository-wide contract for fixed-capacity native KV attention."""

from __future__ import annotations

import ast
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_FAMILIES = _ROOT / "python" / "tensorrt_model_connect" / "families"
_FIXED_KV_OWNERS = {
    _FAMILIES / "qwen" / "graph_ops.py",
    _FAMILIES / "llama" / "graph_ops.py",
    _FAMILIES / "lfm2" / "model.py",
    _FAMILIES / "k2_horizon" / "model" / "model.py",
}


def test_every_fixed_kv_owner_uses_shared_explicit_attention() -> None:
    discovered = {
        path
        for path in _FAMILIES.rglob("*.py")
        if "add_kv_cache_update" in path.read_text(encoding="utf-8")
    }

    assert discovered == _FIXED_KV_OWNERS
    for path in discovered:
        source = path.read_text(encoding="utf-8")
        assert "add_explicit_masked_grouped_query_attention" in source
        assert "add_attention_v2" not in source


def test_no_family_attaches_key_value_lengths_to_attention() -> None:
    assignments: list[tuple[Path, int]] = []
    for path in _FAMILIES.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and target.attr == "key_value_lengths"
                ):
                    assignments.append((path, node.lineno))

    assert assignments == []
