# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep SANA's nested LTX production graph dependencies owner-complete."""

from __future__ import annotations

import ast
from pathlib import Path


NESTED_LTX = Path(__file__).resolve().parents[1] / "components" / "ltx_video"
GRAPH_OPS = NESTED_LTX / "graph_ops.py"
_LOCAL_HELPER_PREFIXES = (
    "_add_",
    "_cast_",
    "_repeat_",
    "_scalar_",
    "add_",
    "make_",
    "reshape_",
)


def _defined_names(tree: ast.AST) -> set[str]:
    names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
    return names


def test_nested_ltx_production_graph_dependency_closure() -> None:
    graph_tree = ast.parse(GRAPH_OPS.read_text(encoding="utf-8"), filename=str(GRAPH_OPS))
    definitions = _defined_names(graph_tree)

    missing_internal = sorted(
        {
            (node.func.id, node.lineno)
            for node in ast.walk(graph_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.startswith(_LOCAL_HELPER_PREFIXES)
            and node.func.id not in definitions
        }
    )

    missing_consumers: list[tuple[str, int, str]] = []
    for path in sorted(NESTED_LTX.glob("*.py")):
        if path.name in {"__init__.py", "graph_ops.py"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "graph_ops"
                and node.attr not in definitions
            ):
                missing_consumers.append((path.name, node.lineno, node.attr))
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module is not None
                and node.module.endswith("graph_ops")
            ):
                missing_consumers.extend(
                    (path.name, node.lineno, alias.name)
                    for alias in node.names
                    if alias.name != "*" and alias.name not in definitions
                )

    assert not missing_internal, f"nested graph_ops has unresolved local calls: {missing_internal}"
    assert not missing_consumers, (
        f"nested LTX production modules reference missing graph_ops: {missing_consumers}"
    )
