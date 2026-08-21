# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILIES_ROOT = REPO_ROOT / "python/tensorrt_model_connect/families"
ONE_GIB = 1 << 30


def _integer_literal(node: ast.expr | None) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.LShift)
        and isinstance(node.left, ast.Constant)
        and isinstance(node.left.value, int)
        and isinstance(node.right, ast.Constant)
        and isinstance(node.right.value, int)
    ):
        return node.left.value << node.right.value
    return None


def _target_mentions_workspace(node: ast.expr) -> bool:
    return isinstance(node, ast.Name) and "workspace" in node.id.lower()


def _function_defaults(node: ast.FunctionDef | ast.AsyncFunctionDef):
    positional = [*node.args.posonlyargs, *node.args.args]
    positional_defaults = zip(positional[-len(node.args.defaults) :], node.args.defaults)
    yield from positional_defaults
    yield from (
        (argument, default)
        for argument, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
        if default is not None
    )


def _one_gib_workspace_lines(path: Path) -> list[int]:
    source = path.read_text(encoding="utf-8")
    if "workspace" not in source.lower():
        return []
    tree = ast.parse(source, filename=str(path))
    violations: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "set_memory_pool_limit"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Attribute)
                and node.args[0].attr == "WORKSPACE"
                and _integer_literal(node.args[1]) == ONE_GIB
            ):
                violations.add(node.lineno)
            for keyword in node.keywords:
                if (
                    keyword.arg is not None
                    and "workspace" in keyword.arg.lower()
                    and _integer_literal(keyword.value) == ONE_GIB
                ):
                    violations.add(keyword.value.lineno)
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and len(node.args) >= 2
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and "workspace" in node.args[0].value.lower()
                and node.args[0].value.lower().endswith("_gib")
                and _integer_literal(node.args[1]) == 1
            ):
                violations.add(node.lineno)
            field_name = next(
                (
                    keyword.value.value
                    for keyword in node.keywords
                    if keyword.arg == "name"
                    and isinstance(keyword.value, ast.Constant)
                    and isinstance(keyword.value.value, str)
                ),
                None,
            )
            field_default = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "default"),
                None,
            )
            if (
                isinstance(field_name, str)
                and "workspace" in field_name.lower()
                and field_name.lower().endswith("_gib")
                and _integer_literal(field_default) == 1
            ):
                violations.add(node.lineno)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(_target_mentions_workspace(target) for target in targets):
                if _integer_literal(node.value) == ONE_GIB:
                    violations.add(node.lineno)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for argument, default in _function_defaults(node):
                if "workspace" in argument.arg.lower() and _integer_literal(default) == ONE_GIB:
                    violations.add(default.lineno)
    return sorted(violations)


def test_model_builders_do_not_impose_one_gib_workspace_limits() -> None:
    violations = {
        path.relative_to(REPO_ROOT): lines
        for path in FAMILIES_ROOT.rglob("*.py")
        if "tests" not in path.parts
        if (lines := _one_gib_workspace_lines(path))
    }

    assert not violations, f"fixed 1 GiB TensorRT workspace limits found: {violations}"
