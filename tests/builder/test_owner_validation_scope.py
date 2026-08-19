# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Require owner tests to load only their own validation bindings."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "python" / "tensorrt_model_connect" / "models"


def _load_suites_call(node: ast.Call) -> bool:
    return (
        isinstance(node.func, ast.Name)
        and node.func.id == "load_suites"
    ) or (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "load_suites"
    )


def test_owner_load_suites_calls_are_explicitly_scoped() -> None:
    violations: list[str] = []
    owners = sorted(
        path
        for path in MODELS_ROOT.iterdir()
        if path.is_dir()
        and (path / "MODEL.toml").is_file()
        and (path / "model.py").is_file()
    )
    assert owners

    for owner in owners:
        tests_dir = owner / "tests"
        if not tests_dir.is_dir():
            continue
        for path in sorted(tests_dir.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
                if not _load_suites_call(call):
                    continue
                keywords = {keyword.arg: keyword.value for keyword in call.keywords}
                owner_scope = keywords.get("_owners")
                required = keywords.get("_require_all_suites")
                scoped_owners = (
                    {
                        item.value
                        for item in owner_scope.elts
                        if isinstance(item, ast.Constant)
                        and isinstance(item.value, str)
                    }
                    if isinstance(owner_scope, ast.Set)
                    else set()
                )
                if scoped_owners != {owner.name} or not (
                    isinstance(required, ast.Constant) and required.value is False
                ):
                    violations.append(
                        f"{path.relative_to(REPO_ROOT)}:{call.lineno}: "
                        "load_suites must set its exact owner scope and "
                        "_require_all_suites=False"
                    )

    assert not violations, "\n".join(violations)
