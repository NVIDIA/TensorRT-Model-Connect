# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static surface checks for model-owned vision-language debug runners."""

from __future__ import annotations

import ast
from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "python" / "tensorrt_model_connect" / "models"
DIFF_VL_TOOL = REPO_ROOT / "tools" / "diff_vl.py"


def _assigned_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.List, ast.Tuple)):
        return set().union(*(_assigned_names(item) for item in target.elts))
    return set()


def _concrete_top_level_bindings(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bindings = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bindings.update(_assigned_names(target))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            bindings.update(_assigned_names(node.target))
    return bindings


def _required_debug_runner_attributes() -> set[str]:
    tree = ast.parse(
        DIFF_VL_TOOL.read_text(encoding="utf-8"),
        filename=str(DIFF_VL_TOOL),
    )
    return {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "vl_debug"
    }


def _owner_debug_runners() -> list[Path]:
    runners = []
    for descriptor_path in sorted(MODELS_ROOT.glob("*/MODEL.toml")):
        with descriptor_path.open("rb") as stream:
            descriptor = tomllib.load(stream)
        if "VLPipelineTest" not in descriptor.get(
            "diff_framework_check_classes", ()
        ):
            continue
        runner_path = descriptor_path.parent / "vl_debug_runner.py"
        if runner_path.is_file():
            runners.append(runner_path)
    return runners


def test_owner_diff_vl_runners_cover_generic_tool_surface() -> None:
    required = _required_debug_runner_attributes()
    runners = _owner_debug_runners()

    assert required, f"No model-owned debug-runner attributes found in {DIFF_VL_TOOL}"
    assert runners, "No VLPipelineTest owners with a root vl_debug_runner.py found"

    violations = []
    for runner_path in runners:
        missing = sorted(required - _concrete_top_level_bindings(runner_path))
        if missing:
            relative_path = runner_path.relative_to(REPO_ROOT)
            violations.append(f"{relative_path}: missing {', '.join(missing)}")

    assert not violations, "\n".join(violations)
