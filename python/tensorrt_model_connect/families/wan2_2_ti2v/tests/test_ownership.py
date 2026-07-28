# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ownership guards for the standalone Wan2.2 implementation."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[5]
PRODUCTION_ROOT = REPO_ROOT / "python/tensorrt_model_connect/families/wan2_2_ti2v"
RUNTIME_ROOT = REPO_ROOT / "src/runtime/models/wan2_2_ti2v"
FORBIDDEN_WAN21 = re.compile(
    r"families[./]wan_t2v(?![A-Za-z0-9_])"
    r"|runtime/models/wan(?![A-Za-z0-9_])"
    r"|libtrtmc_model_wan\.so"
    r"|trtmc_model_wan(?![A-Za-z0-9_])"
    r"|diffusion_wan(?![A-Za-z0-9_])"
)
FORBIDDEN_NATIVE_TEXT = {
    "torch custom ops": re.compile(r"\btorch\.ops\b"),
    "Torch-TensorRT": re.compile(r"\btorch_tensorrt\b"),
    "runtime DSO loading": re.compile(r"\bctypes\.CDLL\b"),
}


def _production_python_sources() -> list[Path]:
    return sorted(
        path
        for path in PRODUCTION_ROOT.rglob("*.py")
        if path.is_file()
        and "tests" not in path.relative_to(PRODUCTION_ROOT).parts
        and "__pycache__" not in path.parts
    )


def test_wan22_production_python_is_native_only_and_does_not_reference_wan21() -> None:
    violations = []
    for path in _production_python_sources():
        source = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(source.splitlines(), 1):
            if FORBIDDEN_WAN21.search(line):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{line_number}: {line.strip()}")
            for label, pattern in FORBIDDEN_NATIVE_TEXT.items():
                if pattern.search(line):
                    violations.append(
                        f"{path.relative_to(REPO_ROOT)}:{line_number}: {label}: {line.strip()}"
                    )
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = (
                function.attr
                if isinstance(function, ast.Attribute)
                else function.id
                if isinstance(function, ast.Name)
                else ""
            )
            if name.startswith("add_plugin") or name == "get_plugin_registry":
                violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {name}")
    assert not violations


def test_wan22_torch_is_build_extra_not_global_runtime_dependency() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "torch>=2.0" not in pyproject["project"]["dependencies"]
    assert pyproject["project"]["optional-dependencies"]["wan"] == ["torch>=2.0"]


def test_wan22_runtime_settings_are_declarative_and_model_owned() -> None:
    checked_paths = [
        *RUNTIME_ROOT.glob("*.cpp"),
        *RUNTIME_ROOT.glob("*.h"),
    ]
    leftovers = [
        str(path.relative_to(REPO_ROOT))
        for path in checked_paths
        if "TRTMC_" "WAN22_" in path.read_text(encoding="utf-8")
    ]
    assert not leftovers

    manifest = (RUNTIME_ROOT / "MODEL.toml").read_text(encoding="utf-8")
    assert (
        'runtime_config_schemas = '
        '["config_schema.cpp|register_wan2_2_ti2v_schema"]'
    ) in manifest
