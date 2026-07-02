# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for tests that exercise one family-owned graph module."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ROOT / "python" / "tensorrt_model_connect" / "families"


def _first_family_with(*module_files: str) -> str:
    for family_dir in sorted(FAMILIES.iterdir()):
        if not (family_dir / "MODEL.toml").is_file():
            continue
        if all((family_dir / module_file).is_file() for module_file in module_files):
            return family_dir.name
    raise RuntimeError(f"No family owns required graph modules: {module_files}")


def load_graph_ops(*required_symbols: str) -> ModuleType:
    last_error: Exception | None = None
    for family_dir in sorted(FAMILIES.iterdir()):
        if not (family_dir / "MODEL.toml").is_file():
            continue
        if not (family_dir / "graph_ops.py").is_file():
            continue
        try:
            module = importlib.import_module(
                f"tensorrt_model_connect.families.{family_dir.name}.graph_ops")
        except ModuleNotFoundError as exc:
            if exc.name == "tensorrt":
                last_error = exc
                continue
            raise
        if all(hasattr(module, symbol) for symbol in required_symbols):
            return module
    if last_error is not None:
        pytest.skip("family graph_ops modules require TensorRT", allow_module_level=True)
    suffix = f" with required symbols {required_symbols}" if required_symbols else ""
    raise RuntimeError(f"No family-owned graph_ops.py found{suffix}")


def load_graph_blocks() -> ModuleType:
    last_error: Exception | None = None
    for family_dir in sorted(FAMILIES.iterdir()):
        if not (family_dir / "MODEL.toml").is_file():
            continue
        if not all((family_dir / module_file).is_file()
                   for module_file in ("graph_ops.py", "graph_blocks.py")):
            continue
        try:
            return importlib.import_module(
                f"tensorrt_model_connect.families.{family_dir.name}.graph_blocks")
        except ModuleNotFoundError as exc:
            if exc.name == "tensorrt":
                last_error = exc
                continue
            raise
    if last_error is not None:
        pytest.skip("family graph_blocks modules require TensorRT", allow_module_level=True)
    raise RuntimeError("No family-owned graph_blocks.py found")
