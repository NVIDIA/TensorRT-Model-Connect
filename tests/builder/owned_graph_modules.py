# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for tests that exercise one family-owned graph module."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ROOT / "python" / "tensorrt_model_connect" / "families"


def _family_dirs():
    for family_dir in sorted(FAMILIES.iterdir()):
        if not (family_dir / "plugin.py").is_file():
            continue
        yield family_dir


def _first_family_with(*module_files: str) -> str:
    for family_dir in _family_dirs():
        if all((family_dir / module_file).is_file() for module_file in module_files):
            return family_dir.name
    raise RuntimeError(f"No family owns required graph modules: {module_files}")


def _load_owned_module(
    module_files: tuple[str, ...],
    required_symbols: tuple[str, ...],
):
    last_error: Exception | None = None
    for module_file in module_files:
        module_name = ".".join(Path(module_file).with_suffix("").parts)
        for family_dir in _family_dirs():
            if not (family_dir / module_file).is_file():
                continue
            try:
                module = importlib.import_module(
                    f"tensorrt_model_connect.families.{family_dir.name}.{module_name}"
                )
            except ModuleNotFoundError as exc:
                if exc.name == "tensorrt":
                    last_error = exc
                    continue
                raise
            if all(hasattr(module, symbol) for symbol in required_symbols):
                return module
    if last_error is not None:
        pytest.skip(
            f"family {module_files} modules require TensorRT",
            allow_module_level=True,
        )
    suffix = f" with required symbols {required_symbols}" if required_symbols else ""
    raise RuntimeError(f"No family-owned module in {module_files} found{suffix}")


def load_owned_callable(
    module_file: str,
    symbol: str,
    *required_parameters: str,
):
    """Load an owner implementation that retains the requested capability."""
    last_error: Exception | None = None
    module_name = ".".join(Path(module_file).with_suffix("").parts)
    for family_dir in _family_dirs():
        if not (family_dir / module_file).is_file():
            continue
        try:
            module = importlib.import_module(
                f"tensorrt_model_connect.families.{family_dir.name}.{module_name}"
            )
        except ModuleNotFoundError as exc:
            if exc.name == "tensorrt":
                last_error = exc
                continue
            raise
        candidate = getattr(module, symbol, None)
        if candidate is None:
            continue
        parameters = inspect.signature(candidate).parameters
        if all(parameter in parameters for parameter in required_parameters):
            return candidate
    if last_error is not None:
        pytest.skip(
            f"family {module_name} modules require TensorRT",
            allow_module_level=True,
        )
    requirement = f" with parameters {required_parameters}" if required_parameters else ""
    raise RuntimeError(
        f"No family-owned {module_file} defines {symbol}{requirement}"
    )


class _OwnedModuleProxy:
    def __init__(self, *module_files: str):
        self._module_files = module_files

    def __getattr__(self, name: str) -> Any:
        module = _load_owned_module(self._module_files, (name,))
        return getattr(module, name)


def load_graph_ops(*required_symbols: str):
    module_files = ("model/model.py", "graph_ops.py")
    if not required_symbols:
        return _OwnedModuleProxy(*module_files)
    return _load_owned_module(module_files, tuple(required_symbols))


def load_family_graph_ops(family: str):
    """Load one family's graph module from either supported layout."""
    family_dir = FAMILIES / family
    module_file = (
        "model/model.py"
        if (family_dir / "model/model.py").is_file()
        else "graph_ops.py"
    )
    module_name = ".".join(Path(module_file).with_suffix("").parts)
    return importlib.import_module(
        f"tensorrt_model_connect.families.{family}.{module_name}"
    )


def load_graph_blocks():
    return _OwnedModuleProxy("model/model.py", "graph_blocks.py")
