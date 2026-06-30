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


def _load_owned_module(module_file: str, required_symbols: tuple[str, ...]):
    last_error: Exception | None = None
    module_name = ".".join(Path(module_file).with_suffix("").parts)
    for family_dir in _family_dirs():
        if not (family_dir / module_file).is_file():
            continue
        try:
            module = importlib.import_module(
                f"tensorrt_model_connect.families.{family_dir.name}.{module_name}")
        except ModuleNotFoundError as exc:
            if exc.name == "tensorrt":
                last_error = exc
                continue
            raise
        if all(hasattr(module, symbol) for symbol in required_symbols):
            return module
    if last_error is not None:
        pytest.skip(
            f"family {module_name} modules require TensorRT",
            allow_module_level=True,
        )
    suffix = f" with required symbols {required_symbols}" if required_symbols else ""
    raise RuntimeError(f"No family-owned {module_file} found{suffix}")


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
    def __init__(self, module_file: str):
        self._module_file = module_file

    def __getattr__(self, name: str) -> Any:
        module = _load_owned_module(self._module_file, (name,))
        return getattr(module, name)


def load_graph_ops(*required_symbols: str):
    if not required_symbols:
        return _OwnedModuleProxy("model/model.py")
    try:
        return _load_owned_module("model/model.py", tuple(required_symbols))
    except RuntimeError:
        proxy = _OwnedModuleProxy("model/model.py")
        for symbol in required_symbols:
            getattr(proxy, symbol)
        return proxy


def load_graph_blocks():
    return _OwnedModuleProxy("model/model.py")
