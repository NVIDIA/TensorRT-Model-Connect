# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check registry — register and look up diff test checks."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


_REGISTRY: list[type] = []
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_MATRIX_PATH = _PROJECT_ROOT / "tests" / "runtime_strategy_matrix.yaml"


def register(cls):
    """Class decorator — registers a DiffTest implementation."""
    _REGISTRY.append(cls)
    return cls


def get_all_tests() -> list[type]:
    """Return all registered test classes."""
    return list(_REGISTRY)


@lru_cache(maxsize=1)
def _strategy_check_class_names() -> dict[str, tuple[str, ...]]:
    """Return runtime_strategy -> diff check class names from matrix metadata."""
    try:
        text = _MATRIX_PATH.read_text(encoding="utf-8")
    except Exception:
        return {}
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]
        except Exception:
            return {}
        raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        return {}
    entries = raw.get("runtime_strategies", {})
    if not isinstance(entries, dict):
        return {}

    result: dict[str, tuple[str, ...]] = {}
    for strategy, entry in entries.items():
        if not isinstance(strategy, str) or not isinstance(entry, dict):
            continue
        checks = entry.get("diff_framework_check_classes", [])
        if not isinstance(checks, list):
            continue
        names = tuple(name for name in checks if isinstance(name, str) and name)
        result[strategy] = names
    return result


def _class_name_index() -> dict[str, type]:
    return {cls.__name__: cls for cls in _REGISTRY}


def get_tests_for_strategy(strategy: str) -> list[type]:
    """Return test classes that apply to the given runtime_strategy."""
    matrix_names = _strategy_check_class_names().get(strategy)
    if matrix_names is not None:
        by_name = _class_name_index()
        return [by_name[name] for name in matrix_names if name in by_name]

    result = []
    for cls in _REGISTRY:
        strategies = getattr(cls, "runtime_strategies", [])
        if "*" in strategies or strategy in strategies:
            result.append(cls)
    return result


def get_strategies_for_test(cls: type) -> list[str]:
    """Return runtime strategies that matrix metadata assigns to ``cls``."""
    class_name = cls.__name__
    matrix_matches = sorted(
        strategy
        for strategy, checks in _strategy_check_class_names().items()
        if class_name in checks
    )
    if matrix_matches:
        return matrix_matches
    return list(getattr(cls, "runtime_strategies", []))


def get_test_by_name(name: str) -> type | None:
    """Look up a test class by name. Returns None if not found."""
    for cls in _REGISTRY:
        if cls.name == name:
            return cls
    return None
