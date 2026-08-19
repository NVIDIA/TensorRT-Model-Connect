# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check registry — register and look up diff test checks."""

from __future__ import annotations

from functools import lru_cache

from tests.e2e_harness.runtime_strategy_metadata import load_runtime_strategy_catalog


_REGISTRY: list[type] = []


def register(cls):
    """Class decorator — registers a DiffTest implementation."""
    _REGISTRY.append(cls)
    return cls


def get_all_tests() -> list[type]:
    """Return all registered test classes."""
    return list(_REGISTRY)


@lru_cache(maxsize=1)
def _strategy_check_class_names() -> dict[str, tuple[str, ...]]:
    """Return runtime_strategy -> owner-local diff check class names."""
    return {
        strategy: metadata.diff_framework_check_classes
        for strategy, metadata in load_runtime_strategy_catalog().items()
    }


def _class_name_index() -> dict[str, type]:
    return {cls.__name__: cls for cls in _REGISTRY}


def get_tests_for_strategy(strategy: str) -> list[type]:
    """Return test classes that apply to the given runtime_strategy."""
    owner_names = _strategy_check_class_names().get(strategy)
    if owner_names is None:
        raise ValueError(f"unknown runtime_strategy {strategy!r}")
    by_name = _class_name_index()
    missing = sorted(set(owner_names) - by_name.keys())
    if missing:
        raise ValueError(
            f"runtime_strategy {strategy!r} references unknown diff checks {missing}"
        )
    return [by_name[name] for name in owner_names]


def get_strategies_for_test(cls: type) -> list[str]:
    """Return runtime strategies whose owner descriptor assigns ``cls``."""
    class_name = cls.__name__
    owner_matches = sorted(
        strategy
        for strategy, checks in _strategy_check_class_names().items()
        if class_name in checks
    )
    return owner_matches


def get_test_by_name(name: str) -> type | None:
    """Look up a test class by name. Returns None if not found."""
    for cls in _REGISTRY:
        if cls.name == name:
            return cls
    return None
