# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime strategy metadata for generic E2E harness decisions."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


_DEFAULT_MATRIX_PATH = Path(__file__).resolve().parent.parent / "runtime_strategy_matrix.yaml"


def _load_yaml_like(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        yaml = None

    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected top-level object")
    return data


@lru_cache(maxsize=None)
def _new_runtime_guard_strategies(matrix_path: str) -> frozenset[str]:
    path = Path(matrix_path)
    data = _load_yaml_like(path)
    raw = data.get("new_runtime_guard_strategies", [])
    if not isinstance(raw, list) or not all(isinstance(item, str) and item for item in raw):
        raise ValueError(
            f"{path}: new_runtime_guard_strategies must be a list of non-empty strings"
        )
    return frozenset(raw)


@lru_cache(maxsize=None)
def _runtime_strategy_entries(matrix_path: str) -> dict[str, dict[str, Any]]:
    path = Path(matrix_path)
    data = _load_yaml_like(path)
    raw = data.get("runtime_strategies", {})
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: runtime_strategies must be a mapping")

    entries: dict[str, dict[str, Any]] = {}
    for strategy, entry in raw.items():
        if not isinstance(strategy, str) or not strategy:
            raise ValueError(f"{path}: runtime strategy keys must be non-empty strings")
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: runtime strategy {strategy!r} must be a mapping")
        entries[strategy] = dict(entry)
    return entries


def runtime_strategy_requires_new_runtime_guard(
    runtime_strategy: str,
    matrix_path: Path | None = None,
) -> bool:
    """Return whether a runtime strategy should assert the new runtime path."""
    path = matrix_path or _DEFAULT_MATRIX_PATH
    return runtime_strategy in _new_runtime_guard_strategies(str(path))


def runtime_strategy_task_strategy(
    runtime_strategy: str,
    matrix_path: Path | None = None,
) -> str | None:
    """Return the declared task strategy for a runtime strategy."""
    path = matrix_path or _DEFAULT_MATRIX_PATH
    entry = _runtime_strategy_entries(str(path)).get(runtime_strategy)
    if entry is None:
        return None
    task_strategy = entry.get("task_strategy")
    return task_strategy if isinstance(task_strategy, str) and task_strategy else None


def runtime_strategy_performance_mode(
    runtime_strategy: str,
    matrix_path: Path | None = None,
    *,
    default: str = "decode",
) -> str:
    """Return the model-declared performance mode for a runtime strategy."""
    path = matrix_path or _DEFAULT_MATRIX_PATH
    entry = _runtime_strategy_entries(str(path)).get(runtime_strategy)
    if entry is None:
        return default
    explicit_mode = entry.get("performance_mode")
    if isinstance(explicit_mode, str) and explicit_mode:
        return explicit_mode
    return default
