# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Derive runtime-strategy metadata from unified model owners."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .task_defaults import PERFORMANCE_MODES, task_defaults


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MODELS_DIR = _PROJECT_ROOT / "python" / "tensorrt_model_connect" / "models"
_RETIRED_CONTROL_FIELDS = frozenset(
    {
        "legacy_runtime_strategy_aliases",
        "new_runtime_guard_strategies",
        "runtime_library",
        "task_strategy",
    }
)


@dataclass(frozen=True)
class RuntimeStrategyMetadata:
    """One owner strategy joined to its exact task and shared defaults."""

    owner: str
    task_strategy: str
    performance_mode: str
    cli_commands: tuple[str, ...]
    cli_exemption: str | None
    diff_framework_check_classes: tuple[str, ...]


def _nonempty_strings(value: object, *, field: str, path: Path) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path}: {field} must be a non-empty list")
    items = tuple(str(item) for item in value if isinstance(item, str) and item)
    if len(items) != len(value) or len(set(items)) != len(items):
        raise ValueError(f"{path}: {field} must contain unique non-empty strings")
    return items


def _optional_strings(value: object, *, field: str, path: Path) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{path}: {field} must be a list")
    items = tuple(str(item) for item in value if isinstance(item, str) and item)
    if len(items) != len(value) or len(set(items)) != len(items):
        raise ValueError(f"{path}: {field} must contain unique non-empty strings")
    return items


def _owner_manifest_paths(owner: Path, descriptor: dict[str, object]) -> tuple[Path, ...]:
    descriptor_path = owner / "MODEL.toml"
    declared = _nonempty_strings(
        descriptor.get("test_manifests"), field="test_manifests", path=descriptor_path
    )
    owner_root = owner.resolve()
    paths: list[Path] = []
    for relative_text in declared:
        relative = Path(relative_text)
        path = (owner / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(owner_root):
            raise ValueError(f"{descriptor_path}: test manifest escapes its owner: {relative_text}")
        if relative.parts[:2] != ("tests", "manifests"):
            raise ValueError(
                f"{descriptor_path}: test manifest must live below tests/manifests: {relative_text}"
            )
        if not path.is_file():
            raise ValueError(f"{descriptor_path}: missing test manifest {relative_text}")
        paths.append(path)

    discovered = {path.resolve() for path in (owner / "tests" / "manifests").glob("*.json")}
    if set(paths) != discovered:
        missing = sorted(str(path.relative_to(owner_root)) for path in discovered - set(paths))
        extra = sorted(str(path.relative_to(owner_root)) for path in set(paths) - discovered)
        raise ValueError(
            f"{descriptor_path}: test_manifests does not match owner files; "
            f"undeclared={missing}, nonexistent={extra}"
        )
    return tuple(paths)


def _validate_performance_sidecar(owner: Path, strategies: tuple[str, ...]) -> None:
    path = owner / "tests" / "perf_validation.json"
    if not path.is_file():
        return
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    entries = raw.get("models", raw) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a list or object with models")
    owned = set(strategies)
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: entry {index} must be an object")
        pipeline_type = entry.get("pipeline_type")
        if pipeline_type not in owned:
            raise ValueError(
                f"{path}: entry {index} pipeline_type {pipeline_type!r} is not "
                f"declared by owner {owner.name!r}"
            )


@lru_cache(maxsize=None)
def _load_runtime_strategy_catalog(models_dir_text: str) -> dict[str, RuntimeStrategyMetadata]:
    models_dir = Path(models_dir_text)
    if not models_dir.is_dir():
        raise ValueError(f"model owner root does not exist: {models_dir}")

    catalog: dict[str, RuntimeStrategyMetadata] = {}
    descriptor_paths = sorted(models_dir.glob("*/MODEL.toml"))
    if not descriptor_paths:
        raise ValueError(f"no model owner descriptors below {models_dir}")

    for descriptor_path in descriptor_paths:
        with descriptor_path.open("rb") as stream:
            descriptor = tomllib.load(stream)
        owner_dir = descriptor_path.parent
        owner = descriptor.get("id")
        if owner != owner_dir.name:
            raise ValueError(
                f"{descriptor_path}: id {owner!r} must match owner directory {owner_dir.name!r}"
            )
        retired = sorted(_RETIRED_CONTROL_FIELDS & descriptor.keys())
        if retired:
            raise ValueError(f"{descriptor_path}: retired control fields {retired}")

        strategies = _nonempty_strings(
            descriptor.get("runtime_strategies"),
            field="runtime_strategies",
            path=descriptor_path,
        )
        performance_override = descriptor.get("performance_mode")
        if performance_override is not None and (
            not isinstance(performance_override, str)
            or performance_override not in PERFORMANCE_MODES
        ):
            raise ValueError(
                f"{descriptor_path}: performance_mode must be one of {sorted(PERFORMANCE_MODES)}"
            )
        cli_exemption = descriptor.get("cli_exemption")
        if cli_exemption is not None and (
            not isinstance(cli_exemption, str) or not cli_exemption.strip()
        ):
            raise ValueError(f"{descriptor_path}: cli_exemption must be a non-empty string")
        diff_checks = _optional_strings(
            descriptor.get("diff_framework_check_classes"),
            field="diff_framework_check_classes",
            path=descriptor_path,
        )
        _validate_performance_sidecar(owner_dir, strategies)

        tasks_by_strategy: dict[str, set[str]] = {strategy: set() for strategy in strategies}
        for manifest_path in _owner_manifest_paths(owner_dir, descriptor):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest_path}: invalid JSON: {exc}") from exc
            if not isinstance(manifest, dict):
                raise ValueError(f"{manifest_path}: expected a JSON object")
            if manifest.get("family") != owner:
                raise ValueError(
                    f"{manifest_path}: family {manifest.get('family')!r} must equal owner {owner!r}"
                )
            runtime_strategy = manifest.get("runtime_strategy")
            if runtime_strategy not in tasks_by_strategy:
                raise ValueError(
                    f"{manifest_path}: runtime_strategy {runtime_strategy!r} is not "
                    f"declared by owner {owner!r}"
                )
            task_strategy = manifest.get("task_strategy")
            if not isinstance(task_strategy, str) or not task_strategy:
                raise ValueError(f"{manifest_path}: task_strategy is required")
            task_defaults(task_strategy)
            tasks_by_strategy[runtime_strategy].add(task_strategy)

        for runtime_strategy, tasks in tasks_by_strategy.items():
            if len(tasks) != 1:
                raise ValueError(
                    f"{descriptor_path}: runtime_strategy {runtime_strategy!r} must map "
                    f"to exactly one task_strategy, found {sorted(tasks)}"
                )
            if runtime_strategy in catalog:
                raise ValueError(
                    f"runtime_strategy {runtime_strategy!r} is declared by both "
                    f"{catalog[runtime_strategy].owner!r} and {owner!r}"
                )
            task_strategy = next(iter(tasks))
            defaults = task_defaults(task_strategy)
            if performance_override == defaults.performance_mode:
                raise ValueError(
                    f"{descriptor_path}: redundant performance_mode override for "
                    f"task_strategy {task_strategy!r}"
                )
            catalog[runtime_strategy] = RuntimeStrategyMetadata(
                owner=owner,
                task_strategy=task_strategy,
                performance_mode=performance_override or defaults.performance_mode,
                cli_commands=() if cli_exemption else defaults.cli_commands,
                cli_exemption=cli_exemption,
                diff_framework_check_classes=diff_checks,
            )

    return catalog


def load_runtime_strategy_catalog(
    models_dir: Path | None = None,
) -> dict[str, RuntimeStrategyMetadata]:
    """Return the exact owner-derived runtime strategy catalog."""
    root = (models_dir or _DEFAULT_MODELS_DIR).resolve()
    return dict(_load_runtime_strategy_catalog(str(root)))


def clear_runtime_strategy_metadata_cache() -> None:
    """Clear cached source metadata for tests that rewrite a fixture tree."""
    _load_runtime_strategy_catalog.cache_clear()


def runtime_strategy_requires_new_runtime_guard(
    runtime_strategy: str,
    models_dir: Path | None = None,
) -> bool:
    """Every declared native strategy must confirm the current runtime path."""
    return runtime_strategy in load_runtime_strategy_catalog(models_dir)


def runtime_strategy_task_strategy(
    runtime_strategy: str,
    models_dir: Path | None = None,
) -> str | None:
    """Return the exact task declared by the owner manifests."""
    metadata = load_runtime_strategy_catalog(models_dir).get(runtime_strategy)
    return metadata.task_strategy if metadata is not None else None


def runtime_strategy_performance_mode(
    runtime_strategy: str,
    models_dir: Path | None = None,
) -> str:
    """Return owner override or the shared task performance default."""
    try:
        return load_runtime_strategy_catalog(models_dir)[runtime_strategy].performance_mode
    except KeyError as exc:
        raise ValueError(f"unknown runtime_strategy {runtime_strategy!r}") from exc


def runtime_strategy_cli_commands(
    runtime_strategy: str,
    models_dir: Path | None = None,
) -> tuple[str, ...]:
    """Return CLI commands inherited from the shared task contract."""
    try:
        return load_runtime_strategy_catalog(models_dir)[runtime_strategy].cli_commands
    except KeyError as exc:
        raise ValueError(f"unknown runtime_strategy {runtime_strategy!r}") from exc


def runtime_strategy_diff_check_classes(
    runtime_strategy: str,
    models_dir: Path | None = None,
) -> tuple[str, ...]:
    """Return owner-local diff-framework applicability metadata."""
    try:
        return load_runtime_strategy_catalog(models_dir)[
            runtime_strategy
        ].diff_framework_check_classes
    except KeyError as exc:
        raise ValueError(f"unknown runtime_strategy {runtime_strategy!r}") from exc
