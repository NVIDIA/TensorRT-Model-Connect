#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the owner-derived runtime-strategy control plane."""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterable
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.e2e_harness.runtime_strategy_metadata import (  # noqa: E402
    RuntimeStrategyMetadata,
    clear_runtime_strategy_metadata_cache,
    load_runtime_strategy_catalog,
)


DEFAULT_MODELS_DIR = PROJECT_ROOT / "python" / "tensorrt_model_connect" / "models"
DEFAULT_DIFF_CHECKS_DIR = PROJECT_ROOT / "tools" / "diff_framework" / "checks"


def _extract_constant_return(class_node: ast.ClassDef, method_name: str) -> str | None:
    for node in class_node.body:
        if not isinstance(node, ast.FunctionDef) or node.name != method_name:
            continue
        for subnode in ast.walk(node):
            if (
                isinstance(subnode, ast.Return)
                and isinstance(subnode.value, ast.Constant)
                and isinstance(subnode.value.value, str)
            ):
                return subnode.value.value
    return None


def _iter_owner_plugin_files(owner: Path, kind: str) -> Iterable[Path]:
    plugin_root = owner / "tests" / "e2e_plugins"
    if not plugin_root.is_dir():
        return
    yield from sorted(
        path
        for path in plugin_root.glob("*.py")
        if not path.name.startswith("_") and path.name != "__init__.py"
    )
    yield from sorted(
        path
        for path in (plugin_root / kind).glob("*.py")
        if not path.name.startswith("_") and path.name != "__init__.py"
    )


def extract_owner_classes_by_task(
    models_dir: Path,
    *,
    kind: str,
    method_name: str,
) -> dict[str, dict[str, set[str]]]:
    """Return owner -> task_strategy -> concrete plugin class names."""
    result: dict[str, dict[str, set[str]]] = {}
    for descriptor in sorted(models_dir.glob("*/MODEL.toml")):
        owner = descriptor.parent.name
        task_classes: dict[str, set[str]] = {}
        for file_path in _iter_owner_plugin_files(descriptor.parent, kind):
            tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                task_strategy = _extract_constant_return(node, method_name)
                if task_strategy is None:
                    continue
                task_classes.setdefault(task_strategy, set()).add(node.name)
        result[owner] = task_classes
    return result


def extract_owner_runner_classes_by_task(
    models_dir: Path,
) -> dict[str, dict[str, set[str]]]:
    return extract_owner_classes_by_task(models_dir, kind="runners", method_name="strategy_name")


def extract_owner_comparator_classes_by_task(
    models_dir: Path,
) -> dict[str, dict[str, set[str]]]:
    return extract_owner_classes_by_task(
        models_dir, kind="comparators", method_name="task_strategy"
    )


def extract_diff_framework_check_classes(checks_dir: Path) -> set[str]:
    """Return executable diff-framework classes, not an applicability list."""
    classes: set[str] = set()
    for file_path in sorted(checks_dir.glob("*.py")):
        if file_path.name.startswith("_"):
            continue
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            has_name = any(
                isinstance(statement, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "name"
                    for target in statement.targets
                )
                for statement in node.body
            )
            if has_name:
                classes.add(node.name)
    return classes


def validate_control_plane_data(
    *,
    catalog: dict[str, RuntimeStrategyMetadata],
    runner_classes_by_owner_task: dict[str, dict[str, set[str]]],
    comparator_classes_by_owner_task: dict[str, dict[str, set[str]]],
    diff_check_classes: set[str],
) -> list[str]:
    """Validate executable coverage without a runtime- or family-keyed matrix."""
    errors: list[str] = []
    if not catalog:
        return ["runtime strategy catalog is empty"]

    for runtime_strategy, metadata in sorted(catalog.items()):
        owner_runners = runner_classes_by_owner_task.get(metadata.owner, {})
        runners = owner_runners.get(metadata.task_strategy, set())
        if not runners:
            errors.append(
                f"{runtime_strategy}: owner {metadata.owner!r} has no runner class "
                f"for task_strategy {metadata.task_strategy!r}"
            )

        owner_comparators = comparator_classes_by_owner_task.get(metadata.owner, {})
        comparators = owner_comparators.get(metadata.task_strategy, set())
        if not comparators:
            errors.append(
                f"{runtime_strategy}: owner {metadata.owner!r} has no comparator class "
                f"for task_strategy {metadata.task_strategy!r}"
            )

        cli_commands = metadata.cli_commands
        cli_exemption = getattr(metadata, "cli_exemption", None)
        if not isinstance(cli_commands, tuple) or any(
            not isinstance(command, str) or not command.strip()
            for command in cli_commands
        ):
            errors.append(
                f"{runtime_strategy}: cli_commands must contain only non-empty strings"
            )
            cli_commands = ()
        if cli_exemption is not None and (
            not isinstance(cli_exemption, str) or not cli_exemption.strip()
        ):
            errors.append(
                f"{runtime_strategy}: cli_exemption must be a non-empty string when provided"
            )
            cli_exemption = None
        if cli_commands and cli_exemption:
            errors.append(
                f"{runtime_strategy}: cli_exemption must be omitted when CLI commands exist"
            )
        if not cli_commands and not cli_exemption:
            errors.append(
                f"{runtime_strategy}: cli_exemption is required when no CLI command exists"
            )

        unknown_checks = sorted(set(metadata.diff_framework_check_classes) - diff_check_classes)
        if unknown_checks:
            errors.append(
                f"{runtime_strategy}: owner descriptor references unknown diff-framework "
                f"classes {unknown_checks}"
            )

    return errors


def validate_control_plane_paths(
    *,
    models_dir: Path = DEFAULT_MODELS_DIR,
    diff_checks_dir: Path = DEFAULT_DIFF_CHECKS_DIR,
) -> list[str]:
    """Load unified owners and validate their derived strategy contracts."""
    clear_runtime_strategy_metadata_cache()
    catalog = load_runtime_strategy_catalog(models_dir)
    return validate_control_plane_data(
        catalog=catalog,
        runner_classes_by_owner_task=extract_owner_runner_classes_by_task(models_dir),
        comparator_classes_by_owner_task=extract_owner_comparator_classes_by_task(models_dir),
        diff_check_classes=extract_diff_framework_check_classes(diff_checks_dir),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate runtime strategies derived from unified model owners and "
            "shared task defaults."
        )
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Path to python/tensorrt_model_connect/models.",
    )
    parser.add_argument(
        "--diff-checks-dir",
        type=Path,
        default=DEFAULT_DIFF_CHECKS_DIR,
        help="Path to executable diff-framework checks.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        errors = validate_control_plane_paths(
            models_dir=args.models_dir.resolve(),
            diff_checks_dir=args.diff_checks_dir.resolve(),
        )
        strategy_count = len(load_runtime_strategy_catalog(args.models_dir.resolve()))
    except Exception as exc:
        print(f"[runtime-strategy-control-plane] ERROR: {exc}", file=sys.stderr)
        return 2

    if errors:
        print("[runtime-strategy-control-plane] FAIL")
        for issue in errors:
            print(f" - {issue}")
        return 1

    print(
        "[runtime-strategy-control-plane] PASS: "
        f"{strategy_count} owner strategies resolve to owner-local manifests, "
        "runner/comparator plugins, shared task defaults, and local diff checks."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
