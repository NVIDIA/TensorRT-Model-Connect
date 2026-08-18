#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate runtime strategy governance against the new builder-based runtime."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_PATH = PROJECT_ROOT / "tests" / "runtime_strategy_matrix.yaml"
DEFAULT_CPP_PATH = PROJECT_ROOT / "src" / "cabi" / "api" / "trtmc_c.cpp"
DEFAULT_BUILDERS_DIR = PROJECT_ROOT / "src" / "runtime" / "builders"
DEFAULT_RUNTIME_REGISTRY_PATH = PROJECT_ROOT / "src" / "runtime" / "registry" / "pipeline_factory.cpp"
DEFAULT_RUNTIME_MODELS_DIR = PROJECT_ROOT / "src" / "runtime" / "models"
DEFAULT_DIFF_CHECKS_DIR = PROJECT_ROOT / "tools" / "diff_framework" / "checks"
DEFAULT_E2E_MODELS_DIR = PROJECT_ROOT / "tests" / "e2e" / "models"
DEFAULT_RUNNERS_DIR = DEFAULT_E2E_MODELS_DIR
DEFAULT_COMPARATORS_DIR = DEFAULT_E2E_MODELS_DIR

_STRING_LITERAL_RE = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')
_RUNTIME_LIKE_RE = re.compile(r"[a-z]+(?:_[a-z0-9]+)+")


def _is_nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _class_basename(class_ref: str) -> str:
    return class_ref.rsplit(".", 1)[-1].strip()


def load_yaml_like(path: Path) -> Any:
    """Load YAML if available, otherwise require JSON-compatible YAML."""
    text = path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        yaml = None

    if yaml is not None:
        return yaml.safe_load(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{path} is not JSON-compatible YAML and PyYAML is unavailable."
        ) from exc


def load_runtime_strategy_matrix(path: Path) -> dict[str, dict[str, Any]]:
    """Load and validate matrix schema."""
    data = load_yaml_like(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected mapping at top level.")

    raw = data.get("runtime_strategies")
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected 'runtime_strategies' mapping.")

    matrix: dict[str, dict[str, Any]] = {}
    for runtime_strategy, entry in raw.items():
        if not _is_nonempty_str(runtime_strategy):
            raise ValueError(f"{path}: runtime strategy keys must be non-empty strings.")
        if not isinstance(entry, dict):
            raise ValueError(
                f"{path}: runtime strategy '{runtime_strategy}' must map to an object."
            )
        matrix[runtime_strategy] = dict(entry)
    return matrix


def _extract_string_literals(text: str) -> set[str]:
    values: set[str] = set()
    for raw in _STRING_LITERAL_RE.findall(text):
        values.add(bytes(raw, "utf-8").decode("unicode_escape"))
    return values


def extract_runtime_strategies_from_cpp(
    path: Path,
    candidate_strategies: Iterable[str] | None = None,
) -> set[str]:
    """Extract runtime strategy keys from a C++ source file.

    The new runtime expresses strategy coverage through `resolve_strategy_family(...)`
    in `trtmc_c.cpp` and the `kStrategies` / comparison literals inside
    `src/runtime/builders/**/*.cpp`. To avoid false positives from unrelated string
    literals like section names, callers should usually pass the expected strategy
    candidate set derived from manifests or the matrix.
    """
    text = path.read_text(encoding="utf-8")
    strings = _extract_string_literals(text)

    if candidate_strategies is not None:
        candidates = set(candidate_strategies)
        return {value for value in strings if value in candidates}

    return {value for value in strings if _RUNTIME_LIKE_RE.fullmatch(value)}


def extract_runtime_strategies_from_cpp_files(
    cpp_paths: Iterable[Path],
    candidate_strategies: Iterable[str] | None = None,
) -> set[str]:
    """Extract runtime_strategy keys from multiple source files."""
    strategies: set[str] = set()
    for file_path in cpp_paths:
        strategies.update(
            extract_runtime_strategies_from_cpp(file_path, candidate_strategies)
        )
    return strategies


def discover_runtime_cpp_files(*, cpp_path: Path, builders_dir: Path) -> list[Path]:
    """Discover runtime sources that define strategy coverage in the new runtime."""
    discovered: list[Path] = []
    if cpp_path.exists():
        discovered.append(cpp_path.resolve())
    if builders_dir.exists():
        discovered.extend(path.resolve() for path in sorted(builders_dir.rglob("*.cpp")))
    return discovered


def discover_runtime_strategy_source_files(
    *,
    cpp_path: Path,
    builders_dir: Path,
    runtime_registry_path: Path,
) -> list[Path]:
    """Discover source files that spell runtime strategy keys."""
    discovered = discover_runtime_cpp_files(cpp_path=cpp_path, builders_dir=builders_dir)
    if runtime_registry_path.exists():
        discovered.append(runtime_registry_path.resolve())
    return discovered


def extract_runtime_strategies_from_model_manifest(path: Path) -> set[str]:
    """Extract runtime strategies from a src/runtime/models/<id>/MODEL.toml."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"runtime_strategies\s*=\s*\[([^\]]*)\]", text)
    if match:
        return set(re.findall(r'"([^"]+)"', match.group(1)))
    match = re.search(r'runtime_strategy\s*=\s*"([^"]+)"', text)
    return {match.group(1)} if match else set()


def extract_runtime_strategies_from_model_manifests(models_dir: Path) -> set[str]:
    """Extract runtime strategies from all runtime model descriptors."""
    strategies: set[str] = set()
    if not models_dir.exists():
        return strategies
    for manifest_path in sorted(models_dir.glob("*/MODEL.toml")):
        strategies.update(extract_runtime_strategies_from_model_manifest(manifest_path))
    return strategies


def _iter_e2e_manifest_paths(models_dir: Path) -> Iterable[Path]:
    if not models_dir.is_dir():
        return
    yield from sorted(models_dir.glob("*.json"))
    yield from sorted(models_dir.glob("*/manifests/*.json"))


def extract_runtime_to_task_strategy_from_manifests(models_dir: Path) -> dict[str, str]:
    """Extract runtime_strategy -> task_strategy declarations from E2E manifests."""
    values: dict[str, set[str]] = {}
    for manifest_path in _iter_e2e_manifest_paths(models_dir):
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        runtime_strategy = raw.get("runtime_strategy")
        task_strategy = raw.get("task_strategy")
        if not isinstance(runtime_strategy, str) or not runtime_strategy:
            continue
        if not isinstance(task_strategy, str) or not task_strategy:
            continue
        values.setdefault(runtime_strategy, set()).add(task_strategy)

    conflicts = {
        runtime: sorted(tasks)
        for runtime, tasks in values.items()
        if len(tasks) > 1
    }
    if conflicts:
        raise ValueError(
            f"{models_dir}: runtime_strategy values map to multiple task_strategy "
            f"values: {conflicts}"
        )
    return {
        runtime: next(iter(tasks))
        for runtime, tasks in sorted(values.items())
    }


def _extract_constant_return(class_node: ast.ClassDef, method_name: str) -> str | None:
    for node in class_node.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != method_name:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Constant):
                if isinstance(sub.value.value, str):
                    return sub.value.value
    return None


def _iter_plugin_python_files(root: Path, kind: str) -> Iterable[Path]:
    """Yield central legacy or model-local E2E plugin files for ``kind``."""
    if (root / kind).is_dir():
        yield from sorted((root / kind).glob("*.py"))
        return
    flat_files = sorted(root.glob("*.py"))
    if flat_files:
        yield from flat_files
        return
    for plugin_dir in sorted(root.glob(f"*/e2e_plugins/{kind}")):
        yield from sorted(plugin_dir.glob("*.py"))


def _extract_class_map_by_method(root: Path, kind: str, method_name: str) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for file_path in _iter_plugin_python_files(root, kind):
        if file_path.name.startswith("_"):
            continue
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            key = _extract_constant_return(node, method_name)
            if key is None:
                continue
            mapping.setdefault(key, set()).add(node.name)
    return mapping


def extract_runner_classes_by_task_strategy(runners_dir: Path) -> dict[str, set[str]]:
    return _extract_class_map_by_method(runners_dir, "runners", "strategy_name")


def extract_comparator_classes_by_task_strategy(
    comparators_dir: Path,
) -> dict[str, set[str]]:
    return _extract_class_map_by_method(comparators_dir, "comparators", "task_strategy")


def extract_diff_framework_check_classes(checks_dir: Path) -> set[str]:
    classes: set[str] = set()
    for file_path in sorted(checks_dir.glob("*.py")):
        if file_path.name.startswith("_"):
            continue
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            has_name = any(
                isinstance(stmt, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "name"
                        for target in stmt.targets)
                for stmt in node.body
            )
            if not has_name:
                continue
            classes.add(node.name)
    return classes


def _append_set_mismatch(
    errors: list[str],
    *,
    left_name: str,
    left_values: set[str],
    right_name: str,
    right_values: set[str],
) -> None:
    missing = sorted(left_values - right_values)
    extra = sorted(right_values - left_values)
    if missing:
        errors.append(
            f"{right_name} missing {len(missing)} runtime strategies from {left_name}: {missing}"
        )
    if extra:
        errors.append(
            f"{right_name} has {len(extra)} extra runtime strategies vs {left_name}: {extra}"
        )


def validate_matrix_data(
    *,
    matrix: dict[str, dict[str, Any]],
    cpp_runtime_strategies: set[str],
    runtime_to_task_strategy: dict[str, str],
    diff_check_classes: set[str],
    runner_classes_by_task: dict[str, set[str]],
    comparator_classes_by_task: dict[str, set[str]],
) -> list[str]:
    """Validate matrix consistency and coverage requirements."""
    errors: list[str] = []

    matrix_strategies = set(matrix.keys())
    manifest_strategies = set(runtime_to_task_strategy.keys())

    missing_matrix_for_sources = sorted(cpp_runtime_strategies - matrix_strategies)
    if missing_matrix_for_sources:
        errors.append(
            "tests/runtime_strategy_matrix.yaml missing runtime strategies from "
            f"runtime sources strategy keys: {missing_matrix_for_sources}"
        )
    missing_matrix_for_manifests = sorted(manifest_strategies - matrix_strategies)
    if missing_matrix_for_manifests:
        errors.append(
            "tests/runtime_strategy_matrix.yaml missing runtime strategies from "
            f"E2E manifests: {missing_matrix_for_manifests}"
        )

    for runtime_strategy in sorted(matrix_strategies):
        entry = matrix[runtime_strategy]
        expected_task = runtime_to_task_strategy.get(runtime_strategy)

        task_strategy = entry.get("task_strategy")
        if expected_task is not None and task_strategy != expected_task:
            errors.append(
                f"{runtime_strategy}: task_strategy='{task_strategy}' "
                f"does not match E2E manifest declaration '{expected_task}'."
            )
        if expected_task is None:
            expected_task = task_strategy

        cli_commands = entry.get("cli_commands")
        if (
            not isinstance(cli_commands, list)
            or not cli_commands
            or not all(_is_nonempty_str(item) for item in cli_commands)
        ):
            errors.append(
                f"{runtime_strategy}: 'cli_commands' must be a non-empty list of strings."
            )

        performance_mode = entry.get("performance_mode")
        if not _is_nonempty_str(performance_mode):
            errors.append(
                f"{runtime_strategy}: 'performance_mode' must be a non-empty string."
            )

        runner_class_ref = entry.get("runner_class")
        if not _is_nonempty_str(runner_class_ref):
            errors.append(f"{runtime_strategy}: 'runner_class' must be a non-empty string.")
        else:
            expected_runner_classes = runner_classes_by_task.get(expected_task, set())
            if not expected_runner_classes:
                errors.append(
                    f"{runtime_strategy}: no runner class found for task_strategy '{expected_task}'."
                )
            elif _class_basename(runner_class_ref) not in expected_runner_classes:
                errors.append(
                    f"{runtime_strategy}: runner_class '{runner_class_ref}' "
                    f"not in discovered runner classes {sorted(expected_runner_classes)}."
                )

        comparator_class_ref = entry.get("comparator_class")
        if comparator_class_ref is not None and not _is_nonempty_str(comparator_class_ref):
            errors.append(
                f"{runtime_strategy}: 'comparator_class' must be a non-empty string when provided."
            )
            comparator_class_ref = None

        if _is_nonempty_str(comparator_class_ref):
            expected_comparator_classes = comparator_classes_by_task.get(expected_task, set())
            if not expected_comparator_classes:
                errors.append(
                    f"{runtime_strategy}: no comparator class found for task_strategy '{expected_task}'."
                )
            elif _class_basename(comparator_class_ref) not in expected_comparator_classes:
                errors.append(
                    f"{runtime_strategy}: comparator_class '{comparator_class_ref}' "
                    f"not in discovered comparator classes {sorted(expected_comparator_classes)}."
                )

        matrix_diff_checks = entry.get("diff_framework_check_classes", [])
        if not isinstance(matrix_diff_checks, list):
            errors.append(
                f"{runtime_strategy}: 'diff_framework_check_classes' must be a list."
            )
            matrix_diff_checks = []
        elif not all(_is_nonempty_str(item) for item in matrix_diff_checks):
            errors.append(
                f"{runtime_strategy}: 'diff_framework_check_classes' entries must be non-empty strings."
            )
            matrix_diff_checks = []

        if len(set(matrix_diff_checks)) != len(matrix_diff_checks):
            errors.append(f"{runtime_strategy}: duplicate entries in diff_framework_check_classes.")

        matrix_diff_check_set = sorted(set(matrix_diff_checks))
        unknown_diff_checks = sorted(set(matrix_diff_check_set) - diff_check_classes)
        if unknown_diff_checks:
            errors.append(
                f"{runtime_strategy}: diff_framework_check_classes references "
                f"unknown check classes {unknown_diff_checks}."
            )

        diff_exemption = entry.get("diff_framework_exemption")
        has_exemption = _is_nonempty_str(diff_exemption)
        if matrix_diff_check_set and has_exemption:
            errors.append(
                f"{runtime_strategy}: diff_framework_exemption must be omitted when checks exist."
            )
        if not matrix_diff_check_set and not has_exemption:
            errors.append(
                f"{runtime_strategy}: requires 'diff_framework_exemption' when no diff-framework checks exist."
            )

        has_comparator = _is_nonempty_str(comparator_class_ref)
        if not has_comparator and not matrix_diff_check_set and not has_exemption:
            errors.append(
                f"{runtime_strategy}: requires comparator/check class coverage or explicit exemption."
            )

    return errors


def validate_matrix_paths(
    *,
    matrix_path: Path = DEFAULT_MATRIX_PATH,
    cpp_path: Path = DEFAULT_CPP_PATH,
    builders_dir: Path = DEFAULT_BUILDERS_DIR,
    runtime_registry_path: Path = DEFAULT_RUNTIME_REGISTRY_PATH,
    runtime_models_dir: Path = DEFAULT_RUNTIME_MODELS_DIR,
    e2e_models_dir: Path = DEFAULT_E2E_MODELS_DIR,
    diff_checks_dir: Path = DEFAULT_DIFF_CHECKS_DIR,
    runners_dir: Path = DEFAULT_RUNNERS_DIR,
    comparators_dir: Path = DEFAULT_COMPARATORS_DIR,
) -> list[str]:
    """Load all sources and validate the runtime strategy matrix."""
    matrix = load_runtime_strategy_matrix(matrix_path)
    runtime_to_task_strategy = extract_runtime_to_task_strategy_from_manifests(
        e2e_models_dir
    )
    candidate_strategies = set(matrix.keys()) | set(runtime_to_task_strategy.keys())

    runtime_cpp_files = discover_runtime_strategy_source_files(
        cpp_path=cpp_path.resolve(),
        builders_dir=builders_dir.resolve(),
        runtime_registry_path=runtime_registry_path.resolve(),
    )
    cpp_runtime_strategies = extract_runtime_strategies_from_cpp_files(
        runtime_cpp_files,
        candidate_strategies,
    )
    cpp_runtime_strategies.update(
        extract_runtime_strategies_from_model_manifests(runtime_models_dir)
    )
    diff_check_classes = extract_diff_framework_check_classes(diff_checks_dir)
    runner_classes_by_task = extract_runner_classes_by_task_strategy(runners_dir)
    comparator_classes_by_task = extract_comparator_classes_by_task_strategy(
        comparators_dir
    )
    return validate_matrix_data(
        matrix=matrix,
        cpp_runtime_strategies=cpp_runtime_strategies,
        runtime_to_task_strategy=runtime_to_task_strategy,
        diff_check_classes=diff_check_classes,
        runner_classes_by_task=runner_classes_by_task,
        comparator_classes_by_task=comparator_classes_by_task,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate tests/runtime_strategy_matrix.yaml consistency."
    )
    parser.add_argument(
        "--matrix",
        type=Path,
        default=DEFAULT_MATRIX_PATH,
        help="Path to runtime strategy matrix YAML file.",
    )
    parser.add_argument(
        "--cpp",
        type=Path,
        default=DEFAULT_CPP_PATH,
        help="Path to src/cabi/api/trtmc_c.cpp.",
    )
    parser.add_argument(
        "--builders-dir",
        type=Path,
        default=DEFAULT_BUILDERS_DIR,
        help="Path to src/runtime/builders directory.",
    )
    parser.add_argument(
        "--runtime-registry",
        type=Path,
        default=DEFAULT_RUNTIME_REGISTRY_PATH,
        help="Path to src/runtime/registry/pipeline_factory.cpp.",
    )
    parser.add_argument(
        "--runtime-models-dir",
        type=Path,
        default=DEFAULT_RUNTIME_MODELS_DIR,
        help="Path to src/runtime/models directory.",
    )
    parser.add_argument(
        "--e2e-models-dir",
        type=Path,
        default=DEFAULT_E2E_MODELS_DIR,
        help="Path to tests/e2e/models directory.",
    )
    parser.add_argument(
        "--diff-checks-dir",
        type=Path,
        default=DEFAULT_DIFF_CHECKS_DIR,
        help="Path to tools/diff_framework/checks directory.",
    )
    parser.add_argument(
        "--runners-dir",
        type=Path,
        default=DEFAULT_RUNNERS_DIR,
        help="Path to tests/e2e/models directory or a legacy runners directory.",
    )
    parser.add_argument(
        "--comparators-dir",
        type=Path,
        default=DEFAULT_COMPARATORS_DIR,
        help="Path to tests/e2e/models directory or a legacy comparators directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        errors = validate_matrix_paths(
            matrix_path=args.matrix,
            cpp_path=args.cpp,
            builders_dir=args.builders_dir,
            runtime_registry_path=args.runtime_registry,
            runtime_models_dir=args.runtime_models_dir,
            e2e_models_dir=args.e2e_models_dir,
            diff_checks_dir=args.diff_checks_dir,
            runners_dir=args.runners_dir,
            comparators_dir=args.comparators_dir,
        )
    except Exception as exc:
        print(f"[runtime-strategy-matrix] ERROR: {exc}", file=sys.stderr)
        return 2

    if errors:
        print("[runtime-strategy-matrix] FAIL")
        for issue in errors:
            print(f" - {issue}")
        return 1

    print(
        "[runtime-strategy-matrix] PASS: matrix is consistent with the new runtime "
        "entrypoint, builder strategy coverage, E2E manifests, and diff-framework checks."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
