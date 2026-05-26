#!/usr/bin/env python3
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
DEFAULT_BUILDERS_DIR = PROJECT_ROOT / "src" / "runtime" / "models"
DEFAULT_REGISTRY_DIR = PROJECT_ROOT / "src" / "runtime" / "registry"
DEFAULT_ENGINE_DEFS_DIR = (
    PROJECT_ROOT / "python" / "tensorrt_model_connect" / "engine_defs"
)
DEFAULT_CONTRACTS_PATH = PROJECT_ROOT / "tests" / "e2e_harness" / "contracts.py"
DEFAULT_DIFF_CHECKS_DIR = PROJECT_ROOT / "tools" / "diff_framework" / "checks"
DEFAULT_RUNNERS_DIR = PROJECT_ROOT / "tests" / "e2e_harness" / "runners"
DEFAULT_COMPARATORS_DIR = PROJECT_ROOT / "tests" / "e2e_harness" / "comparators"

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
        # Runtime strategy keys are plain ASCII identifiers. Keeping the raw
        # literal avoids unicode_escape warnings on unrelated regex strings.
        values.add(raw)
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
    candidate set derived from the contracts or matrix.
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
    """Extract runtime_strategy keys from multiple C++ files."""
    strategies: set[str] = set()
    for file_path in cpp_paths:
        strategies.update(
            extract_runtime_strategies_from_cpp(file_path, candidate_strategies)
        )
    return strategies


def _discover_source_files(directory: Path, suffixes: set[str]) -> list[Path]:
    if not directory.exists():
        return []
    return [
        path.resolve()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix in suffixes
    ]


def discover_runtime_cpp_files(
    *,
    cpp_path: Path,
    builders_dir: Path,
    registry_dir: Path = DEFAULT_REGISTRY_DIR,
    engine_defs_dir: Path = DEFAULT_ENGINE_DEFS_DIR,
) -> list[Path]:
    """Discover runtime sources that define runtime_strategy coverage."""
    discovered: list[Path] = []
    if cpp_path.exists():
        discovered.append(cpp_path.resolve())
    discovered.extend(
        _discover_source_files(builders_dir, {".cpp", ".h", ".toml"})
    )
    discovered.extend(_discover_source_files(registry_dir, {".cpp", ".h"}))
    discovered.extend(_discover_source_files(engine_defs_dir, {".py"}))
    return discovered


def extract_runtime_to_task_strategy(path: Path) -> dict[str, str]:
    """Extract RUNTIME_TO_TASK_STRATEGY literal mapping from contracts.py."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in tree.body:
        value: Any | None = None
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "RUNTIME_TO_TASK_STRATEGY":
                    value = ast.literal_eval(node.value)
                    break
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "RUNTIME_TO_TASK_STRATEGY":
                value = ast.literal_eval(node.value)

        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"{path}: RUNTIME_TO_TASK_STRATEGY must be a dict literal.")

        mapping: dict[str, str] = {}
        for key, mapped in value.items():
            if not isinstance(key, str) or not isinstance(mapped, str):
                raise ValueError(
                    f"{path}: RUNTIME_TO_TASK_STRATEGY keys/values must be strings."
                )
            mapping[key] = mapped
        return mapping

    raise ValueError(f"{path}: RUNTIME_TO_TASK_STRATEGY not found.")


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


def _extract_class_map_by_method(directory: Path, method_name: str) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for file_path in sorted(directory.glob("*.py")):
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
    return _extract_class_map_by_method(runners_dir, "strategy_name")


def extract_comparator_classes_by_task_strategy(
    comparators_dir: Path,
) -> dict[str, set[str]]:
    return _extract_class_map_by_method(comparators_dir, "task_strategy")


def extract_diff_framework_checks(checks_dir: Path) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for file_path in sorted(checks_dir.glob("*.py")):
        if file_path.name.startswith("_"):
            continue
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            runtime_strategies: list[str] | None = None
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == "runtime_strategies":
                        parsed = ast.literal_eval(stmt.value)
                        if not isinstance(parsed, list):
                            raise ValueError(
                                f"{file_path}: class {node.name} runtime_strategies must be a list literal."
                            )
                        if not all(isinstance(item, str) for item in parsed):
                            raise ValueError(
                                f"{file_path}: class {node.name} runtime_strategies must be strings."
                            )
                        runtime_strategies = parsed
                        break
            if runtime_strategies is None:
                continue
            for strategy in runtime_strategies:
                mapping.setdefault(strategy, set()).add(node.name)
    return mapping


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
    diff_checks_by_strategy: dict[str, set[str]],
    runner_classes_by_task: dict[str, set[str]],
    comparator_classes_by_task: dict[str, set[str]],
) -> list[str]:
    """Validate matrix consistency and coverage requirements."""
    errors: list[str] = []

    matrix_strategies = set(matrix.keys())
    contracts_strategies = set(runtime_to_task_strategy.keys())

    _append_set_mismatch(
        errors,
        left_name="contracts.py RUNTIME_TO_TASK_STRATEGY",
        left_values=contracts_strategies,
        right_name="runtime builder sources strategy keys",
        right_values=cpp_runtime_strategies,
    )
    _append_set_mismatch(
        errors,
        left_name="runtime builder sources strategy keys",
        left_values=cpp_runtime_strategies,
        right_name="tests/runtime_strategy_matrix.yaml",
        right_values=matrix_strategies,
    )
    _append_set_mismatch(
        errors,
        left_name="contracts.py RUNTIME_TO_TASK_STRATEGY",
        left_values=contracts_strategies,
        right_name="tests/runtime_strategy_matrix.yaml",
        right_values=matrix_strategies,
    )

    wildcard_diff_checks = diff_checks_by_strategy.get("*", set())
    for runtime_strategy in sorted(matrix_strategies):
        entry = matrix[runtime_strategy]
        expected_task = runtime_to_task_strategy.get(runtime_strategy)

        if expected_task is None:
            errors.append(
                f"Matrix entry '{runtime_strategy}' is not present in RUNTIME_TO_TASK_STRATEGY."
            )
            continue

        task_strategy = entry.get("task_strategy")
        if task_strategy != expected_task:
            errors.append(
                f"{runtime_strategy}: task_strategy='{task_strategy}' "
                f"does not match contracts mapping '{expected_task}'."
            )

        cli_commands = entry.get("cli_commands")
        if (
            not isinstance(cli_commands, list)
            or not cli_commands
            or not all(_is_nonempty_str(item) for item in cli_commands)
        ):
            errors.append(
                f"{runtime_strategy}: 'cli_commands' must be a non-empty list of strings."
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

        actual_diff_checks = sorted(
            diff_checks_by_strategy.get(runtime_strategy, set()) | wildcard_diff_checks
        )
        matrix_diff_check_set = sorted(set(matrix_diff_checks))
        if matrix_diff_check_set != actual_diff_checks:
            errors.append(
                f"{runtime_strategy}: diff_framework_check_classes={matrix_diff_check_set} "
                f"does not match discovered checks={actual_diff_checks}."
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
    registry_dir: Path = DEFAULT_REGISTRY_DIR,
    engine_defs_dir: Path = DEFAULT_ENGINE_DEFS_DIR,
    contracts_path: Path = DEFAULT_CONTRACTS_PATH,
    diff_checks_dir: Path = DEFAULT_DIFF_CHECKS_DIR,
    runners_dir: Path = DEFAULT_RUNNERS_DIR,
    comparators_dir: Path = DEFAULT_COMPARATORS_DIR,
) -> list[str]:
    """Load all sources and validate the runtime strategy matrix."""
    matrix = load_runtime_strategy_matrix(matrix_path)
    runtime_to_task_strategy = extract_runtime_to_task_strategy(contracts_path)
    candidate_strategies = set(matrix.keys()) | set(runtime_to_task_strategy.keys())

    runtime_cpp_files = discover_runtime_cpp_files(
        cpp_path=cpp_path.resolve(),
        builders_dir=builders_dir.resolve(),
        registry_dir=registry_dir.resolve(),
        engine_defs_dir=engine_defs_dir.resolve(),
    )
    cpp_runtime_strategies = extract_runtime_strategies_from_cpp_files(
        runtime_cpp_files,
        candidate_strategies,
    )
    diff_checks_by_strategy = extract_diff_framework_checks(diff_checks_dir)
    runner_classes_by_task = extract_runner_classes_by_task_strategy(runners_dir)
    comparator_classes_by_task = extract_comparator_classes_by_task_strategy(
        comparators_dir
    )
    return validate_matrix_data(
        matrix=matrix,
        cpp_runtime_strategies=cpp_runtime_strategies,
        runtime_to_task_strategy=runtime_to_task_strategy,
        diff_checks_by_strategy=diff_checks_by_strategy,
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
        help="Path to runtime model source directory.",
    )
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=DEFAULT_REGISTRY_DIR,
        help="Path to runtime registry source directory.",
    )
    parser.add_argument(
        "--engine-defs-dir",
        type=Path,
        default=DEFAULT_ENGINE_DEFS_DIR,
        help="Path to Python engine definition source directory.",
    )
    parser.add_argument(
        "--contracts",
        type=Path,
        default=DEFAULT_CONTRACTS_PATH,
        help="Path to tests/e2e_harness/contracts.py.",
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
        help="Path to tests/e2e_harness/runners directory.",
    )
    parser.add_argument(
        "--comparators-dir",
        type=Path,
        default=DEFAULT_COMPARATORS_DIR,
        help="Path to tests/e2e_harness/comparators directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        errors = validate_matrix_paths(
            matrix_path=args.matrix,
            cpp_path=args.cpp,
            builders_dir=args.builders_dir,
            registry_dir=args.registry_dir,
            engine_defs_dir=args.engine_defs_dir,
            contracts_path=args.contracts,
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
        "entrypoint, builder strategy coverage, contracts mapping, and diff-framework checks."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
