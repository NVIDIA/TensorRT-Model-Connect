#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate runtime strategy governance against runtime, CLI, and E2E code."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_PATH = PROJECT_ROOT / "tests" / "runtime_strategy_matrix.yaml"
DEFAULT_CPP_PATH = PROJECT_ROOT / "src" / "cabi" / "api" / "trtmc_c.cpp"
DEFAULT_BUILDERS_DIR = PROJECT_ROOT / "src" / "runtime" / "builders"
DEFAULT_RUNTIME_REGISTRY_PATH = (
    PROJECT_ROOT / "src" / "runtime" / "registry" / "pipeline_factory.cpp"
)
DEFAULT_RUNTIME_MODELS_DIR = PROJECT_ROOT / "src" / "runtime" / "models"
DEFAULT_CLI_ARGS_PATH = PROJECT_ROOT / "src" / "cli" / "args.cpp"
DEFAULT_TORCHTRT_STRATEGIES_DIR = (
    PROJECT_ROOT / "python" / "tensorrt_model_connect" / "engine_defs" / "torch_trt" / "strategies"
)
DEFAULT_DIFF_CHECKS_DIR = PROJECT_ROOT / "tools" / "diff_framework" / "checks"
DEFAULT_E2E_MODELS_DIR = PROJECT_ROOT / "tests" / "e2e" / "models"
DEFAULT_RUNNERS_DIR = DEFAULT_E2E_MODELS_DIR
DEFAULT_COMPARATORS_DIR = DEFAULT_E2E_MODELS_DIR

_STRING_LITERAL_RE = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')
_RUNTIME_LIKE_RE = re.compile(r"[a-z]+(?:_[a-z0-9]+)+")
_KNOWN_CLI_COMMANDS_RE = re.compile(
    r"static\s+const\s+char\s*\*\s*known_cmds\s*\[\s*\]\s*=\s*\{(?P<body>.*?)\};",
    re.DOTALL,
)


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
        raise ValueError(f"{path} is not JSON-compatible YAML and PyYAML is unavailable.") from exc


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
        strategies.update(extract_runtime_strategies_from_cpp(file_path, candidate_strategies))
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
    torchtrt_strategies_dir: Path,
) -> list[Path]:
    """Discover source files that spell runtime strategy keys."""
    discovered = discover_runtime_cpp_files(cpp_path=cpp_path, builders_dir=builders_dir)
    if runtime_registry_path.exists():
        discovered.append(runtime_registry_path.resolve())
    if torchtrt_strategies_dir.exists():
        discovered.extend(
            path.resolve()
            for path in sorted(torchtrt_strategies_dir.glob("*.py"))
            if not path.name.startswith("_")
        )
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


def extract_native_cli_commands(path: Path) -> set[str]:
    """Extract inference subcommands accepted by the native CLI parser."""
    text = path.read_text(encoding="utf-8")
    match = _KNOWN_CLI_COMMANDS_RE.search(text)
    if match is None:
        raise ValueError(f"{path}: could not find the native CLI known_cmds table.")
    commands = _extract_string_literals(match.group("body"))
    if not commands:
        raise ValueError(f"{path}: native CLI known_cmds table is empty.")
    return commands


@dataclass(frozen=True, order=True)
class _SymbolRef:
    module: str
    name: str


@dataclass
class _ModuleContract:
    tree: ast.Module
    is_package: bool
    classes: dict[str, ast.ClassDef]
    functions: dict[str, ast.FunctionDef]
    assignments: dict[str, ast.expr]
    imported_symbols: dict[str, _SymbolRef]
    imported_modules: dict[str, str]


def _module_name(plugin_dir: Path, python_path: Path) -> tuple[str, bool]:
    relative = python_path.relative_to(plugin_dir)
    is_package = relative.name == "__init__.py"
    parts = list(relative.with_suffix("").parts)
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def _resolve_import_module(
    current_module: str,
    *,
    is_package: bool,
    level: int,
    imported_module: str | None,
) -> str:
    if level == 0:
        return imported_module or ""

    package_parts = current_module.split(".") if is_package else current_module.split(".")[:-1]
    parent_count = level - 1
    if parent_count > len(package_parts):
        return imported_module or ""
    base_parts = package_parts[: len(package_parts) - parent_count]
    if imported_module:
        base_parts.extend(imported_module.split("."))
    return ".".join(part for part in base_parts if part)


def _plugin_contract_modules(plugin_dir: Path) -> dict[str, _ModuleContract]:
    modules: dict[str, _ModuleContract] = {}
    if not plugin_dir.is_dir():
        return modules

    for python_path in sorted(plugin_dir.rglob("*.py")):
        module_name, is_package = _module_name(plugin_dir, python_path)
        tree = ast.parse(
            python_path.read_text(encoding="utf-8"),
            filename=str(python_path),
        )
        classes: dict[str, ast.ClassDef] = {}
        functions: dict[str, ast.FunctionDef] = {}
        assignments: dict[str, ast.expr] = {}
        imported_symbols: dict[str, _SymbolRef] = {}
        imported_modules: dict[str, str] = {}
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes[node.name] = node
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if isinstance(node, ast.FunctionDef):
                    functions[node.name] = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        assignments[target.id] = node.value
            elif isinstance(node, ast.ImportFrom):
                source_module = _resolve_import_module(
                    module_name,
                    is_package=is_package,
                    level=node.level,
                    imported_module=node.module,
                )
                for imported_name in node.names:
                    if imported_name.name == "*":
                        continue
                    local_name = imported_name.asname or imported_name.name
                    imported_symbols[local_name] = _SymbolRef(
                        source_module,
                        imported_name.name,
                    )
            elif isinstance(node, ast.Import):
                for imported_name in node.names:
                    local_name = imported_name.asname or imported_name.name.split(".", 1)[0]
                    imported_modules[local_name] = imported_name.name
        modules[module_name] = _ModuleContract(
            tree=tree,
            is_package=is_package,
            classes=classes,
            functions=functions,
            assignments=assignments,
            imported_symbols=imported_symbols,
            imported_modules=imported_modules,
        )
    return modules


def _resolve_reference(
    expression: ast.expr,
    *,
    current_module: str,
    modules: dict[str, _ModuleContract],
) -> _SymbolRef | None:
    contract = modules.get(current_module)
    if contract is None:
        return None
    if isinstance(expression, ast.Name):
        if (
            expression.id in contract.classes
            or expression.id in contract.functions
            or expression.id in contract.assignments
        ):
            return _SymbolRef(current_module, expression.id)
        return contract.imported_symbols.get(expression.id)
    if isinstance(expression, ast.Attribute) and isinstance(expression.value, ast.Name):
        imported_module = contract.imported_modules.get(expression.value.id)
        if imported_module is not None:
            return _SymbolRef(imported_module, expression.attr)
    return None


def _resolve_definition(
    reference: _SymbolRef,
    *,
    modules: dict[str, _ModuleContract],
    kind: str,
    seen: set[_SymbolRef] | None = None,
) -> _SymbolRef | None:
    if seen is None:
        seen = set()
    if reference in seen:
        return None
    seen.add(reference)

    contract = modules.get(reference.module)
    if contract is None:
        return None
    definitions = getattr(contract, kind)
    if reference.name in definitions:
        return reference
    imported = contract.imported_symbols.get(reference.name)
    if imported is None:
        return None
    return _resolve_definition(imported, modules=modules, kind=kind, seen=seen)


def _active_runner_class_refs(
    modules: dict[str, _ModuleContract],
) -> set[_SymbolRef]:
    """Extract module-qualified classes instantiated by the ``runner`` entrypoint."""
    runner_contract = modules.get("runner")
    if runner_contract is None or "runner" not in runner_contract.assignments:
        return set()

    active_classes: set[_SymbolRef] = set()
    visited_assignments: set[_SymbolRef] = set()

    def visit_expression(expression: ast.expr, current_module: str) -> None:
        if isinstance(expression, ast.Call):
            reference = _resolve_reference(
                expression.func,
                current_module=current_module,
                modules=modules,
            )
            if reference is not None:
                class_ref = _resolve_definition(
                    reference,
                    modules=modules,
                    kind="classes",
                )
                if class_ref is not None:
                    active_classes.add(class_ref)
            for argument in (*expression.args, *expression.keywords):
                value = argument.value if isinstance(argument, ast.keyword) else argument
                visit_expression(value, current_module)
            return

        if isinstance(expression, ast.Name):
            reference = _resolve_reference(
                expression,
                current_module=current_module,
                modules=modules,
            )
            if reference is None or reference in visited_assignments:
                return
            assignment_ref = _resolve_definition(
                reference,
                modules=modules,
                kind="assignments",
            )
            if assignment_ref is None:
                return
            visited_assignments.add(assignment_ref)
            assignment_contract = modules[assignment_ref.module]
            visit_expression(
                assignment_contract.assignments[assignment_ref.name],
                assignment_ref.module,
            )
            return

        for child in ast.iter_child_nodes(expression):
            if isinstance(child, ast.expr):
                visit_expression(child, current_module)

    visit_expression(runner_contract.assignments["runner"], "runner")
    return active_classes


def _runner_class_lineage(
    runner_class: _SymbolRef,
    modules: dict[str, _ModuleContract],
) -> set[_SymbolRef]:
    lineage: set[_SymbolRef] = set()

    def visit(reference: _SymbolRef) -> None:
        class_ref = _resolve_definition(reference, modules=modules, kind="classes")
        if class_ref is None or class_ref in lineage:
            return
        lineage.add(class_ref)
        contract = modules[class_ref.module]
        class_node = contract.classes[class_ref.name]
        for base in class_node.bases:
            base_ref = _resolve_reference(
                base,
                current_module=class_ref.module,
                modules=modules,
            )
            if base_ref is not None:
                visit(base_ref)

    visit(runner_class)
    return lineage


def _native_commands_in_node(
    node: ast.AST,
    native_cli_commands: set[str],
) -> set[str]:
    commands: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, (ast.List, ast.Tuple)) or len(child.elts) < 2:
            continue
        command_node = child.elts[1]
        if not isinstance(command_node, ast.Constant):
            continue
        command = command_node.value
        if isinstance(command, str) and command in native_cli_commands:
            commands.add(command)
    return commands


def _called_function_refs(
    node: ast.AST,
    *,
    current_module: str,
    modules: dict[str, _ModuleContract],
) -> set[_SymbolRef]:
    symbols: set[_SymbolRef] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        reference = _resolve_reference(
            child.func,
            current_module=current_module,
            modules=modules,
        )
        if reference is None:
            continue
        function_ref = _resolve_definition(
            reference,
            modules=modules,
            kind="functions",
        )
        if function_ref is not None:
            symbols.add(function_ref)
    return symbols


def _commands_for_runner_class(
    runner_class: _SymbolRef,
    *,
    modules: dict[str, _ModuleContract],
    native_cli_commands: set[str],
) -> set[str]:
    """Resolve commands from one runner class, its bases, and called helpers."""
    commands: set[str] = set()
    visited_classes: set[_SymbolRef] = set()
    visited_functions: set[_SymbolRef] = set()

    def visit_function(reference: _SymbolRef) -> None:
        function_ref = _resolve_definition(reference, modules=modules, kind="functions")
        if function_ref is None or function_ref in visited_functions:
            return
        visited_functions.add(function_ref)
        contract = modules[function_ref.module]
        function_node = contract.functions[function_ref.name]
        commands.update(_native_commands_in_node(function_node, native_cli_commands))
        for called_ref in _called_function_refs(
            function_node,
            current_module=function_ref.module,
            modules=modules,
        ):
            visit_function(called_ref)

    def visit_class(reference: _SymbolRef) -> None:
        class_ref = _resolve_definition(reference, modules=modules, kind="classes")
        if class_ref is None or class_ref in visited_classes:
            return
        visited_classes.add(class_ref)
        contract = modules[class_ref.module]
        class_node = contract.classes[class_ref.name]
        commands.update(_native_commands_in_node(class_node, native_cli_commands))
        for base in class_node.bases:
            base_ref = _resolve_reference(
                base,
                current_module=class_ref.module,
                modules=modules,
            )
            if base_ref is not None:
                visit_class(base_ref)
        for called_ref in _called_function_refs(
            class_node,
            current_module=class_ref.module,
            modules=modules,
        ):
            visit_function(called_ref)

    visit_class(runner_class)
    return commands


def extract_runtime_cli_commands_from_e2e_plugins(
    matrix: dict[str, dict[str, Any]],
    models_dir: Path,
    native_cli_commands: set[str],
) -> dict[str, set[str]]:
    """Map runtimes to native commands used by their declared runner class."""
    runtime_commands: dict[str, set[str]] = {}
    owner_contracts: dict[
        Path,
        tuple[
            dict[str, _ModuleContract],
            set[_SymbolRef],
        ],
    ] = {}

    for manifest_path in _iter_e2e_manifest_paths(models_dir):
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        runtime_strategy = raw.get("runtime_strategy")
        if not _is_nonempty_str(runtime_strategy):
            continue
        runner_class_ref = matrix.get(runtime_strategy, {}).get("runner_class")
        if not _is_nonempty_str(runner_class_ref):
            continue

        if manifest_path.parent.name == "manifests":
            owner_dir = manifest_path.parent.parent
        else:
            family = raw.get("family")
            if not _is_nonempty_str(family):
                continue
            owner_dir = models_dir / family

        if owner_dir not in owner_contracts:
            plugin_dir = owner_dir / "e2e_plugins"
            modules = _plugin_contract_modules(plugin_dir)
            owner_contracts[owner_dir] = (
                modules,
                _active_runner_class_refs(modules),
            )
        modules, active_runner_classes = owner_contracts[owner_dir]
        declared_runner_class = _class_basename(runner_class_ref)
        matching_active_classes = {
            active_class
            for active_class in active_runner_classes
            if declared_runner_class
            in {item.name for item in _runner_class_lineage(active_class, modules)}
        }
        runtime_commands.setdefault(runtime_strategy, set()).update(
            command
            for active_class in matching_active_classes
            for command in _commands_for_runner_class(
                active_class,
                modules=modules,
                native_cli_commands=native_cli_commands,
            )
        )

    return runtime_commands


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

    conflicts = {runtime: sorted(tasks) for runtime, tasks in values.items() if len(tasks) > 1}
    if conflicts:
        raise ValueError(
            f"{models_dir}: runtime_strategy values map to multiple task_strategy "
            f"values: {conflicts}"
        )
    return {runtime: next(iter(tasks)) for runtime, tasks in sorted(values.items())}


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
    """Yield central legacy or model-local E2E plugin files for ``kind``.

    Most model-local implementations live below ``runners/`` or
    ``comparators/``.  A model may instead expose a specialized wrapper through
    the singular ``runner.py`` or ``comparator.py`` entry point; include that
    active entry point so validation follows the object the harness loads.
    """
    if (root / kind).is_dir():
        yield from sorted((root / kind).glob("*.py"))
        return
    flat_files = sorted(root.glob("*.py"))
    if flat_files:
        yield from flat_files
        return
    for plugin_dir in sorted(root.glob(f"*/e2e_plugins/{kind}")):
        yield from sorted(plugin_dir.glob("*.py"))
        entrypoint_name = "runner.py" if kind == "runners" else "comparator.py"
        entrypoint = plugin_dir.parent / entrypoint_name
        if entrypoint.is_file():
            yield entrypoint


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
                and any(
                    isinstance(target, ast.Name) and target.id == "name" for target in stmt.targets
                )
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
    native_cli_commands: set[str],
    runner_cli_commands_by_runtime: dict[str, set[str]],
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
    authoritative_strategies = cpp_runtime_strategies | manifest_strategies
    stale_matrix_strategies = sorted(matrix_strategies - authoritative_strategies)
    if stale_matrix_strategies:
        errors.append(
            "tests/runtime_strategy_matrix.yaml has runtime strategies absent "
            "from runtime sources and E2E manifests: "
            f"{stale_matrix_strategies}"
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
        else:
            duplicate_cli_commands = sorted(
                command for command in set(cli_commands) if cli_commands.count(command) > 1
            )
            if duplicate_cli_commands:
                errors.append(
                    f"{runtime_strategy}: duplicate entries in cli_commands: "
                    f"{duplicate_cli_commands}."
                )

            unknown_cli_commands = sorted(set(cli_commands) - native_cli_commands)
            if unknown_cli_commands:
                errors.append(
                    f"{runtime_strategy}: cli_commands references commands not accepted "
                    f"by the native CLI: {unknown_cli_commands}."
                )

            if runtime_strategy in manifest_strategies:
                runner_cli_commands = runner_cli_commands_by_runtime.get(runtime_strategy, set())
                unsupported_cli_commands = sorted(set(cli_commands) - runner_cli_commands)
                if unsupported_cli_commands:
                    errors.append(
                        f"{runtime_strategy}: cli_commands {unsupported_cli_commands} "
                        "are not used by the model-owned E2E runner/command builders; "
                        f"discovered native commands: {sorted(runner_cli_commands)}."
                    )
                undeclared_runner_commands = sorted(runner_cli_commands - set(cli_commands))
                if undeclared_runner_commands:
                    errors.append(
                        f"{runtime_strategy}: model-owned E2E runner/command builders use "
                        f"native commands missing from cli_commands: "
                        f"{undeclared_runner_commands}."
                    )

        performance_mode = entry.get("performance_mode")
        if not _is_nonempty_str(performance_mode):
            errors.append(f"{runtime_strategy}: 'performance_mode' must be a non-empty string.")

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
            errors.append(f"{runtime_strategy}: 'diff_framework_check_classes' must be a list.")
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
    cli_args_path: Path = DEFAULT_CLI_ARGS_PATH,
    torchtrt_strategies_dir: Path = DEFAULT_TORCHTRT_STRATEGIES_DIR,
    e2e_models_dir: Path = DEFAULT_E2E_MODELS_DIR,
    diff_checks_dir: Path = DEFAULT_DIFF_CHECKS_DIR,
    runners_dir: Path = DEFAULT_RUNNERS_DIR,
    comparators_dir: Path = DEFAULT_COMPARATORS_DIR,
) -> list[str]:
    """Load all sources and validate the runtime strategy matrix."""
    matrix = load_runtime_strategy_matrix(matrix_path)
    runtime_to_task_strategy = extract_runtime_to_task_strategy_from_manifests(e2e_models_dir)
    native_cli_commands = extract_native_cli_commands(cli_args_path)
    runner_cli_commands_by_runtime = extract_runtime_cli_commands_from_e2e_plugins(
        matrix,
        e2e_models_dir,
        native_cli_commands,
    )
    candidate_strategies = set(matrix.keys()) | set(runtime_to_task_strategy.keys())

    runtime_cpp_files = discover_runtime_strategy_source_files(
        cpp_path=cpp_path.resolve(),
        builders_dir=builders_dir.resolve(),
        runtime_registry_path=runtime_registry_path.resolve(),
        torchtrt_strategies_dir=torchtrt_strategies_dir.resolve(),
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
    comparator_classes_by_task = extract_comparator_classes_by_task_strategy(comparators_dir)
    return validate_matrix_data(
        matrix=matrix,
        cpp_runtime_strategies=cpp_runtime_strategies,
        runtime_to_task_strategy=runtime_to_task_strategy,
        diff_check_classes=diff_check_classes,
        runner_classes_by_task=runner_classes_by_task,
        comparator_classes_by_task=comparator_classes_by_task,
        native_cli_commands=native_cli_commands,
        runner_cli_commands_by_runtime=runner_cli_commands_by_runtime,
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
        "--cli-args",
        type=Path,
        default=DEFAULT_CLI_ARGS_PATH,
        help="Path to src/cli/args.cpp containing the native known_cmds table.",
    )
    parser.add_argument(
        "--torchtrt-strategies-dir",
        type=Path,
        default=DEFAULT_TORCHTRT_STRATEGIES_DIR,
        help="Path to torch-trt strategy source files.",
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
            cli_args_path=args.cli_args,
            torchtrt_strategies_dir=args.torchtrt_strategies_dir,
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
        "entrypoint, native CLI, active model-owned runners, E2E manifests, "
        "and diff-framework checks."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
