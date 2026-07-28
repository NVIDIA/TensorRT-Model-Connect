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
from collections.abc import Callable, Iterable
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

    def qualified_name(self) -> str:
        return f"{self.module}.{self.name}"


def _canonical_plugin_class_ref(
    value: str,
    *,
    entrypoint_module: str,
    implementation_package: str,
) -> _SymbolRef | None:
    """Normalize the matrix's one-module shorthand without dropping identity."""
    if "." not in value:
        return None
    module, class_name = value.rsplit(".", 1)
    module = module.strip()
    class_name = class_name.strip()
    if not module or not class_name:
        return None
    if module != entrypoint_module and not module.startswith(f"{implementation_package}."):
        # Existing matrix rows use ``diffusion.Runner`` as a stable shorthand
        # for ``runners.diffusion.Runner``. Multi-segment module paths are
        # already explicit and must not be rewritten.
        if "." not in module:
            module = f"{implementation_package}.{module}"
    return _SymbolRef(module, class_name)


def _canonical_runner_class_ref(value: str) -> _SymbolRef | None:
    return _canonical_plugin_class_ref(
        value,
        entrypoint_module="runner",
        implementation_package="runners",
    )


def _canonical_comparator_class_ref(value: str) -> _SymbolRef | None:
    return _canonical_plugin_class_ref(
        value,
        entrypoint_module="comparator",
        implementation_package="comparators",
    )


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
    if isinstance(expression, ast.Attribute):
        imported_module = _resolve_module_expression(
            expression.value,
            current_module=current_module,
            modules=modules,
        )
        if imported_module is not None:
            return _SymbolRef(imported_module, expression.attr)
    return None


def _resolve_module_expression(
    expression: ast.expr,
    *,
    current_module: str,
    modules: dict[str, _ModuleContract],
) -> str | None:
    """Resolve imported modules, including recursive package re-exports."""
    contract = modules.get(current_module)
    if contract is None:
        return None
    if isinstance(expression, ast.Name):
        imported_module = contract.imported_modules.get(expression.id)
        if imported_module in modules:
            return imported_module
        imported_symbol = contract.imported_symbols.get(expression.id)
        if imported_symbol is None:
            return None
        candidate_module = ".".join(
            part for part in (imported_symbol.module, imported_symbol.name) if part
        )
        return candidate_module if candidate_module in modules else None
    if not isinstance(expression, ast.Attribute):
        return None

    parent_module = _resolve_module_expression(
        expression.value,
        current_module=current_module,
        modules=modules,
    )
    if parent_module is None:
        return None
    parent_contract = modules.get(parent_module)
    if parent_contract is None:
        return None

    imported_module = parent_contract.imported_modules.get(expression.attr)
    if imported_module in modules:
        return imported_module
    imported_symbol = parent_contract.imported_symbols.get(expression.attr)
    if imported_symbol is None:
        return None
    candidate_module = ".".join(
        part for part in (imported_symbol.module, imported_symbol.name) if part
    )
    return candidate_module if candidate_module in modules else None


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
    assignment = contract.assignments.get(reference.name)
    if assignment is not None:
        alias = _resolve_reference(
            assignment,
            current_module=reference.module,
            modules=modules,
        )
        if alias is None:
            return None
        return _resolve_definition(alias, modules=modules, kind=kind, seen=seen)
    imported = contract.imported_symbols.get(reference.name)
    if imported is None:
        return None
    return _resolve_definition(imported, modules=modules, kind=kind, seen=seen)


@dataclass(frozen=True)
class _EntrypointBinding:
    expression: ast.expr | None = None
    ambiguous: bool = False


@dataclass(frozen=True)
class _EntrypointClassResolution:
    classes: frozenset[_SymbolRef] = frozenset()
    is_collection: bool = False
    ambiguous: bool = False


@dataclass(frozen=True)
class _StaticAssignmentState:
    expression: ast.expr | None = None
    ambiguous: bool = False


_UNKNOWN_CONSTANT = object()


def _safe_constant_value(
    expression: ast.expr,
    *,
    constant_resolver: Callable[[ast.Name], Any] | None = None,
) -> Any:
    """Evaluate side-effect-free literal expressions without executing code."""
    if isinstance(expression, ast.Constant):
        return expression.value

    if isinstance(expression, ast.Name) and constant_resolver is not None:
        return constant_resolver(expression)

    if isinstance(expression, (ast.List, ast.Set, ast.Tuple)):
        values: list[Any] = []
        for element in expression.elts:
            value = _safe_constant_value(
                element,
                constant_resolver=constant_resolver,
            )
            if value is _UNKNOWN_CONSTANT:
                return _UNKNOWN_CONSTANT
            values.append(value)
        if isinstance(expression, ast.List):
            return values
        if isinstance(expression, ast.Tuple):
            return tuple(values)
        try:
            return set(values)
        except TypeError:
            return _UNKNOWN_CONSTANT

    if isinstance(expression, ast.Dict):
        keys: list[Any] = []
        values: list[Any] = []
        for key_expression, value_expression in zip(
            expression.keys,
            expression.values,
            strict=True,
        ):
            if key_expression is None:
                return _UNKNOWN_CONSTANT
            key = _safe_constant_value(
                key_expression,
                constant_resolver=constant_resolver,
            )
            value = _safe_constant_value(
                value_expression,
                constant_resolver=constant_resolver,
            )
            if key is _UNKNOWN_CONSTANT or value is _UNKNOWN_CONSTANT:
                return _UNKNOWN_CONSTANT
            keys.append(key)
            values.append(value)
        try:
            return dict(zip(keys, values, strict=True))
        except TypeError:
            return _UNKNOWN_CONSTANT

    if isinstance(expression, ast.Compare):
        left_expression = expression.left
        left = _safe_constant_value(
            left_expression,
            constant_resolver=constant_resolver,
        )
        if left is _UNKNOWN_CONSTANT:
            return _UNKNOWN_CONSTANT
        for operator, comparator_expression in zip(
            expression.ops,
            expression.comparators,
            strict=True,
        ):
            right = _safe_constant_value(
                comparator_expression,
                constant_resolver=constant_resolver,
            )
            if right is _UNKNOWN_CONSTANT:
                return _UNKNOWN_CONSTANT
            try:
                if isinstance(operator, ast.Eq):
                    matches = left == right
                elif isinstance(operator, ast.NotEq):
                    matches = left != right
                elif isinstance(operator, ast.Lt):
                    matches = left < right
                elif isinstance(operator, ast.LtE):
                    matches = left <= right
                elif isinstance(operator, ast.Gt):
                    matches = left > right
                elif isinstance(operator, ast.GtE):
                    matches = left >= right
                elif isinstance(operator, ast.In):
                    matches = left in right
                elif isinstance(operator, ast.NotIn):
                    matches = left not in right
                elif isinstance(operator, (ast.Is, ast.IsNot)):
                    if not (
                        isinstance(left_expression, ast.Constant)
                        and isinstance(comparator_expression, ast.Constant)
                        and any(
                            left_expression.value is singleton
                            for singleton in (None, True, False, Ellipsis)
                        )
                        and any(
                            comparator_expression.value is singleton
                            for singleton in (None, True, False, Ellipsis)
                        )
                    ):
                        return _UNKNOWN_CONSTANT
                    matches = left is right
                    if isinstance(operator, ast.IsNot):
                        matches = not matches
                else:
                    return _UNKNOWN_CONSTANT
            except (TypeError, ValueError):
                return _UNKNOWN_CONSTANT
            if not matches:
                return False
            left = right
            left_expression = comparator_expression
        return True

    if isinstance(expression, ast.IfExp):
        test_value = _safe_constant_value(
            expression.test,
            constant_resolver=constant_resolver,
        )
        if test_value is _UNKNOWN_CONSTANT:
            return _UNKNOWN_CONSTANT
        selected = expression.body if bool(test_value) else expression.orelse
        return _safe_constant_value(
            selected,
            constant_resolver=constant_resolver,
        )

    if isinstance(expression, ast.UnaryOp):
        operand = _safe_constant_value(
            expression.operand,
            constant_resolver=constant_resolver,
        )
        if operand is _UNKNOWN_CONSTANT:
            return _UNKNOWN_CONSTANT
        if isinstance(expression.op, ast.Not):
            return not bool(operand)
        if isinstance(operand, (complex, float, int)):
            if isinstance(expression.op, ast.UAdd):
                return +operand
            if isinstance(expression.op, ast.USub):
                return -operand
            if isinstance(expression.op, ast.Invert) and isinstance(operand, int):
                return ~operand

    if isinstance(expression, ast.BoolOp):
        values = iter(expression.values)
        first = _safe_constant_value(
            next(values),
            constant_resolver=constant_resolver,
        )
        if first is _UNKNOWN_CONSTANT:
            return _UNKNOWN_CONSTANT
        result = first
        for value_expression in values:
            if isinstance(expression.op, ast.And) and not bool(result):
                return result
            if isinstance(expression.op, ast.Or) and bool(result):
                return result
            result = _safe_constant_value(
                value_expression,
                constant_resolver=constant_resolver,
            )
            if result is _UNKNOWN_CONSTANT:
                return _UNKNOWN_CONSTANT
        return result

    if isinstance(expression, ast.Subscript) and not isinstance(expression.slice, ast.Slice):
        container = _safe_constant_value(
            expression.value,
            constant_resolver=constant_resolver,
        )
        index = _safe_constant_value(
            expression.slice,
            constant_resolver=constant_resolver,
        )
        if (
            isinstance(container, (list, tuple))
            and isinstance(index, int)
            and index >= -len(container)
            and index < len(container)
        ):
            return container[index]

    return _UNKNOWN_CONSTANT


def _safe_truth_value(
    expression: ast.expr,
    *,
    constant_resolver: Callable[[ast.Name], Any] | None = None,
) -> bool | None:
    value = _safe_constant_value(
        expression,
        constant_resolver=constant_resolver,
    )
    return None if value is _UNKNOWN_CONSTANT else bool(value)


def _compare_operation_is_provably_side_effect_free(
    left: ast.expr,
    operator: ast.cmpop,
    right: ast.expr,
    *,
    constant_resolver: Callable[[ast.Name], Any] | None = None,
) -> bool:
    """Whether one comparison cannot dispatch to plugin-defined Python code."""
    if isinstance(operator, (ast.Is, ast.IsNot)):
        return True
    return all(
        _safe_constant_value(
            operand,
            constant_resolver=constant_resolver,
        )
        is not _UNKNOWN_CONSTANT
        for operand in (left, right)
    )


def _truth_test_is_provably_side_effect_free(
    expression: ast.expr,
    *,
    constant_resolver: Callable[[ast.Name], Any] | None = None,
) -> bool:
    """Whether ``bool(expression)`` cannot invoke plugin-defined Python code."""
    if (
        _safe_constant_value(
            expression,
            constant_resolver=constant_resolver,
        )
        is not _UNKNOWN_CONSTANT
    ):
        return True
    if isinstance(expression, ast.Compare):
        left = expression.left
        for operator, comparator in zip(
            expression.ops,
            expression.comparators,
            strict=True,
        ):
            if not _compare_operation_is_provably_side_effect_free(
                left,
                operator,
                comparator,
                constant_resolver=constant_resolver,
            ):
                return False
            left = comparator
        return True
    if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
        return _truth_test_is_provably_side_effect_free(
            expression.operand,
            constant_resolver=constant_resolver,
        )
    return False


def _expression_evaluation_is_provably_side_effect_free(
    expression: ast.expr,
    *,
    constant_resolver: Callable[[ast.Name], Any] | None = None,
) -> bool:
    """Whether evaluating an expression cannot execute plugin-defined code."""
    if isinstance(expression, (ast.Constant, ast.Name)):
        return True

    if isinstance(expression, ast.NamedExpr):
        return isinstance(
            expression.target, ast.Name
        ) and _expression_evaluation_is_provably_side_effect_free(
            expression.value,
            constant_resolver=constant_resolver,
        )

    if isinstance(expression, ast.Lambda):
        return all(
            _expression_evaluation_is_provably_side_effect_free(
                default,
                constant_resolver=constant_resolver,
            )
            for default in (
                *expression.args.defaults,
                *(value for value in expression.args.kw_defaults if value is not None),
            )
        )

    if isinstance(expression, (ast.List, ast.Tuple)):
        for element in expression.elts:
            if isinstance(element, ast.Starred):
                value = _safe_constant_value(
                    element.value,
                    constant_resolver=constant_resolver,
                )
                if not isinstance(value, (list, tuple)):
                    return False
                continue
            if not _expression_evaluation_is_provably_side_effect_free(
                element,
                constant_resolver=constant_resolver,
            ):
                return False
        return True

    if isinstance(expression, (ast.Set, ast.Dict)):
        return (
            _safe_constant_value(
                expression,
                constant_resolver=constant_resolver,
            )
            is not _UNKNOWN_CONSTANT
        )

    if isinstance(expression, ast.BoolOp):
        for index, value_expression in enumerate(expression.values):
            if not _expression_evaluation_is_provably_side_effect_free(
                value_expression,
                constant_resolver=constant_resolver,
            ):
                return False
            if index == len(expression.values) - 1:
                return True
            if not _truth_test_is_provably_side_effect_free(
                value_expression,
                constant_resolver=constant_resolver,
            ):
                return False
            value = _safe_truth_value(
                value_expression,
                constant_resolver=constant_resolver,
            )
            if value is not None and (
                (isinstance(expression.op, ast.And) and not value)
                or (isinstance(expression.op, ast.Or) and value)
            ):
                return True
        return True

    if isinstance(expression, ast.IfExp):
        if not _expression_evaluation_is_provably_side_effect_free(
            expression.test,
            constant_resolver=constant_resolver,
        ) or not _truth_test_is_provably_side_effect_free(
            expression.test,
            constant_resolver=constant_resolver,
        ):
            return False
        test_value = _safe_truth_value(
            expression.test,
            constant_resolver=constant_resolver,
        )
        branches = (
            (expression.body, expression.orelse)
            if test_value is None
            else (expression.body if test_value else expression.orelse,)
        )
        return all(
            _expression_evaluation_is_provably_side_effect_free(
                branch,
                constant_resolver=constant_resolver,
            )
            for branch in branches
        )

    if isinstance(expression, ast.Compare):
        if not _expression_evaluation_is_provably_side_effect_free(
            expression.left,
            constant_resolver=constant_resolver,
        ):
            return False
        left = expression.left
        for operator, comparator in zip(
            expression.ops,
            expression.comparators,
            strict=True,
        ):
            if not _expression_evaluation_is_provably_side_effect_free(
                comparator,
                constant_resolver=constant_resolver,
            ) or not _compare_operation_is_provably_side_effect_free(
                left,
                operator,
                comparator,
                constant_resolver=constant_resolver,
            ):
                return False
            comparison = ast.Compare(
                left=left,
                ops=[operator],
                comparators=[comparator],
            )
            if (
                _safe_truth_value(
                    comparison,
                    constant_resolver=constant_resolver,
                )
                is False
            ):
                return True
            left = comparator
        return True

    if isinstance(expression, ast.UnaryOp):
        if not _expression_evaluation_is_provably_side_effect_free(
            expression.operand,
            constant_resolver=constant_resolver,
        ):
            return False
        if isinstance(expression.op, ast.Not):
            return _truth_test_is_provably_side_effect_free(
                expression.operand,
                constant_resolver=constant_resolver,
            )
        return (
            _safe_constant_value(
                expression,
                constant_resolver=constant_resolver,
            )
            is not _UNKNOWN_CONSTANT
        )

    if isinstance(expression, ast.BinOp):
        return (
            _safe_constant_value(
                expression,
                constant_resolver=constant_resolver,
            )
            is not _UNKNOWN_CONSTANT
        )

    if isinstance(expression, ast.Slice):
        return all(
            value is None
            or _expression_evaluation_is_provably_side_effect_free(
                value,
                constant_resolver=constant_resolver,
            )
            for value in (expression.lower, expression.upper, expression.step)
        )

    # Calls, attributes, subscriptions, comprehensions, awaiting/yielding, and
    # formatted values can dispatch to plugin-defined Python protocols.
    return False


def _plain_local_class_constructor_is_provably_side_effect_free(
    expression: ast.expr,
    *,
    known_classes: dict[str, ast.ClassDef] | None,
) -> bool:
    """Recognize a zero-argument local class construction with no callbacks."""
    if (
        known_classes is None
        or not isinstance(expression, ast.Call)
        or expression.args
        or expression.keywords
        or not isinstance(expression.func, ast.Name)
    ):
        return False
    class_node = known_classes.get(expression.func.id)
    if class_node is None or class_node.decorator_list or class_node.bases or class_node.keywords:
        return False
    for body_statement in class_node.body:
        if isinstance(body_statement, ast.Pass):
            continue
        if (
            isinstance(body_statement, ast.Expr)
            and isinstance(body_statement.value, ast.Constant)
            and isinstance(body_statement.value.value, str)
        ):
            continue
        if (
            isinstance(body_statement, (ast.FunctionDef, ast.AsyncFunctionDef))
            and body_statement.name not in {"__init__", "__new__"}
            and not body_statement.decorator_list
            and not body_statement.args.defaults
            and not any(value is not None for value in body_statement.args.kw_defaults)
        ):
            continue
        return False
    return True


def _plain_class_definition_is_provably_side_effect_free(
    statement: ast.ClassDef,
    *,
    known_classes: dict[str, ast.ClassDef] | None = None,
) -> bool:
    """Recognize the narrow class form whose module-time execution is inert."""
    if statement.decorator_list or statement.bases or statement.keywords:
        return False
    for body_statement in statement.body:
        if isinstance(body_statement, ast.Pass):
            continue
        if (
            isinstance(body_statement, ast.Expr)
            and isinstance(body_statement.value, ast.Constant)
            and isinstance(body_statement.value.value, str)
        ):
            continue
        if isinstance(body_statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                body_statement.decorator_list
                or body_statement.args.defaults
                or any(value is not None for value in body_statement.args.kw_defaults)
            ):
                return False
            continue
        if isinstance(body_statement, (ast.Assign, ast.AnnAssign)):
            value = body_statement.value
            targets = (
                body_statement.targets
                if isinstance(body_statement, ast.Assign)
                else [body_statement.target]
            )
            if (
                value is None
                or not all(isinstance(target, ast.Name) for target in targets)
                or not (
                    _expression_evaluation_is_provably_side_effect_free(value)
                    or _plain_local_class_constructor_is_provably_side_effect_free(
                        value,
                        known_classes=known_classes,
                    )
                )
            ):
                return False
            continue
        return False
    return True


def _direct_assignment_value(
    statement: ast.stmt,
    assignment_name: str,
) -> ast.expr | None:
    if isinstance(statement, ast.Assign):
        targets = statement.targets
    elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
        targets = [statement.target]
    else:
        return None

    if any(isinstance(target, ast.Name) and target.id == assignment_name for target in targets):
        return statement.value
    return None


def _same_entrypoint_binding(
    left: _EntrypointBinding,
    right: _EntrypointBinding,
) -> bool:
    if left.ambiguous or right.ambiguous:
        return left.ambiguous and right.ambiguous
    if left.expression is None or right.expression is None:
        return left.expression is None and right.expression is None
    return ast.dump(left.expression, include_attributes=False) == ast.dump(
        right.expression,
        include_attributes=False,
    )


def _entrypoint_binding_from_expression(
    expression: ast.expr,
    *,
    constant_resolver: Callable[[ast.Name], Any] | None = None,
) -> _EntrypointBinding:
    """Resolve a direct assignment expression when its branch is knowable."""
    if not isinstance(expression, ast.IfExp):
        return _EntrypointBinding(expression=expression)

    test_value = _safe_truth_value(
        expression.test,
        constant_resolver=constant_resolver,
    )
    if test_value is not None:
        selected = expression.body if test_value else expression.orelse
        return _entrypoint_binding_from_expression(
            selected,
            constant_resolver=constant_resolver,
        )

    # Preserve the conditional expression. Its two branches may still resolve
    # to the same concrete class, in which case the runtime choice is
    # unambiguous even though the condition is not statically known.
    return _EntrypointBinding(expression=expression)


class _SameScopeRebindingVisitor(ast.NodeVisitor):
    """Find bindings without descending into nested Python scopes."""

    def __init__(
        self,
        assignment_name: str,
        *,
        constant_resolver: Callable[[ast.Name], Any] | None = None,
    ) -> None:
        self.assignment_name = assignment_name
        self.constant_resolver = constant_resolver
        self.rebinds = False

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == self.assignment_name and isinstance(
            node.ctx,
            (ast.Store, ast.Del),
        ):
            self.rebinds = True

    def _visit_definition_expressions(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> None:
        if node.name == self.assignment_name:
            self.rebinds = True
        for expression in (
            *node.decorator_list,
            *node.args.defaults,
            *(default for default in node.args.kw_defaults if default is not None),
        ):
            self.visit(expression)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_definition_expressions(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_definition_expressions(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if node.name == self.assignment_name:
            self.rebinds = True
        for expression in (
            *node.decorator_list,
            *node.bases,
            *(keyword.value for keyword in node.keywords),
        ):
            self.visit(expression)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        # Defaults execute when the module creates the lambda. The body remains
        # a nested scope and does not execute until the lambda is called.
        for expression in (
            *node.args.defaults,
            *(default for default in node.args.kw_defaults if default is not None),
        ):
            self.visit(expression)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        for expression in node.values:
            self.visit(expression)
            value = _safe_truth_value(
                expression,
                constant_resolver=self.constant_resolver,
            )
            if value is None:
                continue
            if isinstance(node.op, ast.And) and not value:
                return
            if isinstance(node.op, ast.Or) and value:
                return

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.visit(node.test)
        test_value = _safe_truth_value(
            node.test,
            constant_resolver=self.constant_resolver,
        )
        if test_value is None:
            self.visit(node.body)
            self.visit(node.orelse)
            return
        self.visit(node.body if test_value else node.orelse)

    def visit_Compare(self, node: ast.Compare) -> None:
        self.visit(node.left)
        left = node.left
        for operator, comparator in zip(
            node.ops,
            node.comparators,
            strict=True,
        ):
            self.visit(comparator)
            comparison = ast.Compare(
                left=left,
                ops=[operator],
                comparators=[comparator],
            )
            value = _safe_truth_value(
                comparison,
                constant_resolver=self.constant_resolver,
            )
            if value is False:
                return
            left = comparator

    def _visit_comprehension_expressions(
        self,
        generators: list[ast.comprehension],
        expressions: Iterable[ast.expr],
    ) -> None:
        for generator in generators:
            # Comprehension targets are local in Python 3. The iterable is
            # evaluated outside that local scope, while assignment expressions
            # in filters or results bind the containing scope. Inspect those
            # expressions without treating the local target as a rebind.
            self.visit(generator.iter)
            for condition in generator.ifs:
                self.visit(condition)
        for expression in expressions:
            self.visit(expression)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self._visit_comprehension_expressions(node.generators, [node.elt])

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self._visit_comprehension_expressions(node.generators, [node.elt])

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self._visit_comprehension_expressions(
            node.generators,
            [node.key, node.value],
        )

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self._visit_comprehension_expressions(node.generators, [node.elt])

    def visit_alias(self, node: ast.alias) -> None:
        bound_name = node.asname or node.name.split(".", 1)[0]
        if bound_name == self.assignment_name:
            self.rebinds = True

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name == self.assignment_name:
            self.rebinds = True
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name == self.assignment_name:
            self.rebinds = True
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name == self.assignment_name:
            self.rebinds = True

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest == self.assignment_name:
            self.rebinds = True
        self.generic_visit(node)


def _may_rebind_entrypoint(
    node: ast.AST,
    assignment_name: str,
    *,
    constant_resolver: Callable[[ast.Name], Any] | None = None,
) -> bool:
    visitor = _SameScopeRebindingVisitor(
        assignment_name,
        constant_resolver=constant_resolver,
    )
    visitor.visit(node)
    return visitor.rebinds


def _same_static_assignment_state(
    left: _StaticAssignmentState,
    right: _StaticAssignmentState,
) -> bool:
    if left.ambiguous or right.ambiguous:
        return left.ambiguous and right.ambiguous
    if left.expression is None or right.expression is None:
        return left.expression is None and right.expression is None
    return ast.dump(left.expression, include_attributes=False) == ast.dump(
        right.expression,
        include_attributes=False,
    )


def _merge_static_assignment_states(
    left: _StaticAssignmentState,
    right: _StaticAssignmentState,
) -> _StaticAssignmentState:
    if _same_static_assignment_state(left, right):
        return left
    return _StaticAssignmentState(ambiguous=True)


def _assignment_state_after_expression(
    expression: ast.expr,
    *,
    assignment_name: str,
    initial: _StaticAssignmentState,
    constant_resolver: Callable[[ast.Name], Any] | None,
) -> _StaticAssignmentState:
    """Apply same-scope effects, failing closed at unknown Python callbacks."""
    if isinstance(expression, ast.NamedExpr):
        state = _assignment_state_after_expression(
            expression.value,
            assignment_name=assignment_name,
            initial=initial,
            constant_resolver=constant_resolver,
        )
        if isinstance(expression.target, ast.Name) and expression.target.id == assignment_name:
            return _StaticAssignmentState(expression=expression.value)
        if _may_rebind_entrypoint(
            expression.target,
            assignment_name,
            constant_resolver=constant_resolver,
        ):
            return _StaticAssignmentState(ambiguous=True)
        return state

    if isinstance(expression, ast.Lambda):
        state = initial
        for default in (
            *expression.args.defaults,
            *(value for value in expression.args.kw_defaults if value is not None),
        ):
            state = _assignment_state_after_expression(
                default,
                assignment_name=assignment_name,
                initial=state,
                constant_resolver=constant_resolver,
            )
        return state

    if isinstance(expression, ast.BoolOp):

        def visit_from(
            index: int,
            state: _StaticAssignmentState,
        ) -> _StaticAssignmentState:
            value_expression = expression.values[index]
            state = _assignment_state_after_expression(
                value_expression,
                assignment_name=assignment_name,
                initial=state,
                constant_resolver=constant_resolver,
            )
            if index == len(expression.values) - 1:
                return state

            if not _truth_test_is_provably_side_effect_free(
                value_expression,
                constant_resolver=constant_resolver,
            ):
                return _StaticAssignmentState(ambiguous=True)
            value = _safe_truth_value(
                value_expression,
                constant_resolver=constant_resolver,
            )
            if value is not None:
                short_circuits = (isinstance(expression.op, ast.And) and not value) or (
                    isinstance(expression.op, ast.Or) and value
                )
                return state if short_circuits else visit_from(index + 1, state)

            continued = visit_from(index + 1, state)
            return _merge_static_assignment_states(state, continued)

        return visit_from(0, initial)

    if isinstance(expression, ast.IfExp):
        state = _assignment_state_after_expression(
            expression.test,
            assignment_name=assignment_name,
            initial=initial,
            constant_resolver=constant_resolver,
        )
        if not _truth_test_is_provably_side_effect_free(
            expression.test,
            constant_resolver=constant_resolver,
        ):
            return _StaticAssignmentState(ambiguous=True)
        test_value = _safe_truth_value(
            expression.test,
            constant_resolver=constant_resolver,
        )
        if test_value is not None:
            selected = expression.body if test_value else expression.orelse
            return _assignment_state_after_expression(
                selected,
                assignment_name=assignment_name,
                initial=state,
                constant_resolver=constant_resolver,
            )
        body_state = _assignment_state_after_expression(
            expression.body,
            assignment_name=assignment_name,
            initial=state,
            constant_resolver=constant_resolver,
        )
        else_state = _assignment_state_after_expression(
            expression.orelse,
            assignment_name=assignment_name,
            initial=state,
            constant_resolver=constant_resolver,
        )
        return _merge_static_assignment_states(body_state, else_state)

    if isinstance(expression, (ast.List, ast.Tuple, ast.Set)):
        state = initial
        for element in expression.elts:
            value_expression = element.value if isinstance(element, ast.Starred) else element
            state = _assignment_state_after_expression(
                value_expression,
                assignment_name=assignment_name,
                initial=state,
                constant_resolver=constant_resolver,
            )
            if isinstance(element, ast.Starred):
                value = _safe_constant_value(
                    element.value,
                    constant_resolver=constant_resolver,
                )
                if not isinstance(value, (list, tuple)):
                    return _StaticAssignmentState(ambiguous=True)
            elif isinstance(expression, ast.Set) and (
                _safe_constant_value(
                    element,
                    constant_resolver=constant_resolver,
                )
                is _UNKNOWN_CONSTANT
            ):
                return _StaticAssignmentState(ambiguous=True)
        return state

    if isinstance(expression, ast.Dict):
        state = initial
        for key, value in zip(expression.keys, expression.values, strict=True):
            if key is None:
                state = _assignment_state_after_expression(
                    value,
                    assignment_name=assignment_name,
                    initial=state,
                    constant_resolver=constant_resolver,
                )
                return _StaticAssignmentState(ambiguous=True)
            state = _assignment_state_after_expression(
                key,
                assignment_name=assignment_name,
                initial=state,
                constant_resolver=constant_resolver,
            )
            state = _assignment_state_after_expression(
                value,
                assignment_name=assignment_name,
                initial=state,
                constant_resolver=constant_resolver,
            )
            if (
                _safe_constant_value(
                    key,
                    constant_resolver=constant_resolver,
                )
                is _UNKNOWN_CONSTANT
            ):
                return _StaticAssignmentState(ambiguous=True)
        return state

    if isinstance(expression, ast.Call):
        state = _assignment_state_after_expression(
            expression.func,
            assignment_name=assignment_name,
            initial=initial,
            constant_resolver=constant_resolver,
        )
        for argument in expression.args:
            value_expression = argument.value if isinstance(argument, ast.Starred) else argument
            state = _assignment_state_after_expression(
                value_expression,
                assignment_name=assignment_name,
                initial=state,
                constant_resolver=constant_resolver,
            )
            if isinstance(argument, ast.Starred):
                return _StaticAssignmentState(ambiguous=True)
        for keyword in expression.keywords:
            state = _assignment_state_after_expression(
                keyword.value,
                assignment_name=assignment_name,
                initial=state,
                constant_resolver=constant_resolver,
            )
            if keyword.arg is None:
                return _StaticAssignmentState(ambiguous=True)
        # Invoking even a syntactically simple callable can mutate module globals.
        return _StaticAssignmentState(ambiguous=True)

    if isinstance(expression, ast.Compare):
        state = _assignment_state_after_expression(
            expression.left,
            assignment_name=assignment_name,
            initial=initial,
            constant_resolver=constant_resolver,
        )

        def visit_comparators(
            index: int,
            left: ast.expr,
            current: _StaticAssignmentState,
        ) -> _StaticAssignmentState:
            comparator = expression.comparators[index]
            current = _assignment_state_after_expression(
                comparator,
                assignment_name=assignment_name,
                initial=current,
                constant_resolver=constant_resolver,
            )
            operator = expression.ops[index]
            if not _compare_operation_is_provably_side_effect_free(
                left,
                operator,
                comparator,
                constant_resolver=constant_resolver,
            ):
                return _StaticAssignmentState(ambiguous=True)
            if index == len(expression.comparators) - 1:
                return current

            comparison = ast.Compare(
                left=left,
                ops=[operator],
                comparators=[comparator],
            )
            value = _safe_truth_value(
                comparison,
                constant_resolver=constant_resolver,
            )
            if value is False:
                return current
            continued = visit_comparators(index + 1, comparator, current)
            if value is True:
                return continued
            return _merge_static_assignment_states(current, continued)

        return visit_comparators(0, expression.left, state)

    if isinstance(expression, ast.UnaryOp):
        state = _assignment_state_after_expression(
            expression.operand,
            assignment_name=assignment_name,
            initial=initial,
            constant_resolver=constant_resolver,
        )
        if isinstance(expression.op, ast.Not):
            is_safe = _truth_test_is_provably_side_effect_free(
                expression.operand,
                constant_resolver=constant_resolver,
            )
        else:
            is_safe = (
                _safe_constant_value(
                    expression,
                    constant_resolver=constant_resolver,
                )
                is not _UNKNOWN_CONSTANT
            )
        return state if is_safe else _StaticAssignmentState(ambiguous=True)

    if isinstance(expression, ast.BinOp):
        state = _assignment_state_after_expression(
            expression.left,
            assignment_name=assignment_name,
            initial=initial,
            constant_resolver=constant_resolver,
        )
        state = _assignment_state_after_expression(
            expression.right,
            assignment_name=assignment_name,
            initial=state,
            constant_resolver=constant_resolver,
        )
        if (
            _safe_constant_value(
                expression,
                constant_resolver=constant_resolver,
            )
            is _UNKNOWN_CONSTANT
        ):
            return _StaticAssignmentState(ambiguous=True)
        return state

    if isinstance(expression, ast.Attribute):
        _assignment_state_after_expression(
            expression.value,
            assignment_name=assignment_name,
            initial=initial,
            constant_resolver=constant_resolver,
        )
        return _StaticAssignmentState(ambiguous=True)

    if isinstance(expression, ast.Subscript):
        state = _assignment_state_after_expression(
            expression.value,
            assignment_name=assignment_name,
            initial=initial,
            constant_resolver=constant_resolver,
        )
        state = _assignment_state_after_expression(
            expression.slice,
            assignment_name=assignment_name,
            initial=state,
            constant_resolver=constant_resolver,
        )
        if (
            _safe_constant_value(
                expression,
                constant_resolver=constant_resolver,
            )
            is _UNKNOWN_CONSTANT
        ):
            return _StaticAssignmentState(ambiguous=True)
        return state

    if _may_rebind_entrypoint(
        expression,
        assignment_name,
        constant_resolver=constant_resolver,
    ):
        return _StaticAssignmentState(ambiguous=True)
    if not _expression_evaluation_is_provably_side_effect_free(
        expression,
        constant_resolver=constant_resolver,
    ):
        return _StaticAssignmentState(ambiguous=True)
    return initial


def _single_static_assignment_expression(
    contract: _ModuleContract,
    assignment_name: str,
    *,
    before_lineno: int | None,
    constant_resolver: Callable[[ast.Name], Any] | None = None,
) -> tuple[ast.expr | None, bool]:
    """Return the last deterministic binding that exists at a use site."""
    state = _StaticAssignmentState()
    for statement in contract.tree.body:
        if before_lineno is not None and statement.lineno >= before_lineno:
            break

        direct_value = _direct_assignment_value(statement, assignment_name)
        if direct_value is not None:
            state = _assignment_state_after_expression(
                direct_value,
                assignment_name=assignment_name,
                initial=state,
                constant_resolver=constant_resolver,
            )
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            target_count = sum(
                1
                for target in targets
                for node in ast.walk(target)
                if isinstance(node, ast.Name)
                and node.id == assignment_name
                and isinstance(node.ctx, (ast.Store, ast.Del))
            )
            if target_count == 1:
                # Assignment targets bind after their value expression has
                # finished evaluating, so this unconditional binding overrides
                # any earlier uncertainty or assignment-expression effect.
                state = _StaticAssignmentState(expression=direct_value)
            elif target_count:
                state = _StaticAssignmentState(ambiguous=True)
            continue

        if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr)):
            value = statement.value
            if value is not None and not (
                isinstance(statement, (ast.Assign, ast.AnnAssign))
                and _plain_local_class_constructor_is_provably_side_effect_free(
                    value,
                    known_classes=contract.classes,
                )
            ):
                state = _assignment_state_after_expression(
                    value,
                    assignment_name=assignment_name,
                    initial=state,
                    constant_resolver=constant_resolver,
                )
            continue

        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for expression in (
                *statement.decorator_list,
                *statement.args.defaults,
                *(value for value in statement.args.kw_defaults if value is not None),
            ):
                state = _assignment_state_after_expression(
                    expression,
                    assignment_name=assignment_name,
                    initial=state,
                    constant_resolver=constant_resolver,
                )
            if statement.decorator_list:
                # Applying an arbitrary decorator is an implicit call after the
                # decorator expressions and defaults have been evaluated.
                state = _StaticAssignmentState(ambiguous=True)
            if statement.name == assignment_name:
                state = _StaticAssignmentState(ambiguous=True)
            continue

        if isinstance(statement, ast.ClassDef):
            for expression in (
                *statement.decorator_list,
                *statement.bases,
                *(keyword.value for keyword in statement.keywords),
            ):
                state = _assignment_state_after_expression(
                    expression,
                    assignment_name=assignment_name,
                    initial=state,
                    constant_resolver=constant_resolver,
                )
            # Non-trivial class execution can invoke descriptors, metaclasses,
            # __init_subclass__, decorators, and arbitrary class-body code.
            if not _plain_class_definition_is_provably_side_effect_free(
                statement,
                known_classes=contract.classes,
            ):
                state = _StaticAssignmentState(ambiguous=True)
            if statement.name == assignment_name:
                state = _StaticAssignmentState(ambiguous=True)
            continue

        if _may_rebind_entrypoint(
            statement,
            assignment_name,
            constant_resolver=constant_resolver,
        ):
            state = _StaticAssignmentState(ambiguous=True)
        elif not isinstance(statement, ast.Pass):
            # Dynamic compound statements are conservatively opaque. A later
            # unconditional assignment still restores a precise state.
            state = _StaticAssignmentState(ambiguous=True)

    return state.expression, state.ambiguous


def _safe_constant_value_with_assignments(
    expression: ast.expr,
    *,
    current_module: str,
    modules: dict[str, _ModuleContract],
    seen_assignments: frozenset[_SymbolRef] = frozenset(),
) -> tuple[Any, bool]:
    """Resolve safe constants through one prior, unambiguous assignment."""
    ambiguous_assignment = False

    def resolve_name(name_expression: ast.Name) -> Any:
        nonlocal ambiguous_assignment
        reference = _resolve_reference(
            name_expression,
            current_module=current_module,
            modules=modules,
        )
        if reference is None:
            return _UNKNOWN_CONSTANT
        assignment_ref = _resolve_definition(
            reference,
            modules=modules,
            kind="assignments",
        )
        if assignment_ref is None:
            return _UNKNOWN_CONSTANT
        if assignment_ref in seen_assignments:
            ambiguous_assignment = True
            return _UNKNOWN_CONSTANT

        assignment_contract = modules[assignment_ref.module]
        before_lineno = name_expression.lineno if assignment_ref.module == current_module else None

        def assignment_constant_resolver(nested_name: ast.Name) -> Any:
            nonlocal ambiguous_assignment
            nested_value, nested_ambiguity = _safe_constant_value_with_assignments(
                nested_name,
                current_module=assignment_ref.module,
                modules=modules,
                seen_assignments=seen_assignments | {assignment_ref},
            )
            if nested_ambiguity:
                ambiguous_assignment = True
            return nested_value

        assigned_expression, assignment_is_ambiguous = _single_static_assignment_expression(
            assignment_contract,
            assignment_ref.name,
            before_lineno=before_lineno,
            constant_resolver=assignment_constant_resolver,
        )
        if assignment_is_ambiguous:
            ambiguous_assignment = True
            return _UNKNOWN_CONSTANT
        if assigned_expression is None:
            return _UNKNOWN_CONSTANT

        value, nested_ambiguity = _safe_constant_value_with_assignments(
            assigned_expression,
            current_module=assignment_ref.module,
            modules=modules,
            seen_assignments=seen_assignments | {assignment_ref},
        )
        if nested_ambiguity:
            ambiguous_assignment = True
        return value

    value = _safe_constant_value(
        expression,
        constant_resolver=resolve_name,
    )
    return value, ambiguous_assignment


def _entrypoint_binding_after(
    statements: list[ast.stmt],
    *,
    assignment_name: str,
    initial: _EntrypointBinding,
    constant_resolver: Callable[[ast.Name], Any] | None = None,
    known_classes: dict[str, ast.ClassDef] | None = None,
) -> _EntrypointBinding:
    """Resolve a module-level entrypoint binding along statically known paths."""

    def transfer_expression(
        expression: ast.expr,
        current: _EntrypointBinding,
    ) -> _EntrypointBinding:
        state = _assignment_state_after_expression(
            expression,
            assignment_name=assignment_name,
            initial=_StaticAssignmentState(
                expression=current.expression,
                ambiguous=current.ambiguous,
            ),
            constant_resolver=constant_resolver,
        )
        return _EntrypointBinding(
            expression=state.expression,
            ambiguous=state.ambiguous,
        )

    binding = initial
    for statement in statements:
        assignment_value = _direct_assignment_value(statement, assignment_name)
        if assignment_value is not None:
            binding = _entrypoint_binding_from_expression(
                assignment_value,
                constant_resolver=constant_resolver,
            )
            continue

        if not isinstance(statement, ast.If):
            if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Expr)):
                value = statement.value
                if value is not None and not (
                    isinstance(statement, (ast.Assign, ast.AnnAssign))
                    and _plain_local_class_constructor_is_provably_side_effect_free(
                        value,
                        known_classes=known_classes,
                    )
                ):
                    binding = transfer_expression(value, binding)
                targets = (
                    statement.targets
                    if isinstance(statement, ast.Assign)
                    else [statement.target]
                    if isinstance(statement, ast.AnnAssign)
                    else []
                )
                if any(
                    _may_rebind_entrypoint(
                        target,
                        assignment_name,
                        constant_resolver=constant_resolver,
                    )
                    for target in targets
                ):
                    binding = _EntrypointBinding(ambiguous=True)
                continue

            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for expression in (
                    *statement.decorator_list,
                    *statement.args.defaults,
                    *(value for value in statement.args.kw_defaults if value is not None),
                ):
                    binding = transfer_expression(expression, binding)
                if statement.decorator_list or statement.name == assignment_name:
                    binding = _EntrypointBinding(ambiguous=True)
                continue

            if isinstance(statement, ast.ClassDef):
                for expression in (
                    *statement.decorator_list,
                    *statement.bases,
                    *(keyword.value for keyword in statement.keywords),
                ):
                    binding = transfer_expression(expression, binding)
                if not _plain_class_definition_is_provably_side_effect_free(
                    statement,
                    known_classes=known_classes,
                ):
                    binding = _EntrypointBinding(ambiguous=True)
                continue

            if not isinstance(statement, ast.Pass) or _may_rebind_entrypoint(
                statement,
                assignment_name,
                constant_resolver=constant_resolver,
            ):
                binding = _EntrypointBinding(ambiguous=True)
            continue

        branch_initial = transfer_expression(
            statement.test,
            binding,
        )
        if not _truth_test_is_provably_side_effect_free(
            statement.test,
            constant_resolver=constant_resolver,
        ):
            branch_initial = _EntrypointBinding(ambiguous=True)
        test_value = _safe_truth_value(
            statement.test,
            constant_resolver=constant_resolver,
        )
        if test_value is not None:
            selected_branch = statement.body if test_value else statement.orelse
            binding = _entrypoint_binding_after(
                selected_branch,
                assignment_name=assignment_name,
                initial=branch_initial,
                constant_resolver=constant_resolver,
                known_classes=known_classes,
            )
            continue

        body_binding = _entrypoint_binding_after(
            statement.body,
            assignment_name=assignment_name,
            initial=branch_initial,
            constant_resolver=constant_resolver,
            known_classes=known_classes,
        )
        else_binding = _entrypoint_binding_after(
            statement.orelse,
            assignment_name=assignment_name,
            initial=branch_initial,
            constant_resolver=constant_resolver,
            known_classes=known_classes,
        )
        if _same_entrypoint_binding(body_binding, else_binding):
            binding = body_binding
        else:
            binding = _EntrypointBinding(ambiguous=True)
    return binding


def _module_entrypoint_expression(
    modules: dict[str, _ModuleContract],
    *,
    entrypoint_module: str,
    assignment_name: str,
) -> ast.expr | None:
    contract = modules[entrypoint_module]

    def constant_resolver(expression: ast.Name) -> Any:
        value, _ = _safe_constant_value_with_assignments(
            expression,
            current_module=entrypoint_module,
            modules=modules,
        )
        return value

    binding = _entrypoint_binding_after(
        contract.tree.body,
        assignment_name=assignment_name,
        initial=_EntrypointBinding(),
        constant_resolver=constant_resolver,
        known_classes=contract.classes,
    )
    if binding.ambiguous:
        return None
    return binding.expression


def _active_entrypoint_class_refs(
    modules: dict[str, _ModuleContract],
    *,
    entrypoint_module: str,
    assignment_name: str,
) -> set[_SymbolRef]:
    """Extract module-qualified classes instantiated by a plugin entrypoint."""
    entrypoint_contract = modules.get(entrypoint_module)
    if entrypoint_contract is None:
        return set()
    entrypoint_expression = _module_entrypoint_expression(
        modules,
        entrypoint_module=entrypoint_module,
        assignment_name=assignment_name,
    )
    if entrypoint_expression is None:
        return set()

    def assignment_expression(
        expression: ast.expr,
        current_module: str,
        seen_assignments: frozenset[_SymbolRef],
    ) -> tuple[
        tuple[ast.expr, str, frozenset[_SymbolRef]] | None,
        bool,
    ]:
        reference = _resolve_reference(
            expression,
            current_module=current_module,
            modules=modules,
        )
        if reference is None:
            return None, False
        assignment_ref = _resolve_definition(
            reference,
            modules=modules,
            kind="assignments",
        )
        if assignment_ref is None:
            return None, False
        if assignment_ref in seen_assignments:
            return None, True

        assignment_contract = modules[assignment_ref.module]
        before_lineno = expression.lineno if assignment_ref.module == current_module else None
        assigned_expression, assignment_is_ambiguous = _single_static_assignment_expression(
            assignment_contract,
            assignment_ref.name,
            before_lineno=before_lineno,
        )
        if assignment_is_ambiguous:
            return None, True
        if assigned_expression is None:
            return None, False
        return (
            (
                assigned_expression,
                assignment_ref.module,
                seen_assignments | {assignment_ref},
            ),
            False,
        )

    def trusted_environment_lookup(
        expression: ast.expr,
        current_module: str,
    ) -> bool:
        if not isinstance(expression, ast.Call):
            return False
        if (
            len(expression.args) != 1
            or expression.keywords
            or not isinstance(expression.args[0], ast.Constant)
            or not isinstance(expression.args[0].value, str)
        ):
            return False

        function = expression.func
        contract = modules[current_module]
        if (
            isinstance(function, ast.Attribute)
            and function.attr == "getenv"
            and isinstance(function.value, ast.Name)
        ):
            return contract.imported_modules.get(function.value.id) == "os"
        return (
            isinstance(function, ast.Attribute)
            and function.attr == "get"
            and isinstance(function.value, ast.Attribute)
            and function.value.attr == "environ"
            and isinstance(function.value.value, ast.Name)
            and contract.imported_modules.get(function.value.value.id) == "os"
        )

    def safe_primitive_expression(
        expression: ast.expr,
        current_module: str,
        seen_assignments: frozenset[_SymbolRef],
    ) -> bool:
        if isinstance(expression, ast.Constant):
            return isinstance(
                expression.value,
                (bool, bytes, complex, float, int, str, type(None)),
            )
        if trusted_environment_lookup(expression, current_module):
            return True
        if isinstance(expression, (ast.List, ast.Set, ast.Tuple)):
            return all(
                not isinstance(element, ast.Starred)
                and safe_primitive_expression(
                    element,
                    current_module,
                    seen_assignments,
                )
                for element in expression.elts
            )
        if isinstance(expression, ast.Dict):
            return all(
                key is not None
                and safe_primitive_expression(
                    key,
                    current_module,
                    seen_assignments,
                )
                and safe_primitive_expression(
                    value,
                    current_module,
                    seen_assignments,
                )
                for key, value in zip(
                    expression.keys,
                    expression.values,
                    strict=True,
                )
            )
        if isinstance(expression, ast.UnaryOp) and isinstance(
            expression.op,
            (ast.Invert, ast.UAdd, ast.USub),
        ):
            return safe_primitive_expression(
                expression.operand,
                current_module,
                seen_assignments,
            )
        if result_is_builtin_bool(
            expression,
            current_module,
            seen_assignments,
        ):
            return evaluation_is_side_effect_free(
                expression,
                current_module,
                seen_assignments,
            )

        assignment, assignment_is_ambiguous = assignment_expression(
            expression,
            current_module,
            seen_assignments,
        )
        if assignment_is_ambiguous or assignment is None:
            return False
        return safe_primitive_expression(*assignment)

    def result_is_builtin_bool(
        expression: ast.expr,
        current_module: str,
        seen_assignments: frozenset[_SymbolRef],
    ) -> bool:
        """Whether the value itself is provably an exact built-in bool."""
        if isinstance(expression, ast.Constant):
            return isinstance(expression.value, bool)
        if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
            return True
        if isinstance(expression, ast.Compare):
            return all(
                safe_primitive_expression(
                    value,
                    current_module,
                    seen_assignments,
                )
                for value in (expression.left, *expression.comparators)
            )
        if isinstance(expression, ast.BoolOp):
            return all(
                result_is_builtin_bool(value, current_module, seen_assignments)
                for value in expression.values
            )
        if isinstance(expression, ast.IfExp):
            return result_is_builtin_bool(
                expression.body,
                current_module,
                seen_assignments,
            ) and result_is_builtin_bool(
                expression.orelse,
                current_module,
                seen_assignments,
            )
        assignment, assignment_is_ambiguous = assignment_expression(
            expression,
            current_module,
            seen_assignments,
        )
        if assignment_is_ambiguous or assignment is None:
            return False
        return result_is_builtin_bool(*assignment)

    def evaluation_is_side_effect_free(
        expression: ast.expr,
        current_module: str,
        seen_assignments: frozenset[_SymbolRef],
    ) -> bool:
        """Whether evaluating the value can avoid plugin-defined Python code."""
        if isinstance(expression, ast.Constant):
            return True
        if trusted_environment_lookup(expression, current_module):
            return True
        if isinstance(expression, (ast.List, ast.Set, ast.Tuple)):
            return all(
                not isinstance(element, ast.Starred)
                and evaluation_is_side_effect_free(
                    element,
                    current_module,
                    seen_assignments,
                )
                for element in expression.elts
            )
        if isinstance(expression, ast.Dict):
            return all(
                key is not None
                and evaluation_is_side_effect_free(
                    key,
                    current_module,
                    seen_assignments,
                )
                and evaluation_is_side_effect_free(
                    value,
                    current_module,
                    seen_assignments,
                )
                for key, value in zip(
                    expression.keys,
                    expression.values,
                    strict=True,
                )
            )
        if isinstance(expression, ast.Compare):
            return all(
                safe_primitive_expression(
                    value,
                    current_module,
                    seen_assignments,
                )
                for value in (expression.left, *expression.comparators)
            )
        if isinstance(expression, ast.UnaryOp):
            return safe_primitive_expression(
                expression.operand,
                current_module,
                seen_assignments,
            )
        if isinstance(expression, ast.BoolOp):
            return all(
                safe_primitive_expression(
                    value,
                    current_module,
                    seen_assignments,
                )
                for value in expression.values
            )
        if isinstance(expression, ast.IfExp):
            return all(
                safe_primitive_expression(
                    value,
                    current_module,
                    seen_assignments,
                )
                for value in (
                    expression.test,
                    expression.body,
                    expression.orelse,
                )
            )

        assignment, assignment_is_ambiguous = assignment_expression(
            expression,
            current_module,
            seen_assignments,
        )
        if assignment_is_ambiguous or assignment is None:
            return False
        return evaluation_is_side_effect_free(*assignment)

    def stable_unknown_truth_test(
        expression: ast.expr,
        current_module: str,
        seen_assignments: frozenset[_SymbolRef],
    ) -> bool:
        """Allow unknown choices only when truth-testing cannot run user code."""
        if not isinstance(expression, ast.Name):
            return False
        assignment, assignment_is_ambiguous = assignment_expression(
            expression,
            current_module,
            seen_assignments,
        )
        if assignment_is_ambiguous or assignment is None:
            return False
        return result_is_builtin_bool(
            *assignment,
        ) and evaluation_is_side_effect_free(*assignment)

    def sequence_elements(
        expression: ast.expr,
        current_module: str,
        seen_assignments: frozenset[_SymbolRef],
    ) -> list[tuple[ast.expr, str, frozenset[_SymbolRef]]] | None:
        if isinstance(expression, (ast.List, ast.Tuple)):
            elements: list[tuple[ast.expr, str, frozenset[_SymbolRef]]] = []
            for element in expression.elts:
                if isinstance(element, ast.Starred):
                    expanded = sequence_elements(
                        element.value,
                        current_module,
                        seen_assignments,
                    )
                    if expanded is None:
                        return None
                    elements.extend(expanded)
                else:
                    elements.append(
                        (element, current_module, seen_assignments),
                    )
            return elements

        if isinstance(expression, ast.IfExp):
            test_value, assignment_is_ambiguous = _safe_constant_value_with_assignments(
                expression.test,
                current_module=current_module,
                modules=modules,
                seen_assignments=seen_assignments,
            )
            if assignment_is_ambiguous or test_value is _UNKNOWN_CONSTANT:
                return None
            selected = expression.body if bool(test_value) else expression.orelse
            return sequence_elements(selected, current_module, seen_assignments)

        assignment, assignment_is_ambiguous = assignment_expression(
            expression,
            current_module,
            seen_assignments,
        )
        if assignment_is_ambiguous or assignment is None:
            return None
        return sequence_elements(*assignment)

    def constant_index_value(
        expression: ast.expr,
        current_module: str,
        seen_assignments: frozenset[_SymbolRef],
    ) -> tuple[Any, bool]:
        return _safe_constant_value_with_assignments(
            expression,
            current_module=current_module,
            modules=modules,
            seen_assignments=seen_assignments,
        )

    def resolve_expression(
        expression: ast.expr,
        current_module: str,
        seen_assignments: frozenset[_SymbolRef],
    ) -> _EntrypointClassResolution:
        if isinstance(expression, ast.Call):
            assignment, assignment_is_ambiguous = assignment_expression(
                expression.func,
                current_module,
                seen_assignments,
            )
            if assignment_is_ambiguous:
                return _EntrypointClassResolution(ambiguous=True)
            if assignment is not None:
                factory_expression, factory_module, factory_seen = assignment
                return resolve_factory(
                    factory_expression,
                    factory_module,
                    factory_seen,
                )
            return resolve_factory(
                expression.func,
                current_module,
                seen_assignments,
            )

        if isinstance(expression, (ast.List, ast.Tuple)):
            elements = sequence_elements(
                expression,
                current_module,
                seen_assignments,
            )
            if elements is None:
                return _EntrypointClassResolution(ambiguous=True)
            active_classes: set[_SymbolRef] = set()
            for element, element_module, element_seen in elements:
                resolution = resolve_expression(
                    element,
                    element_module,
                    element_seen,
                )
                if resolution.ambiguous or resolution.is_collection or len(resolution.classes) != 1:
                    return _EntrypointClassResolution(ambiguous=True)
                active_classes.update(resolution.classes)
            return _EntrypointClassResolution(
                classes=frozenset(active_classes),
                is_collection=True,
            )

        if isinstance(expression, ast.IfExp):
            test_value, assignment_is_ambiguous = _safe_constant_value_with_assignments(
                expression.test,
                current_module=current_module,
                modules=modules,
                seen_assignments=seen_assignments,
            )
            if assignment_is_ambiguous:
                return _EntrypointClassResolution(ambiguous=True)
            if test_value is not _UNKNOWN_CONSTANT:
                selected = expression.body if bool(test_value) else expression.orelse
                return resolve_expression(selected, current_module, seen_assignments)

            if not stable_unknown_truth_test(
                expression.test,
                current_module,
                seen_assignments,
            ):
                return _EntrypointClassResolution(ambiguous=True)
            body_resolution = resolve_expression(
                expression.body,
                current_module,
                seen_assignments,
            )
            else_resolution = resolve_expression(
                expression.orelse,
                current_module,
                seen_assignments,
            )
            if (
                body_resolution.ambiguous
                or else_resolution.ambiguous
                or body_resolution != else_resolution
            ):
                return _EntrypointClassResolution(ambiguous=True)
            return body_resolution

        if isinstance(expression, ast.Subscript):
            if isinstance(expression.slice, ast.Slice):
                return _EntrypointClassResolution(ambiguous=True)
            elements = sequence_elements(
                expression.value,
                current_module,
                seen_assignments,
            )
            if elements is None:
                return _EntrypointClassResolution(ambiguous=True)
            index, assignment_is_ambiguous = constant_index_value(
                expression.slice,
                current_module,
                seen_assignments,
            )
            if assignment_is_ambiguous:
                return _EntrypointClassResolution(ambiguous=True)
            if index is not _UNKNOWN_CONSTANT:
                if not isinstance(index, int) or not (-len(elements) <= index < len(elements)):
                    return _EntrypointClassResolution(ambiguous=True)
                return resolve_expression(*elements[index])

            if not isinstance(expression.slice, ast.Name) or not elements:
                return _EntrypointClassResolution(ambiguous=True)
            candidate_class: _SymbolRef | None = None
            for element, element_module, element_seen in elements:
                resolution = resolve_expression(
                    element,
                    element_module,
                    element_seen,
                )
                if resolution.ambiguous or resolution.is_collection or len(resolution.classes) != 1:
                    return _EntrypointClassResolution(ambiguous=True)
                (current_class,) = resolution.classes
                if candidate_class is None:
                    candidate_class = current_class
                elif current_class != candidate_class:
                    return _EntrypointClassResolution(ambiguous=True)
            return _EntrypointClassResolution(
                classes=frozenset({candidate_class}),
            )

        assignment, assignment_is_ambiguous = assignment_expression(
            expression,
            current_module,
            seen_assignments,
        )
        if assignment_is_ambiguous:
            return _EntrypointClassResolution(ambiguous=True)
        if assignment is not None:
            return resolve_expression(*assignment)
        return _EntrypointClassResolution(ambiguous=True)

    def resolve_factory(
        expression: ast.expr,
        current_module: str,
        seen_assignments: frozenset[_SymbolRef],
    ) -> _EntrypointClassResolution:
        if isinstance(expression, ast.IfExp):
            test_value, assignment_is_ambiguous = _safe_constant_value_with_assignments(
                expression.test,
                current_module=current_module,
                modules=modules,
                seen_assignments=seen_assignments,
            )
            if assignment_is_ambiguous:
                return _EntrypointClassResolution(ambiguous=True)
            if test_value is not _UNKNOWN_CONSTANT:
                selected = expression.body if bool(test_value) else expression.orelse
                return resolve_factory(selected, current_module, seen_assignments)
            if not stable_unknown_truth_test(
                expression.test,
                current_module,
                seen_assignments,
            ):
                return _EntrypointClassResolution(ambiguous=True)
            body_resolution = resolve_factory(
                expression.body,
                current_module,
                seen_assignments,
            )
            else_resolution = resolve_factory(
                expression.orelse,
                current_module,
                seen_assignments,
            )
            if (
                body_resolution.ambiguous
                or else_resolution.ambiguous
                or body_resolution != else_resolution
            ):
                return _EntrypointClassResolution(ambiguous=True)
            return body_resolution

        if isinstance(expression, ast.Subscript):
            if isinstance(expression.slice, ast.Slice):
                return _EntrypointClassResolution(ambiguous=True)
            elements = sequence_elements(
                expression.value,
                current_module,
                seen_assignments,
            )
            if elements is None:
                return _EntrypointClassResolution(ambiguous=True)
            index, assignment_is_ambiguous = constant_index_value(
                expression.slice,
                current_module,
                seen_assignments,
            )
            if assignment_is_ambiguous:
                return _EntrypointClassResolution(ambiguous=True)
            if index is not _UNKNOWN_CONSTANT:
                if not isinstance(index, int) or not (-len(elements) <= index < len(elements)):
                    return _EntrypointClassResolution(ambiguous=True)
                return resolve_factory(*elements[index])
            if not isinstance(expression.slice, ast.Name) or not elements:
                return _EntrypointClassResolution(ambiguous=True)
            candidate: _EntrypointClassResolution | None = None
            for element in elements:
                resolution = resolve_factory(*element)
                if resolution.ambiguous:
                    return _EntrypointClassResolution(ambiguous=True)
                if candidate is None:
                    candidate = resolution
                elif resolution != candidate:
                    return _EntrypointClassResolution(ambiguous=True)
            return candidate or _EntrypointClassResolution(ambiguous=True)

        assignment, assignment_is_ambiguous = assignment_expression(
            expression,
            current_module,
            seen_assignments,
        )
        if assignment_is_ambiguous:
            return _EntrypointClassResolution(ambiguous=True)
        if assignment is not None:
            return resolve_factory(*assignment)

        reference = _resolve_reference(
            expression,
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
                return _EntrypointClassResolution(
                    classes=frozenset({class_ref}),
                )
        return _EntrypointClassResolution(ambiguous=True)

    resolution = resolve_expression(
        entrypoint_expression,
        entrypoint_module,
        frozenset(),
    )
    return set() if resolution.ambiguous else set(resolution.classes)


def _active_runner_class_refs(
    modules: dict[str, _ModuleContract],
) -> set[_SymbolRef]:
    return _active_entrypoint_class_refs(
        modules,
        entrypoint_module="runner",
        assignment_name="runner",
    )


def _active_comparator_class_refs(
    modules: dict[str, _ModuleContract],
) -> set[_SymbolRef]:
    return _active_entrypoint_class_refs(
        modules,
        entrypoint_module="comparator",
        assignment_name="comparator",
    )


def _class_lineage(
    class_reference: _SymbolRef,
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

    visit(class_reference)
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
    """Map runtimes to native commands shared by every manifest owner."""
    runtime_owner_commands: dict[str, dict[Path, set[str]]] = {}
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
        owner_commands = runtime_owner_commands.setdefault(runtime_strategy, {}).setdefault(
            owner_dir,
            set(),
        )
        declared_runner_class = _canonical_runner_class_ref(runner_class_ref)
        if declared_runner_class is None:
            continue
        matching_active_classes = {
            active_class
            for active_class in active_runner_classes
            if declared_runner_class in _class_lineage(active_class, modules)
        }
        owner_commands.update(
            command
            for active_class in matching_active_classes
            for command in _commands_for_runner_class(
                active_class,
                modules=modules,
                native_cli_commands=native_cli_commands,
            )
        )

    runtime_commands: dict[str, set[str]] = {}
    for runtime_strategy, commands_by_owner in runtime_owner_commands.items():
        owner_command_sets = iter(commands_by_owner.values())
        shared_commands = set(next(owner_command_sets))
        for owner_commands in owner_command_sets:
            shared_commands.intersection_update(owner_commands)
        runtime_commands[runtime_strategy] = shared_commands
    return runtime_commands


def extract_active_comparator_classes_by_runtime(
    models_dir: Path,
) -> dict[str, set[str]]:
    """Map each manifest runtime to its owner's active comparator lineage."""
    runtime_classes: dict[str, set[str]] = {}
    owner_contracts: dict[
        Path,
        tuple[dict[str, _ModuleContract], set[_SymbolRef]],
    ] = {}

    for manifest_path in _iter_e2e_manifest_paths(models_dir):
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        runtime_strategy = raw.get("runtime_strategy")
        task_strategy = raw.get("task_strategy")
        if not _is_nonempty_str(runtime_strategy) or not _is_nonempty_str(task_strategy):
            continue

        if manifest_path.parent.name == "manifests":
            owner_dir = manifest_path.parent.parent
        else:
            family = raw.get("family")
            if not _is_nonempty_str(family):
                continue
            owner_dir = models_dir / family

        if owner_dir not in owner_contracts:
            modules = _plugin_contract_modules(owner_dir / "e2e_plugins")
            owner_contracts[owner_dir] = (
                modules,
                _active_comparator_class_refs(modules),
            )
        modules, active_comparator_classes = owner_contracts[owner_dir]
        owner_runtime_lineage: set[str] = set()
        for active_class in active_comparator_classes:
            lineage = _class_lineage(active_class, modules)
            lineage_tasks = {
                task
                for class_ref in lineage
                if (
                    task := _extract_constant_return(
                        modules[class_ref.module].classes[class_ref.name],
                        "task_strategy",
                    )
                )
                is not None
            }
            if task_strategy not in lineage_tasks:
                continue
            owner_runtime_lineage.update(class_ref.qualified_name() for class_ref in lineage)
        if runtime_strategy not in runtime_classes:
            runtime_classes[runtime_strategy] = owner_runtime_lineage
        else:
            # One matrix row governs every owner of a runtime. Requiring the
            # shared intersection prevents one owner from lending its active
            # comparator to another owner that does not expose that lineage.
            runtime_classes[runtime_strategy].intersection_update(owner_runtime_lineage)

    return runtime_classes


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


def _qualified_plugin_class_name(
    file_path: Path,
    *,
    kind: str,
    class_name: str,
) -> str:
    entrypoint_module = "runner" if kind == "runners" else "comparator"
    if file_path.name == f"{entrypoint_module}.py":
        module = entrypoint_module
    else:
        module = f"{kind}.{file_path.stem}"
    return f"{module}.{class_name}"


def _extract_class_map_by_method(
    root: Path,
    kind: str,
    method_name: str,
) -> dict[str, set[str]]:
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
            mapping.setdefault(key, set()).add(
                _qualified_plugin_class_name(
                    file_path,
                    kind=kind,
                    class_name=node.name,
                )
            )
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
    active_comparator_classes_by_runtime: dict[str, set[str]],
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
            canonical_runner_ref = _canonical_runner_class_ref(runner_class_ref)
            if not expected_runner_classes:
                errors.append(
                    f"{runtime_strategy}: no runner class found for task_strategy '{expected_task}'."
                )
            elif (
                canonical_runner_ref is None
                or canonical_runner_ref.qualified_name() not in expected_runner_classes
            ):
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
            canonical_comparator_ref = _canonical_comparator_class_ref(comparator_class_ref)
            if not expected_comparator_classes:
                errors.append(
                    f"{runtime_strategy}: no comparator class found for task_strategy '{expected_task}'."
                )
            elif (
                canonical_comparator_ref is None
                or canonical_comparator_ref.qualified_name() not in expected_comparator_classes
            ):
                errors.append(
                    f"{runtime_strategy}: comparator_class '{comparator_class_ref}' "
                    f"not in discovered comparator classes {sorted(expected_comparator_classes)}."
                )
            if runtime_strategy in manifest_strategies:
                active_owner_comparators = active_comparator_classes_by_runtime.get(
                    runtime_strategy, set()
                )
                if (
                    canonical_comparator_ref is None
                    or canonical_comparator_ref.qualified_name() not in active_owner_comparators
                ):
                    errors.append(
                        f"{runtime_strategy}: comparator_class '{comparator_class_ref}' "
                        "is not in the active comparator lineage for its manifest "
                        "owner/runtime; discovered active comparator classes: "
                        f"{sorted(active_owner_comparators)}."
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
    active_comparator_classes_by_runtime = extract_active_comparator_classes_by_runtime(
        e2e_models_dir
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
        active_comparator_classes_by_runtime=active_comparator_classes_by_runtime,
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
