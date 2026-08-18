#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inventory and audit model-family specialization boundaries.

The auditor is development tooling only.  It never becomes an import surface
for family implementations: every runtime implementation remains physically
owned by its model family.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tomllib
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
FAMILIES_PACKAGE = "tensorrt_model_connect.families"
APPROVED_FAMILIES_ROOT_IMPORTS = frozenset()
MISPLACED_MODEL_PATHS = frozenset(
    {
        "runtime_config_schema.py",
        "python_profile_verify.py",
        "python_profile_requirements",
    }
)
STRATEGY_PARAMETERS = frozenset(
    {
        "activation",
        "alibi_bias_scale",
        "cross_attn_norm",
        "debug_layer_outputs",
        "embed_input",
        "ffn_activation",
        "hidden_state_output",
        "interleaved_rope",
        "mlp_type",
        "norm_type",
        "parallel_residual",
        "partial_rotary_factor",
        "position_type",
        "qk_norm",
        "scale_attn_weights",
        "use_rope",
    }
)
TEXT_SCAN_SUFFIXES = frozenset({".json", ".py", ".sh", ".toml", ".yaml", ".yml"})
Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


@dataclass(frozen=True)
class ModuleInfo:
    module: str
    path: Path
    relative_path: str
    tree: ast.Module
    is_init: bool
    imports: frozenset[str]
    symbol_imports: frozenset[tuple[str, str]]
    module_aliases: tuple[tuple[str, str], ...]
    definitions: tuple[str, ...]


@dataclass(frozen=True)
class DynamicEntryPoint:
    source: str
    path: str
    symbol: str | None
    exists: bool
    module: str | None
    symbol_exists: bool | None


def family_dirs(repo_root: Path, selected: tuple[str, ...]) -> list[Path]:
    root = repo_root / "python/tensorrt_model_connect/families"
    discovered = sorted(
        path for path in root.iterdir() if (path / "model.py").is_file()
    )
    if not selected:
        return discovered
    by_name = {path.name: path for path in discovered}
    missing = sorted(set(selected) - by_name.keys())
    if missing:
        raise SystemExit("Unknown model family: " + ", ".join(missing))
    return [by_name[name] for name in selected]


def _module_name(repo_root: Path, path: Path) -> tuple[str, bool] | None:
    try:
        relative = path.relative_to(repo_root / "python")
    except ValueError:
        return None
    parts = list(relative.parts)
    is_init = parts[-1] == "__init__.py"
    parts[-1] = Path(parts[-1]).stem
    if is_init:
        parts.pop()
    return ".".join(parts), is_init


def _resolve_from_module(
    current_module: str | None,
    current_is_init: bool,
    node: ast.ImportFrom,
) -> str:
    if node.level == 0:
        return node.module or ""
    if current_module is None:
        return ""
    package = current_module if current_is_init else current_module.rpartition(".")[0]
    parts = package.split(".") if package else []
    parents = node.level - 1
    if parents > len(parts):
        return ""
    base = parts[: len(parts) - parents]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _parse_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _module_references(
    tree: ast.Module,
    *,
    current_module: str | None,
    current_is_init: bool,
    known_modules: frozenset[str],
) -> tuple[set[str], set[tuple[str, str]], dict[str, str]]:
    imports: set[str] = set()
    symbol_imports: set[tuple[str, str]] = set()
    aliases: dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = alias.name
                if target in known_modules:
                    imports.add(target)
                    aliases[alias.asname or target.rsplit(".", 1)[-1]] = target
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        base = _resolve_from_module(current_module, current_is_init, node)
        if base in known_modules:
            imports.add(base)
        for alias in node.names:
            if alias.name == "*":
                if base in known_modules:
                    symbol_imports.add((base, "*"))
                continue
            child = f"{base}.{alias.name}" if base else alias.name
            if child in known_modules:
                imports.add(child)
                aliases[alias.asname or alias.name] = child
            elif base in known_modules:
                symbol_imports.add((base, alias.name))

    return imports, symbol_imports, aliases


def _family_modules(repo_root: Path, family_dir: Path) -> dict[str, ModuleInfo]:
    paths = sorted(
        path
        for path in family_dir.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    names: dict[Path, tuple[str, bool]] = {}
    for path in paths:
        result = _module_name(repo_root, path)
        if result is not None:
            names[path] = result
    known = frozenset(module for module, _ in names.values())
    modules: dict[str, ModuleInfo] = {}
    for path, (module, is_init) in names.items():
        tree = _parse_tree(path)
        imports, symbol_imports, aliases = _module_references(
            tree,
            current_module=module,
            current_is_init=is_init,
            known_modules=known,
        )
        definitions = tuple(
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        )
        modules[module] = ModuleInfo(
            module=module,
            path=path,
            relative_path=path.relative_to(family_dir).as_posix(),
            tree=tree,
            is_init=is_init,
            imports=frozenset(imports),
            symbol_imports=frozenset(symbol_imports),
            module_aliases=tuple(sorted(aliases.items())),
            definitions=definitions,
        )
    return modules


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)


def _manifest_path(repo_root: Path, family_dir: Path, raw: str) -> Path:
    path = Path(raw)
    if raw.startswith("families/"):
        return repo_root / "python/tensorrt_model_connect" / path
    if raw.startswith("model/") or raw.startswith("weights/"):
        return family_dir / path
    return family_dir / path


def _assigned_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_assigned_names(child) for child in target.elts))
    return set()


def _module_bound_names(info: ModuleInfo) -> set[str]:
    names = set(info.definitions)
    for node in info.tree.body:
        if isinstance(node, ast.Assign):
            names.update(
                set().union(*(_assigned_names(target) for target in node.targets))
            )
        elif isinstance(node, ast.AnnAssign):
            names.update(_assigned_names(node.target))
        elif isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names)
    return names


def _dynamic_entrypoints(
    repo_root: Path,
    family_dir: Path,
    modules: dict[str, ModuleInfo],
) -> tuple[list[DynamicEntryPoint], set[str], dict[str, set[str]]]:
    manifest_path = family_dir / "MODEL.toml"
    manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
    module_by_path = {info.path.resolve(): name for name, info in modules.items()}
    roots: set[str] = set()
    symbols: dict[str, set[str]] = defaultdict(set)
    entries: list[DynamicEntryPoint] = []

    model_path = family_dir / "model.py"
    model_name = f"{FAMILIES_PACKAGE}.{family_dir.name}.model"
    model_exists = model_name in modules
    model_symbols = _module_bound_names(modules[model_name]) if model_exists else set()
    for required_symbol in ("matches", "build"):
        entries.append(
            DynamicEntryPoint(
                source="family model convention",
                path="model.py",
                symbol=required_symbol,
                exists=model_path.is_file(),
                module=model_name if model_exists else None,
                symbol_exists=(required_symbol in model_symbols) if model_exists else None,
            )
        )
    if model_exists:
        roots.add(model_name)
        symbols[model_name].update({"matches", "build"})

    seen: set[tuple[str, str | None]] = set()
    for value in _walk_strings(manifest):
        parts = [part.strip() for part in value.split("|")]
        for index, part in enumerate(parts):
            if "*" in part or "?" in part:
                continue
            if not re.search(r"\.(?:py|txt)$", part):
                continue
            symbol = None
            if part.endswith(".py") and index + 1 < len(parts):
                candidate = parts[index + 1]
                if candidate.isidentifier() and candidate not in {"true", "false"}:
                    symbol = candidate
            key = (part, symbol)
            if key in seen:
                continue
            seen.add(key)
            path = _manifest_path(repo_root, family_dir, part)
            module = module_by_path.get(path.resolve()) if path.exists() else None
            symbol_exists = None
            if symbol is not None:
                symbol_exists = bool(
                    module and symbol in _module_bound_names(modules[module])
                )
            entries.append(
                DynamicEntryPoint(
                    source="MODEL.toml",
                    path=part,
                    symbol=symbol,
                    exists=path.is_file(),
                    module=module,
                    symbol_exists=symbol_exists,
                )
            )
            if module:
                roots.add(module)
                if symbol:
                    symbols[module].add(symbol)

    schema = family_dir / "runtime_config_schema.py"
    if schema.is_file():
        module = module_by_path.get(schema.resolve())
        entries.append(
            DynamicEntryPoint(
                source="runtime_config.schemas",
                path="runtime_config_schema.py",
                symbol=None,
                exists=True,
                module=module,
                symbol_exists=None,
            )
        )
        if module:
            roots.add(module)

    return sorted(entries, key=lambda item: (item.source, item.path, item.symbol or "")), roots, symbols


def _scan_paths(repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    for relative in ("tests", "tools", "scripts", "examples", ".github"):
        root = repo_root / relative
        if not root.is_dir():
            continue
        paths.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in TEXT_SCAN_SUFFIXES
            and "__pycache__" not in path.parts
            and ".family-source-validation" not in path.parts
        )
    return sorted(paths)


def _external_module_roots(
    repo_root: Path,
    family_dir: Path,
    modules: dict[str, ModuleInfo],
    scan_paths: list[Path],
) -> tuple[set[str], dict[str, set[str]], dict[str, list[str]]]:
    known = frozenset(modules)
    roots: set[str] = set()
    symbols: dict[str, set[str]] = defaultdict(set)
    sources: dict[str, list[str]] = defaultdict(list)
    family_needle = f"{FAMILIES_PACKAGE}.{family_dir.name}."

    for path in scan_paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if family_needle not in text and family_dir.name not in path.parts:
            continue
        found: set[str] = set()
        if path.suffix == ".py":
            try:
                tree = ast.parse(text, filename=str(path))
            except SyntaxError:
                tree = None
            if tree is not None:
                imported, imported_symbols, _ = _module_references(
                    tree,
                    current_module=None,
                    current_is_init=False,
                    known_modules=known,
                )
                found.update(imported)
                for module, symbol in imported_symbols:
                    symbols[module].add(symbol)
                    found.add(module)
        for module in sorted(known, key=len, reverse=True):
            matches = re.finditer(
                rf"(?<![A-Za-z0-9_.]){re.escape(module)}"
                r"(?:\.([A-Za-z_][A-Za-z0-9_]*))?",
                text,
            )
            for match in matches:
                suffix = match.group(1)
                if suffix and f"{module}.{suffix}" in known:
                    continue
                found.add(module)
                if suffix:
                    symbols[module].add(suffix)
        for module in sorted(found):
            roots.add(module)
            sources[module].append(path.relative_to(repo_root).as_posix())

    return roots, symbols, {key: sorted(set(value)) for key, value in sources.items()}


def _convention_tool_roots(
    repo_root: Path,
    family_dir: Path,
    modules: dict[str, ModuleInfo],
) -> tuple[set[str], dict[str, list[str]], list[dict[str, str]]]:
    """Resolve generic tools that select family modules from runtime metadata."""
    roots: set[str] = set()
    sources: dict[str, list[str]] = defaultdict(list)
    missing: list[dict[str, str]] = []
    model_text = (family_dir / "model.py").read_text(encoding="utf-8")
    is_vision_language = bool(
        re.search(
            r"runtime_strategy\s*=\s*['\"][A-Za-z0-9_]+_vision_language['\"]",
            model_text,
        )
    )
    manifests = repo_root / "tests/e2e/models" / family_dir.name / "manifests"
    if manifests.is_dir() and not is_vision_language:
        for path in manifests.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if str(payload.get("runtime_strategy") or "").endswith("_vision_language"):
                is_vision_language = True
                break

    if is_vision_language:
        tool_path = (
            repo_root
            / "tools/families"
            / family_dir.name
            / "vl_debug_runner.py"
        )
        family_path = family_dir / "vl_debug_runner.py"
        if family_path.is_file():
            module = f"{FAMILIES_PACKAGE}.{family_dir.name}.vl_debug_runner"
            if module in modules:
                roots.add(module)
                sources[module].append("tools/diff_vl.py::<family-dispatch>")
        elif not tool_path.is_file():
            missing.append(
                {
                    "source": "tools/diff_vl.py::<family-dispatch>",
                    "path": f"tools/families/{family_dir.name}/vl_debug_runner.py",
                    "reason": "missing_path",
                }
            )
    return roots, sources, missing


def _closure(modules: dict[str, ModuleInfo], seeds: Iterable[str]) -> set[str]:
    reachable: set[str] = set()
    pending = deque(seed for seed in seeds if seed in modules)
    while pending:
        module = pending.popleft()
        if module in reachable:
            continue
        reachable.add(module)
        pending.extend(modules[module].imports - reachable)
    return reachable


def _definition_nodes(tree: ast.Module) -> dict[str, list[Definition]]:
    result: dict[str, list[Definition]] = defaultdict(list)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            result[node.name].append(node)
    return result


def _loaded_local_names(node: ast.AST, names: set[str]) -> set[str]:
    loaded = {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name)
        and isinstance(child.ctx, ast.Load)
        and child.id in names
    }
    loaded.update(
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant)
        and isinstance(child.value, str)
        and child.value in names
    )
    return loaded


def _attribute_symbol_roots(info: ModuleInfo) -> dict[str, set[str]]:
    aliases = dict(info.module_aliases)
    roots: dict[str, set[str]] = defaultdict(set)
    for node in ast.walk(info.tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
        ):
            roots[aliases[node.value.id]].add(node.attr)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in aliases
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            roots[aliases[node.args[0].id]].add(node.args[1].value)
    return roots


def _self_module_symbol_roots(info: ModuleInfo) -> set[str]:
    """Resolve aliases such as ``graph_ops = sys.modules[__name__]``."""
    aliases: set[str] = set()
    for node in info.tree.body:
        if not isinstance(node, ast.Assign):
            continue
        value = node.value
        if not (
            isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Attribute)
            and isinstance(value.value.value, ast.Name)
            and value.value.value.id == "sys"
            and value.value.attr == "modules"
            and isinstance(value.slice, ast.Name)
            and value.slice.id == "__name__"
        ):
            continue
        aliases.update(
            target.id for target in node.targets if isinstance(target, ast.Name)
        )
    return {
        node.attr
        for node in ast.walk(info.tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in aliases
    }


def _unreachable_symbols(
    modules: dict[str, ModuleInfo],
    production_modules: set[str],
    dynamic_symbols: dict[str, set[str]],
) -> list[dict[str, Any]]:
    external_roots: dict[str, set[str]] = defaultdict(set)
    for source_name in production_modules:
        source = modules[source_name]
        for target, symbol in source.symbol_imports:
            if target in production_modules:
                if symbol == "*":
                    external_roots[target].update(modules[target].definitions)
                else:
                    external_roots[target].add(symbol)
        for target, names in _attribute_symbol_roots(source).items():
            if target in production_modules:
                external_roots[target].update(names)
    for module, names in dynamic_symbols.items():
        external_roots[module].update(names)

    results: list[dict[str, Any]] = []
    for module_name in sorted(production_modules):
        info = modules[module_name]
        definitions = _definition_nodes(info.tree)
        if not definitions:
            continue
        names = set(definitions)
        dependencies = {
            name: set().union(
                *(_loaded_local_names(node, names) for node in nodes)
            )
            for name, nodes in definitions.items()
        }
        roots = set(external_roots[module_name]) & names
        roots.update(_self_module_symbol_roots(info) & names)
        for node in info.tree.body:
            if not isinstance(
                node,
                (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom),
            ):
                roots.update(_loaded_local_names(node, names))
        pending = deque(sorted(roots))
        reachable = set(roots)
        while pending:
            for dependency in dependencies[pending.popleft()]:
                if dependency not in reachable:
                    reachable.add(dependency)
                    pending.append(dependency)

        for name in sorted(names - reachable):
            nodes = definitions[name]
            results.append(
                {
                    "module": module_name,
                    "path": info.relative_path,
                    "symbol": name,
                    "line": min(node.lineno for node in nodes),
                }
            )
    return results


def _literal_value(node: ast.AST) -> Any:
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return {"dynamic": ast.unparse(node)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return {"literal": repr(value)}


def _strategy_switches(
    modules: dict[str, ModuleInfo],
    production_modules: set[str],
) -> list[dict[str, Any]]:
    definitions: dict[str, list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]] = (
        defaultdict(list)
    )
    for module_name in production_modules:
        for node in modules[module_name].tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions[node.name].append((module_name, node))

    calls: dict[str, list[tuple[ast.Call, str]]] = defaultdict(list)
    for module_name in production_modules:
        info = modules[module_name]
        for node in ast.walk(info.tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                function = node.func.id
            elif isinstance(node.func, ast.Attribute):
                function = node.func.attr
            else:
                continue
            if function not in definitions:
                continue
            calls[function].append((node, info.relative_path))

    result: list[dict[str, Any]] = []
    candidates = sorted({
        (function, parameter)
        for function, owners in definitions.items()
        for _, node in owners
        for parameter in {
            arg.arg for arg in (*node.args.args, *node.args.kwonlyargs)
        } & STRATEGY_PARAMETERS
    })
    for function, parameter in candidates:
        function_calls = calls.get(function, [])
        values: list[tuple[Any, str, int]] = []
        for call, path in function_calls:
            matching = [kw for kw in call.keywords if kw.arg == parameter]
            if len(matching) != 1:
                values = []
                break
            values.append(
                (_literal_value(matching[0].value), path, call.lineno)
            )
        if not values:
            continue
        serialized = {json.dumps(value, sort_keys=True) for value, _, _ in values}
        if len(serialized) != 1:
            continue
        value = values[0][0]
        if isinstance(value, dict) and "dynamic" in value:
            continue
        owners = [
            module
            for module, node in definitions[function]
            if parameter in {arg.arg for arg in (*node.args.args, *node.args.kwonlyargs)}
        ]
        if not owners:
            continue
        result.append(
            {
                "function": function,
                "parameter": parameter,
                "value": value,
                "definitions": sorted(owners),
                "call_sites": [
                    {"path": path, "line": line}
                    for _, path, line in sorted(values, key=lambda item: (item[1], item[2]))
                ],
            }
        )
    return result


def _sibling_imports(family: str, modules: dict[str, ModuleInfo]) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    prefix = f"{FAMILIES_PACKAGE}."
    for info in modules.values():
        for node in ast.walk(info.tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                target = _resolve_from_module(info.module, info.is_init, node)
                if target:
                    targets.append(target)
            for target in targets:
                if not target.startswith(prefix):
                    continue
                suffix = target[len(prefix):]
                owner = suffix.split(".", 1)[0]
                if owner == family or owner in APPROVED_FAMILIES_ROOT_IMPORTS:
                    continue
                violations.append(
                    {
                        "path": info.relative_path,
                        "line": node.lineno,
                        "target": target,
                    }
                )
    return sorted(violations, key=lambda item: (item["path"], item["line"]))


def _noncanonical_model_paths(family_dir: Path) -> list[str]:
    result = []
    if not (family_dir / "model.py").is_file():
        result.append("model.py")
    if (family_dir / "model").exists():
        result.append("model/")
    if (family_dir / "plugin.py").exists():
        result.append("plugin.py")
    return result


def _source_metrics(family_dir: Path) -> dict[str, int]:
    files = [
        path
        for path in family_dir.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    model_files = [path for path in files if path == family_dir / "model.py"]

    def lines(paths: list[Path]) -> int:
        total = 0
        for path in paths:
            try:
                total += len(path.read_text(encoding="utf-8").splitlines())
            except UnicodeDecodeError:
                pass
        return total

    return {
        "files": len(files),
        "lines": lines(files),
        "bytes": sum(path.stat().st_size for path in files),
        "model_files": len(model_files),
        "model_lines": lines(model_files),
        "model_bytes": sum(path.stat().st_size for path in model_files),
    }


def audit_family(
    repo_root: Path,
    family_dir: Path,
    *,
    scan_paths: list[Path] | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    family_dir = family_dir.resolve()
    family = family_dir.name
    modules = _family_modules(repo_root, family_dir)
    package_module = f"{FAMILIES_PACKAGE}.{family}"
    dynamic_entries, dynamic_roots, dynamic_symbols = _dynamic_entrypoints(
        repo_root, family_dir, modules
    )
    external_roots, external_symbols, external_sources = _external_module_roots(
        repo_root,
        family_dir,
        modules,
        scan_paths if scan_paths is not None else _scan_paths(repo_root),
    )
    convention_roots, convention_sources, missing_conventions = _convention_tool_roots(
        repo_root, family_dir, modules
    )
    external_roots.update(convention_roots)
    for module, sources in convention_sources.items():
        external_sources[module] = sorted(
            set(external_sources.get(module, ())) | set(sources)
        )

    production_seeds = {package_module, *dynamic_roots}
    production_modules = _closure(modules, production_seeds)
    production_packages = {
        module_name
        for module_name, info in modules.items()
        if info.is_init
        and any(
            child.startswith(module_name + ".")
            for child in production_modules
        )
    }
    production_modules = _closure(
        modules, production_modules | production_packages
    )
    tool_modules = _closure(modules, external_roots) - production_modules
    unreachable_modules = set(modules) - production_modules - tool_modules

    symbol_roots: dict[str, set[str]] = defaultdict(set)
    for mapping in (dynamic_symbols, external_symbols):
        for module, symbols in mapping.items():
            if module in production_modules:
                symbol_roots[module].update(symbols)
    unreachable_symbols = _unreachable_symbols(
        modules,
        production_modules,
        symbol_roots,
    )
    sibling_imports = _sibling_imports(family, modules)
    noncanonical = _noncanonical_model_paths(family_dir)
    misplaced = sorted(
        path
        for path in noncanonical
        if path.removeprefix("model/").rstrip("/") in MISPLACED_MODEL_PATHS
    )
    fixed_switches = _strategy_switches(modules, production_modules)
    missing_dynamic = [
        {
            "source": entry.source,
            "path": entry.path,
            "symbol": entry.symbol,
            "reason": "missing_path" if not entry.exists else "missing_symbol",
        }
        for entry in dynamic_entries
        if not entry.exists or entry.symbol_exists is False
    ]
    missing_dynamic.extend(missing_conventions)
    missing_dynamic.sort(
        key=lambda item: (
            item["source"],
            item["path"],
            str(item.get("symbol", "")),
        )
    )

    module_rows = []
    for module_name, info in sorted(modules.items()):
        if module_name in production_modules:
            classification = "production"
        elif module_name in tool_modules:
            classification = "tool_test_only"
        else:
            classification = "unreachable"
        module_rows.append(
            {
                "module": module_name,
                "path": info.relative_path,
                "classification": classification,
                "lines": len(info.path.read_text(encoding="utf-8").splitlines()),
                "bytes": info.path.stat().st_size,
                "definitions": list(info.definitions),
                "external_sources": external_sources.get(module_name, []),
            }
        )

    tool_model_modules = sorted(
        info.relative_path
        for module_name, info in modules.items()
        if module_name in tool_modules
    )
    violations: list[dict[str, Any]] = []
    categories = {
        "fixed_strategy_switch": fixed_switches,
        "missing_dynamic_entrypoint": missing_dynamic,
        "noncanonical_model_path": [{"path": path} for path in noncanonical],
        "sibling_family_import": sibling_imports,
        "tool_test_only_model_module": [{"path": path} for path in tool_model_modules],
        "unreachable_module": [
            {"module": module, "path": modules[module].relative_path}
            for module in sorted(unreachable_modules)
        ],
        "unreachable_symbol": unreachable_symbols,
    }
    for kind, items in categories.items():
        violations.extend({"kind": kind, **item} for item in items)

    return {
        "family": family,
        "metrics": _source_metrics(family_dir),
        "entrypoints": {
            "production_modules": sorted(production_seeds),
            "dynamic": [
                {
                    "source": entry.source,
                    "path": entry.path,
                    "symbol": entry.symbol,
                    "exists": entry.exists,
                    "module": entry.module,
                    "symbol_exists": entry.symbol_exists,
                }
                for entry in dynamic_entries
            ],
        },
        "modules": module_rows,
        "production_modules": sorted(production_modules),
        "tool_test_only_modules": sorted(tool_modules),
        "unreachable_modules": sorted(unreachable_modules),
        "unreachable_symbols": unreachable_symbols,
        "fixed_strategy_switches": fixed_switches,
        "missing_dynamic_entrypoints": missing_dynamic,
        "sibling_family_imports": sibling_imports,
        "noncanonical_model_paths": noncanonical,
        "misplaced_model_paths": misplaced,
        "violations": sorted(
            violations,
            key=lambda item: (
                item["kind"],
                str(item.get("path", "")),
                str(item.get("module", "")),
                str(item.get("symbol", "")),
            ),
        ),
    }


def audit_repo(repo_root: Path, selected: tuple[str, ...]) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    scan_paths = _scan_paths(repo_root)
    families = [
        audit_family(repo_root, family_dir, scan_paths=scan_paths)
        for family_dir in family_dirs(repo_root, selected)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "families": families,
        "summary": {
            "families": len(families),
            "files": sum(item["metrics"]["files"] for item in families),
            "lines": sum(item["metrics"]["lines"] for item in families),
            "bytes": sum(item["metrics"]["bytes"] for item in families),
            "model_files": sum(item["metrics"]["model_files"] for item in families),
            "model_lines": sum(item["metrics"]["model_lines"] for item in families),
            "model_bytes": sum(item["metrics"]["model_bytes"] for item in families),
            "violations": sum(len(item["violations"]) for item in families),
            "unreachable_modules": sum(
                len(item["unreachable_modules"]) for item in families
            ),
            "unreachable_symbols": sum(
                len(item["unreachable_symbols"]) for item in families
            ),
            "fixed_strategy_switches": sum(
                len(item["fixed_strategy_switches"]) for item in families
            ),
        },
    }


def _write_report(report: dict[str, Any], destination: Path | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if destination is None:
        sys.stdout.write(text)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inventory", "audit"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.all and not args.family:
        raise SystemExit("Select at least one --family or pass --all")
    if args.all and args.family:
        raise SystemExit("Use --all or --family, not both")
    selected = () if args.all else tuple(args.family)
    report = audit_repo(args.repo_root, selected)
    _write_report(report, args.json)
    if args.command == "audit" and report["summary"]["violations"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
