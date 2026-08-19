# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static closure checks for owner-local graph operation modules."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
MODELS_ROOT = PYTHON_ROOT / "tensorrt_model_connect" / "models"
Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


def _owner_dirs() -> list[Path]:
    return sorted(
        path
        for path in MODELS_ROOT.iterdir()
        if path.is_dir()
        and (path / "MODEL.toml").is_file()
        and (path / "model.py").is_file()
    )


def _module_name(path: Path) -> tuple[str, bool]:
    relative = path.relative_to(PYTHON_ROOT).with_suffix("")
    parts = list(relative.parts)
    is_init = parts[-1] == "__init__"
    if is_init:
        parts.pop()
    return ".".join(parts), is_init


def _resolved_from_module(
    current_module: str,
    current_is_init: bool,
    node: ast.ImportFrom,
) -> str:
    if node.level == 0:
        return node.module or ""
    package = current_module if current_is_init else current_module.rpartition(".")[0]
    parts = package.split(".") if package else []
    parents = node.level - 1
    if parents > len(parts):
        return ""
    base = parts[: len(parts) - parents]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _assigned_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_assigned_names(item) for item in target.elts))
    return set()


def _concrete_bindings(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bindings = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bindings.update(_assigned_names(target))
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            bindings.update(_assigned_names(node.target))
    return bindings


def _graph_aliases_and_imports(
    path: Path,
    graph_modules: dict[str, Path],
) -> tuple[dict[str, str], list[tuple[str, str, int]]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    current_module, current_is_init = _module_name(path)
    aliases: dict[str, str] = {}
    direct_imports: list[tuple[str, str, int]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for imported in node.names:
                if imported.name in graph_modules:
                    aliases[imported.asname or imported.name.rsplit(".", 1)[-1]] = (
                        imported.name
                    )
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        base = _resolved_from_module(current_module, current_is_init, node)
        if base in graph_modules:
            direct_imports.extend((base, imported.name, node.lineno) for imported in node.names)
            continue
        for imported in node.names:
            candidate = f"{base}.{imported.name}" if base else imported.name
            if candidate in graph_modules:
                aliases[imported.asname or imported.name] = candidate

    # Cover lazy loaders that import ``graph_ops`` under a temporary alias and
    # then assign it to a module-global name used by builder functions.
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Name):
                continue
            target_module = aliases.get(node.value.id)
            if target_module is None:
                continue
            for target in node.targets:
                for name in _assigned_names(target):
                    if aliases.get(name) != target_module:
                        aliases[name] = target_module
                        changed = True
    return aliases, direct_imports


def test_owner_graph_ops_references_have_concrete_bindings() -> None:
    violations: list[str] = []
    owners = _owner_dirs()
    assert owners

    for owner in owners:
        graph_modules = {
            _module_name(path)[0]: path
            for path in owner.rglob("graph_ops.py")
            if "tests" not in path.relative_to(owner).parts
        }
        if not graph_modules:
            continue
        bindings = {
            module: _concrete_bindings(path)
            for module, path in graph_modules.items()
        }
        sources = sorted(
            path
            for path in owner.rglob("*.py")
            if "tests" not in path.relative_to(owner).parts
        )
        for source in sources:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
            aliases, direct_imports = _graph_aliases_and_imports(source, graph_modules)
            for module, symbol, line in direct_imports:
                if symbol == "*" or symbol not in bindings[module]:
                    violations.append(
                        f"{owner.name}: {source.relative_to(REPO_ROOT)}:{line}: "
                        f"imports missing {symbol!r} from "
                        f"{graph_modules[module].relative_to(REPO_ROOT)}"
                    )
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in aliases
                ):
                    continue
                module = aliases[node.value.id]
                if node.attr not in bindings[module]:
                    violations.append(
                        f"{owner.name}: {source.relative_to(REPO_ROOT)}:{node.lineno}: "
                        f"references missing {node.attr!r} in "
                        f"{graph_modules[module].relative_to(REPO_ROOT)}"
                    )

    assert not violations, "\n".join(sorted(set(violations)))
