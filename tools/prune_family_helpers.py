#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Remove unreachable definitions from family-owned implementation modules."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

try:
    from tools import family_specialization
except ModuleNotFoundError:  # Direct execution puts tools/ on sys.path.
    import family_specialization


TARGET_MODULES = (
    "checkpoint_mapper.py",
    "model.py",
    "default_decoder.py",
    "default_dual_profile_decoder.py",
    "default_dual_profile_decoder_tp.py",
    "standard_decoder_builder.py",
    "dual_profile_decoder_builder.py",
    "dual_profile_decoder_tp_builder.py",
    "vl_debug_runner.py",
)
EXTERNAL_MODULE_ROOTS = {
    "model.py": frozenset({"add_constant", "add_matmul_rhs_constant"}),
    "vl_debug_runner.py": frozenset(
        {
            "load_vision_engine_from_bundle",
            "load_config_from_bundle",
            "load_preprocessor_config_from_bundle",
            "VisionTrtRunner",
            "preprocess_image_inputs_for_trt",
            "load_section_from_bundle",
            "load_engine_from_bundle",
            "TrtRunner",
            "VLTrtRunner",
        }
    ),
}
Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


@dataclass(frozen=True)
class PruneResult:
    path: Path
    removed_names: tuple[str, ...]
    removed_lines: int


def _module_name(family: str, family_dir: Path, path: Path) -> str:
    parts = list(path.relative_to(family_dir).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    suffix = ".".join(parts)
    base = f"tensorrt_model_connect.families.{family}"
    return f"{base}.{suffix}" if suffix else base


def _resolved_from_module(
    current_module: str,
    current_path: Path,
    node: ast.ImportFrom,
) -> str | None:
    if node.level == 0:
        return node.module
    package = (
        current_module
        if current_path.name == "__init__.py"
        else current_module.rpartition(".")[0]
    )
    parts = package.split(".") if package else []
    parents = node.level - 1
    if parents > len(parts):
        return None
    base = parts[:len(parts) - parents]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _definition_map(tree: ast.Module) -> dict[str, list[Definition]]:
    definitions: dict[str, list[Definition]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.setdefault(node.name, []).append(node)
    return definitions


def _loaded_definition_names(node: ast.AST, names: set[str]) -> set[str]:
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


def _external_roots(
    target_path: Path,
    family_dir: Path,
    definitions: set[str],
) -> set[str]:
    family = family_dir.name
    target_module = _module_name(family, family_dir, target_path)
    roots: set[str] = set()

    for path in family_dir.rglob("*.py"):
        if path == target_path:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        current_module = _module_name(family, family_dir, path)
        aliases: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == target_module:
                        aliases.add(alias.asname or alias.name.rsplit(".", 1)[-1])
                continue
            if not isinstance(node, ast.ImportFrom):
                continue
            resolved = _resolved_from_module(current_module, path, node)
            if resolved == target_module:
                for alias in node.names:
                    if alias.name == "*":
                        roots.update(definitions)
                    elif alias.name in definitions:
                        roots.add(alias.name)
                continue
            for alias in node.names:
                candidate = f"{resolved}.{alias.name}" if resolved else alias.name
                if candidate == target_module:
                    aliases.add(alias.asname or alias.name)

        roots.update(
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in aliases
            and node.attr in definitions
        )

    return roots


def reachable_definitions(path: Path, family_dir: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    definitions = _definition_map(tree)
    names = set(definitions)
    dependencies = {
        name: set().union(
            *(_loaded_definition_names(node, names) for node in nodes)
        )
        for name, nodes in definitions.items()
    }
    roots = _external_roots(path, family_dir, names)

    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            roots.update(_loaded_definition_names(node, names))

    # Generic tools receive selected family modules dynamically, so their
    # calls cannot be attributed to a concrete family by the import scanner.
    roots.update(names & EXTERNAL_MODULE_ROOTS.get(path.name, frozenset()))

    reachable = set(roots)
    pending = deque(roots)
    while pending:
        for dependency in dependencies[pending.popleft()]:
            if dependency not in reachable:
                reachable.add(dependency)
                pending.append(dependency)
    return reachable


def _removal_spans(
    lines: list[str],
    definitions: dict[str, list[Definition]],
    removed_names: set[str],
) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for name in removed_names:
        for node in definitions[name]:
            start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
            end = node.end_lineno or node.lineno
            # Include an immediately preceding comment/blank section, stopping
            # at code owned by the previous definition or module statement.
            while start > 1:
                previous = lines[start - 2]
                if previous.strip() and not previous.lstrip().startswith("#"):
                    break
                start -= 1
            spans.append((start - 1, end))

    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def prune_file(path: Path, family_dir: Path, *, write: bool) -> PruneResult:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    definitions = _definition_map(tree)
    reachable = reachable_definitions(path, family_dir)
    removed = set(definitions) - reachable
    return prune_named_definitions(path, removed, write=write)


def prune_named_definitions(
    path: Path,
    names: set[str],
    *,
    write: bool,
) -> PruneResult:
    """Remove named top-level definitions emitted by the specialization audit."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    definitions = _definition_map(tree)
    removed = set(names)
    if not removed:
        return PruneResult(path, (), 0)
    missing = removed - definitions.keys()
    if missing:
        raise ValueError(
            f"Audit symbols no longer exist in {path}: {', '.join(sorted(missing))}"
        )

    lines = text.splitlines(keepends=True)
    spans = _removal_spans(lines, definitions, removed)
    removed_lines = sum(end - start for start, end in spans)
    if write:
        for start, end in reversed(spans):
            del lines[start:end]
        updated = "".join(lines)
        updated = re.sub(r"\n{4,}", "\n\n\n", updated)
        path.write_text(updated, encoding="utf-8")
    return PruneResult(path, tuple(sorted(removed)), removed_lines)


def family_dirs(repo_root: Path, selected: tuple[str, ...]) -> list[Path]:
    root = repo_root / "python/tensorrt_model_connect/families"
    if selected:
        paths = [root / family for family in selected]
        missing = [
            path.name for path in paths if not (path / "model.py").is_file()
        ]
        if missing:
            raise SystemExit("Unknown model family: " + ", ".join(missing))
        return paths
    else:
        return sorted(
            path for path in root.iterdir() if (path / "model.py").is_file()
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--family", action="append", default=[])
    parser.add_argument(
        "--write",
        action="store_true",
        help="Rewrite helper files; without this flag, report and fail on dead code",
    )
    parser.add_argument(
        "--strict-audit",
        action="store_true",
        help="Prune exactly the unreachable symbols reported by family_specialization.py",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    results: list[PruneResult] = []
    if args.strict_audit:
        report = family_specialization.audit_repo(repo_root, tuple(args.family))
        for family in report["families"]:
            family_dir = (
                repo_root
                / "python/tensorrt_model_connect/families"
                / family["family"]
            )
            by_path: dict[str, set[str]] = {}
            for item in family["unreachable_symbols"]:
                by_path.setdefault(item["path"], set()).add(item["symbol"])
            for relative_path, names in sorted(by_path.items()):
                result = prune_named_definitions(
                    family_dir / relative_path,
                    names,
                    write=args.write,
                )
                if result.removed_names:
                    results.append(result)
    else:
        for family_dir in family_dirs(repo_root, tuple(args.family)):
            for filename in TARGET_MODULES:
                for path in sorted(family_dir.rglob(filename)):
                    if "__pycache__" in path.parts or not path.is_file():
                        continue
                    result = prune_file(path, family_dir, write=args.write)
                    if result.removed_names:
                        results.append(result)
    for result in results:
        print(
            f"{result.path.relative_to(repo_root)}: "
            f"{len(result.removed_names)} definitions, "
            f"{result.removed_lines} lines "
            f"[{', '.join(result.removed_names)}]"
        )
    if results:
        print(
            f"total: {len(results)} files, "
            f"{sum(len(result.removed_names) for result in results)} definitions, "
            f"{sum(result.removed_lines for result in results)} lines"
        )
    return 0 if args.write or not results else 1


if __name__ == "__main__":
    sys.exit(main())
