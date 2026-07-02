#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that every family plugin has at least one E2E model manifest.

Runs as a CI gate right after build. Uses AST parsing to discover plugins
without importing them (no TRT/torch required).

Exit codes:
  0  — all plugins covered (or only exempt plugins uncovered)
  1  — at least one non-exempt plugin has no E2E manifest
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

# Plugins that are known-WIP and not yet wired into the E2E harness.
# Keep this list as small as possible: every entry is a coverage gap.
_EXEMPT_PLUGINS: set[str] = set()

REPO_ROOT = Path(__file__).resolve().parent.parent
FAMILIES_DIR = REPO_ROOT / "python" / "tensorrt_model_connect" / "families"
MODELS_DIR = REPO_ROOT / "tests" / "e2e" / "models"


def iter_manifest_paths() -> list[Path]:
    """Return E2E manifests from flat and model-owned layouts."""
    if not MODELS_DIR.is_dir():
        return []
    return sorted({
        *MODELS_DIR.glob("*.json"),
        *MODELS_DIR.glob("*/manifests/*.json"),
    })


def discover_plugin_names() -> set[str]:
    """Scan family plugin .py files with AST to extract plugin name attrs."""
    names: set[str] = set()
    plugin_files = list(FAMILIES_DIR.glob("*.py"))
    plugin_files.extend(FAMILIES_DIR.glob("*/plugin.py"))
    for py_file in sorted(plugin_files):
        if py_file.name.startswith("_") or py_file.stem in {"__init__", "base"}:
            continue
        if any(part.startswith("_") for part in py_file.relative_to(FAMILIES_DIR).parts[:-1]):
            continue
        try:
            tree = ast.parse(py_file.read_text(), filename=str(py_file))
        except SyntaxError:
            continue

        # Find the class name in ``plugin = ClassName(...)``
        plugin_class_name: str | None = None
        for node in ast.iter_child_nodes(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "plugin"
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
            ):
                plugin_class_name = node.value.func.id
                break
        if plugin_class_name is None:
            continue

        # Extract the ``name`` string attribute from that class.
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef) and node.name == plugin_class_name:
                for item in node.body:
                    if (
                        isinstance(item, ast.Assign)
                        and len(item.targets) == 1
                        and isinstance(item.targets[0], ast.Name)
                        and item.targets[0].id == "name"
                        and isinstance(item.value, ast.Constant)
                        and isinstance(item.value.value, str)
                    ):
                        names.add(item.value.value)
                break
    return names


def discover_manifest_families() -> dict[str, list[str]]:
    """Load all E2E manifests and return {family: [model_names]}."""
    families: dict[str, list[str]] = {}
    for manifest_path in iter_manifest_paths():
        with open(manifest_path) as f:
            data = json.load(f)
        family = data.get("family")
        name = data.get("name", manifest_path.stem)
        if family:
            families.setdefault(family, []).append(name)
    return families


def main() -> int:
    plugin_names = discover_plugin_names()
    manifest_families = discover_manifest_families()
    covered = plugin_names & set(manifest_families)
    exempt = plugin_names & _EXEMPT_PLUGINS
    uncovered = plugin_names - set(manifest_families) - _EXEMPT_PLUGINS

    print("=== Family Plugin E2E Coverage Report ===")
    print(f"Total plugins discovered: {len(plugin_names)}")
    print(f"Covered by E2E manifest: {len(covered)}")
    if exempt:
        print(f"Exempt (WIP):            {len(exempt)} ({', '.join(sorted(exempt))})")
    print(f"Uncovered:               {len(uncovered)}")
    print()

    # Detailed listing
    for name in sorted(plugin_names):
        models = manifest_families.get(name, [])
        if models:
            print(f"  [OK]      {name} ({len(models)} model(s): {', '.join(models)})")
        elif name in _EXEMPT_PLUGINS:
            print(f"  [EXEMPT]  {name}")
        else:
            print(f"  [MISSING] {name}")

    print()

    if uncovered:
        print(
            f"ERROR: {len(uncovered)} plugin(s) have no E2E manifest: "
            f"{', '.join(sorted(uncovered))}"
        )
        print(
            "Add a JSON manifest in tests/e2e/models/<family>/manifests/ "
            "with 'family' matching the plugin name."
        )
        return 1

    if exempt:
        print(
            f"WARNING: {len(exempt)} plugin(s) are exempt from E2E coverage: "
            f"{', '.join(sorted(exempt))}. "
            f"Wire their runtime_strategy into the E2E harness to remove the exemption."
        )

    print("All non-exempt plugins have E2E manifest coverage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
