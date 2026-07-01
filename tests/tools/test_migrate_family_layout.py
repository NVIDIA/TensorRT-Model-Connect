# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from pathlib import Path

from tools import migrate_family_layout as layout


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_migrate_rehomes_components_and_rewrites_imports(tmp_path: Path) -> None:
    family = tmp_path / "python/tensorrt_model_connect/families/demo"
    _write(family / "__init__.py", "from .plugin import plugin\n")
    _write(family / "config.py", "class ModelConfig:\n    pass\n")
    _write(
        family / "checkpoint_mapper.py",
        "from .config import ModelConfig\n"
        "from ...quantization.context import QuantContext\n\n"
        "def load_weights():\n"
        "    return ModelConfig, QuantContext\n",
    )
    _write(family / "graph_ops.py", "def add_constant():\n    return 1\n")
    _write(
        family / "builder.py",
        "from . import graph_ops\n"
        "from .config import ModelConfig\n"
        "from ...parallel_config import ParallelConfig\n",
    )
    _write(
        family / "plugin.py",
        "from . import graph_ops\n"
        "from .builder import build\n"
        "from .checkpoint_mapper import load_weights\n\n"
        "plugin = object()\n",
    )
    _write(
        family / "diff_vl.py",
        "from . import graph_ops\n\n"
        "print(graph_ops.add_constant())\n",
    )
    _write(
        tmp_path / "tests/test_demo.py",
        "from tensorrt_model_connect.families.demo.graph_ops import add_constant\n",
    )

    layout.migrate(tmp_path)

    assert not layout.layout_violations(tmp_path)
    assert {path.name for path in family.iterdir()} == layout.CANONICAL_TOP_LEVEL
    assert "from .model import model as graph_ops" in (family / "plugin.py").read_text()
    assert "from .model.builder import build" in (family / "plugin.py").read_text()
    assert "from .weights import load_weights" in (family / "plugin.py").read_text()

    builder = (family / "model/builder.py").read_text()
    assert "from . import model as graph_ops" in builder
    assert "from ..config import ModelConfig" in builder
    assert "from ....parallel_config import ParallelConfig" in builder

    weights = (family / "weights/__init__.py").read_text()
    assert "from ..config import ModelConfig" in weights
    assert "from ....quantization.context import QuantContext" in weights

    external = (tmp_path / "tests/test_demo.py").read_text()
    assert "families.demo.model.model" in external
    moved_tool = tmp_path / "tools/families/demo/diff_vl.py"
    assert moved_tool.is_file()
    assert "from tensorrt_model_connect.families.demo.model import model as graph_ops" in (
        moved_tool.read_text()
    )
    consolidated = (family / "model/model.py").read_text()
    assert "def add_constant" in consolidated
    assert not (family / "model/graph_ops.py").exists()


def test_migrate_adds_missing_manifest(tmp_path: Path) -> None:
    family = tmp_path / "python/tensorrt_model_connect/families/demo"
    _write(family / "__init__.py", "")
    _write(family / "config.py", "")
    _write(family / "checkpoint_mapper.py", "")
    _write(family / "graph_ops.py", "")
    _write(family / "plugin.py", "plugin = object()\n")

    layout.migrate(tmp_path)

    manifest = (family / "MODEL.toml").read_text()
    assert 'id = "demo"' in manifest
    assert 'module = "plugin"' in manifest


def test_migrate_can_target_one_family(tmp_path: Path) -> None:
    selected = tmp_path / "python/tensorrt_model_connect/families/selected"
    untouched = tmp_path / "python/tensorrt_model_connect/families/untouched"
    for family in (selected, untouched):
        _write(family / "__init__.py", "from .plugin import plugin\n")
        _write(family / "config.py", "")
        _write(family / "checkpoint_mapper.py", "")
        _write(family / "graph_ops.py", "")
        _write(family / "plugin.py", "plugin = object()\n")

    layout.migrate(tmp_path, ("selected",))

    assert not layout.layout_violations(tmp_path, ("selected",))
    assert (selected / "model/model.py").is_file()
    assert (untouched / "graph_ops.py").is_file()
    assert not (untouched / "model").exists()


def test_migrated_family_imports_use_canonical_components() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    families_root = repo_root / "python/tensorrt_model_connect/families"
    prefix = "tensorrt_model_connect.families."
    allowed_components = {"config", "model", "plugin", "weights"}
    violations: list[str] = []

    for source_root in (".github", "python", "scripts", "tests", "tools"):
        root = repo_root / source_root
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            modules: list[tuple[int, str]] = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.extend((node.lineno, alias.name) for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules.append((node.lineno, node.module))

            for line, module in modules:
                if not module.startswith(prefix):
                    continue
                parts = module[len(prefix):].split(".")
                family_dir = families_root / parts[0]
                if not (family_dir / "plugin.py").is_file():
                    continue
                if not (family_dir / "model/model.py").is_file():
                    continue
                component_parts = parts[1:]
                if component_parts and component_parts[0] not in allowed_components:
                    violations.append(f"{path}:{line}: non-canonical family import {module}")
                    continue
                if not component_parts:
                    continue
                target = family_dir.joinpath(*component_parts)
                if not (target.with_suffix(".py").is_file() or (target / "__init__.py").is_file()):
                    violations.append(f"{path}:{line}: missing family module {module}")

    assert not violations, "\n".join(violations)
