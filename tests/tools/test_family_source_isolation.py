# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
from pathlib import Path, PurePosixPath

import pytest

from tools import family_source_isolation as isolation
from tools import prune_family_helpers


REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILIES_ROOT = REPO_ROOT / "python/tensorrt_model_connect/families"


def _module_exists(family_dir: Path, parts: tuple[str, ...]) -> bool:
    target = family_dir.joinpath(*parts)
    return target.with_suffix(".py").is_file() or (target / "__init__.py").is_file()


def test_resolve_selection_uses_manifest_runtime_owner() -> None:
    selection = isolation.resolve_selection(REPO_ROOT, "magpie_tts")

    assert selection.family == "magpie_tts"
    assert selection.runtime_models == ("magpie",)
    assert "magpie-tts-357m" in selection.e2e_models


@pytest.mark.parametrize(
    ("path", "included"),
    [
        ("python/tensorrt_model_connect/families/__init__.py", True),
        ("python/tensorrt_model_connect/families/base.py", True),
        ("python/tensorrt_model_connect/families/_time_series_trt.py", False),
        ("python/tensorrt_model_connect/families/qwen/plugin.py", True),
        ("python/tensorrt_model_connect/families/llama/plugin.py", False),
        ("src/runtime/models/qwen/plugin.cpp", True),
        ("src/runtime/models/llama/plugin.cpp", False),
        ("tests/cpp/models/qwen/test_qwen_tensor_names.cpp", True),
        ("tests/cpp/models/llama/test_llama_pipeline.cpp", False),
        ("tests/e2e_harness/bundle_group_runner.py", True),
        ("tests/e2e/models/qwen/MODEL.toml", True),
        ("tests/e2e/models/llama/MODEL.toml", False),
        ("tools/families/qwen/bench_flashinfer_e2e.py", True),
        ("tools/families/llama/example.py", False),
        ("tests/builder/families/qwen/test_family.py", True),
        ("tests/builder/families/flux/test_family.py", False),
        ("python/tensorrt_model_connect/build_cli.py", True),
    ],
)
def test_include_path_enforces_family_boundaries(path: str, included: bool) -> None:
    selection = isolation.FamilySourceSelection(
        family="qwen",
        runtime_models=("qwen",),
        e2e_models=("qwen3-0.6b-fp16",),
    )

    assert isolation.include_path(PurePosixPath(path), selection) is included


def test_materialize_contains_only_selected_owned_directories(tmp_path: Path) -> None:
    selection = isolation.resolve_selection(REPO_ROOT, "qwen")
    output = tmp_path / "qwen-source"

    copied = isolation.materialize(REPO_ROOT, output, selection)

    assert copied > 0
    assert (output / "CMakeLists.txt").is_file()
    assert (output / "python/tensorrt_model_connect/families/base.py").is_file()
    assert (output / "python/tensorrt_model_connect/families/qwen/plugin.py").is_file()
    qwen_family = output / "python/tensorrt_model_connect/families/qwen"
    assert any(
        path.is_file()
        for path in (qwen_family / "model/model.py", qwen_family / "graph_ops.py")
    )
    assert not (output / "python/tensorrt_model_connect/families/llama").exists()
    assert any(
        path.is_file()
        for path in (
            output / "tools/families/qwen/bench_flashinfer_e2e.py",
            qwen_family / "bench_flashinfer_e2e.py",
        )
    )
    assert not (output / "tools/families/flux").exists()
    assert not (
        output / "python/tensorrt_model_connect/families/_time_series_trt.py"
    ).exists()
    assert (output / "src/runtime/models/qwen/MODEL.toml").is_file()
    assert not (output / "src/runtime/models/llama").exists()
    assert (output / "tests/cpp/models/qwen").is_dir()
    assert not (output / "tests/cpp/models/llama").exists()
    assert (output / "tests/e2e/models/qwen/MODEL.toml").is_file()
    assert not (output / "tests/e2e/models/llama").exists()

    metadata = json.loads(
        (output / ".trtmc-family-source.json").read_text(encoding="utf-8")
    )
    assert metadata["family"] == "qwen"
    assert metadata["runtime_models"] == ["qwen"]
    assert metadata["copied_files"] == copied


def test_resolve_selection_rejects_unknown_family() -> None:
    with pytest.raises(SystemExit, match="Unknown Python model family"):
        isolation.resolve_selection(REPO_ROOT, "not_a_family")


def test_family_imports_resolve_without_sibling_or_unapproved_shared_modules() -> None:
    violations: list[str] = []
    families_prefix = "tensorrt_model_connect.families"

    for family_dir in sorted(FAMILIES_ROOT.iterdir()):
        if not (family_dir / "plugin.py").is_file():
            continue
        family = family_dir.name
        for path in sorted(family_dir.rglob("*.py")):
            relative = path.relative_to(family_dir)
            package_parts = relative.parent.parts
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        prefix = f"{families_prefix}."
                        if alias.name.startswith(prefix):
                            owner = alias.name[len(prefix):].split(".", 1)[0]
                            if owner != family:
                                violations.append(
                                    f"{relative}:{node.lineno}: imports sibling {owner}"
                                )
                    continue

                if not isinstance(node, ast.ImportFrom):
                    continue
                if node.level == 0:
                    module = node.module or ""
                    if module == families_prefix:
                        names = {alias.name for alias in node.names}
                        unsupported = names - {"find_plugin"}
                        if unsupported:
                            violations.append(
                                f"{relative}:{node.lineno}: imports unapproved families "
                                f"surface {sorted(unsupported)}"
                            )
                    elif module.startswith(f"{families_prefix}."):
                        owner = module[len(families_prefix) + 1:].split(".", 1)[0]
                        if owner != family:
                            violations.append(
                                f"{relative}:{node.lineno}: imports sibling {owner}"
                            )
                    continue

                parents = node.level - 1
                if parents <= len(package_parts):
                    local_parts = package_parts[:len(package_parts) - parents]
                    if node.module:
                        target = (*local_parts, *node.module.split("."))
                        if not _module_exists(family_dir, target):
                            violations.append(
                                f"{relative}:{node.lineno}: missing local module "
                                f"{'.'.join(target)}"
                            )
                    else:
                        for alias in node.names:
                            target = (*local_parts, alias.name)
                            if not _module_exists(family_dir, target):
                                violations.append(
                                    f"{relative}:{node.lineno}: missing local module "
                                    f"{'.'.join(target)}"
                                )
                    continue

                # Escaping exactly one package reaches families/.  Only its
                # registry function and protocol module are approved shared
                # infrastructure.  Escaping farther reaches generic package
                # infrastructure such as parallel_config and quantization.
                if parents == len(package_parts) + 1:
                    if node.module == "base":
                        continue
                    if node.module is None and {
                        alias.name for alias in node.names
                    } <= {"find_plugin"}:
                        continue
                    violations.append(
                        f"{relative}:{node.lineno}: imports unapproved families-root "
                        f"module {node.module or [a.name for a in node.names]}"
                    )

    assert not violations, "\n".join(violations)


def test_helper_pruner_keeps_transitive_and_quantization_dependencies(
    tmp_path: Path,
) -> None:
    family_dir = tmp_path / "demo"
    model_dir = family_dir / "model"
    model_dir.mkdir(parents=True)
    (family_dir / "MODEL.toml").write_text('id = "demo"\n', encoding="utf-8")
    graph_ops = model_dir / "model.py"
    graph_ops.write_text(
        "def add_constant():\n"
        "    return 1\n\n"
        "def add_matmul_rhs_constant():\n"
        "    return 2\n\n"
        "def _helper():\n"
        "    return 3\n\n"
        "def used():\n"
        "    return _helper()\n\n"
        "def unused():\n"
        "    return 4\n",
        encoding="utf-8",
    )
    (model_dir / "builder.py").write_text(
        "from . import model as graph_ops\n\n"
        "def build():\n"
        "    return graph_ops.used()\n",
        encoding="utf-8",
    )

    result = prune_family_helpers.prune_file(
        graph_ops, family_dir, write=True
    )

    assert result.removed_names == ("unused",)
    updated = graph_ops.read_text(encoding="utf-8")
    assert "def add_constant" in updated
    assert "def add_matmul_rhs_constant" in updated
    assert "def _helper" in updated
    assert "def used" in updated
    assert "def unused" not in updated


def test_helper_pruner_removes_exact_audited_names(tmp_path: Path) -> None:
    module = tmp_path / "model.py"
    module.write_text(
        "def keep():\n"
        "    return 1\n\n"
        "def remove():\n"
        "    return 2\n",
        encoding="utf-8",
    )

    result = prune_family_helpers.prune_named_definitions(
        module, {"remove"}, write=True
    )

    assert result.removed_names == ("remove",)
    assert "def keep" in module.read_text(encoding="utf-8")
    assert "def remove" not in module.read_text(encoding="utf-8")
