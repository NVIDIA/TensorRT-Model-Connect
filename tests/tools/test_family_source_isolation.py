# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
from pathlib import Path, PurePosixPath

import pytest

from tools import family_source_isolation as isolation


REPO_ROOT = Path(__file__).resolve().parents[2]
FAMILIES_ROOT = REPO_ROOT / "python/tensorrt_model_connect/models"


def _module_exists(family_dir: Path, parts: tuple[str, ...]) -> bool:
    target = family_dir.joinpath(*parts)
    return target.with_suffix(".py").is_file() or (target / "__init__.py").is_file()


def test_resolve_selection_uses_manifest_runtime_owner() -> None:
    selection = isolation.resolve_selection(REPO_ROOT, "wan_t2v")

    assert selection.family == "wan_t2v"
    assert selection.runtime_models == ("wan_t2v",)
    assert "wan21-t2v-1.3b" in selection.e2e_models


@pytest.mark.parametrize(
    ("path", "included"),
    [
        ("python/tensorrt_model_connect/models/__init__.py", True),
        ("python/tensorrt_model_connect/models/base.py", False),
        ("python/tensorrt_model_connect/models/qwen/model.py", True),
        ("python/tensorrt_model_connect/models/llama/model.py", False),
        ("python/tensorrt_model_connect/models/qwen/runtime/plugin.cpp", True),
        ("python/tensorrt_model_connect/models/llama/runtime/plugin.cpp", False),
        ("python/tensorrt_model_connect/models/qwen/tests/cpp/test_qwen_tensor_names.cpp", True),
        ("python/tensorrt_model_connect/models/llama/tests/cpp/test_llama_pipeline.cpp", False),
        ("tests/e2e_harness/model_runner.py", True),
        ("python/tensorrt_model_connect/models/qwen/MODEL.toml", True),
        ("python/tensorrt_model_connect/models/llama/MODEL.toml", False),
        ("python/tensorrt_model_connect/models/qwen/tools/bench_flashinfer_e2e.py", True),
        ("python/tensorrt_model_connect/models/llama/tools/example.py", False),
        ("python/tensorrt_model_connect/models/qwen/tests/test_family.py", True),
        ("python/tensorrt_model_connect/models/flux/tests/test_family.py", False),
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
    assert not (output / "python/tensorrt_model_connect/models/base.py").exists()
    assert (output / "python/tensorrt_model_connect/models/qwen/model.py").is_file()
    qwen_family = output / "python/tensorrt_model_connect/models/qwen"
    assert (qwen_family / "model.py").is_file()
    assert not (output / "python/tensorrt_model_connect/models/llama").exists()
    assert (qwen_family / "tools/bench_flashinfer_e2e.py").is_file()
    assert (qwen_family / "runtime/plugin.cpp").is_file()
    assert any((qwen_family / "tests/cpp").glob("*.cpp"))
    assert (qwen_family / "MODEL.toml").is_file()
    assert not (output / "python/tensorrt_model_connect/models/llama").exists()

    metadata = json.loads((output / ".trtmc-family-source.json").read_text(encoding="utf-8"))
    assert metadata["family"] == "qwen"
    assert metadata["runtime_models"] == ["qwen"]
    assert metadata["copied_files"] == copied


def test_resolve_selection_rejects_unknown_family() -> None:
    with pytest.raises(SystemExit, match="Unknown Python model family"):
        isolation.resolve_selection(REPO_ROOT, "not_a_family")


def test_family_imports_resolve_without_sibling_or_unapproved_shared_modules() -> None:
    violations: list[str] = []
    families_prefix = "tensorrt_model_connect.models"

    for family_dir in sorted(FAMILIES_ROOT.iterdir()):
        if not (family_dir / "model.py").is_file():
            continue
        family = family_dir.name
        for path in sorted(family_dir.rglob("*.py")):
            relative = path.relative_to(family_dir)
            if relative.parts and relative.parts[0] == "tests":
                continue
            package_parts = relative.parent.parts
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        prefix = f"{families_prefix}."
                        if alias.name.startswith(prefix):
                            owner = alias.name[len(prefix) :].split(".", 1)[0]
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
                        if names:
                            violations.append(
                                f"{relative}:{node.lineno}: imports unapproved families "
                                f"surface {sorted(names)}"
                            )
                    elif module.startswith(f"{families_prefix}."):
                        owner = module[len(families_prefix) + 1 :].split(".", 1)[0]
                        if owner != family:
                            violations.append(f"{relative}:{node.lineno}: imports sibling {owner}")
                    continue

                parents = node.level - 1
                if parents <= len(package_parts):
                    local_parts = package_parts[: len(package_parts) - parents]
                    if node.module:
                        target = (*local_parts, *node.module.split("."))
                        if not _module_exists(family_dir, target):
                            violations.append(
                                f"{relative}:{node.lineno}: missing local module {'.'.join(target)}"
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

                # Escaping exactly one package reaches models/. Model owners
                # may use generic package leaves, but never a shared model
                # protocol or sibling-dispatch surface.
                if parents == len(package_parts) + 1:
                    violations.append(
                        f"{relative}:{node.lineno}: imports unapproved models-root "
                        f"module {node.module or [a.name for a in node.names]}"
                    )

    assert not violations, "\n".join(violations)
