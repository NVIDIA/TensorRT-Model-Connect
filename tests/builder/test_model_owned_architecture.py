# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Architecture contract for model-owned bundle construction."""

from __future__ import annotations

import ast
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
FAMILIES_ROOT = REPO_ROOT / "python/tensorrt_model_connect/models"
E2E_MODELS_ROOT = REPO_ROOT / "python/tensorrt_model_connect/models"
ENGINE_BUILDER = REPO_ROOT / "python/tensorrt_model_connect/engine_builder.py"
RETIRED_FULL_ROPE_TABLE_APIS = {
    "make_rope_table",
    "make_yarn_rope_table",
}


def _family_dirs() -> list[Path]:
    return sorted(
        path
        for path in FAMILIES_ROOT.iterdir()
        if path.is_dir() and (path / "MODEL.toml").is_file()
    )


def _top_level_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _module_bindings(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bindings: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bindings.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            bindings.update(target.id for target in targets if isinstance(target, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bindings.update(alias.asname or alias.name for alias in node.names)
    return bindings


def _imported_module_path(importer: Path, node: ast.ImportFrom) -> Path:
    module_parts = tuple((node.module or "").split(".")) if node.module else ()
    if node.level:
        importer_parts = importer.relative_to(PYTHON_ROOT).with_suffix("").parts
        package_parts = importer_parts[:-1]
        parent_count = node.level - 1
        if parent_count > len(package_parts):
            return Path()
        module_parts = package_parts[: len(package_parts) - parent_count] + module_parts
    module = PYTHON_ROOT.joinpath(*module_parts)
    source = module.with_suffix(".py")
    return source if source.is_file() else module / "__init__.py"


def _class_methods(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    classes = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    if len(classes) != 1:
        return set()
    return {
        node.name
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_every_family_has_one_required_model_entrypoint() -> None:
    families = _family_dirs()
    assert families
    assert list(FAMILIES_ROOT.rglob("plugin.py")) == []
    for family in families:
        model = family / "model.py"
        assert model.is_file(), f"{family.name} must own model.py"
        functions = _top_level_functions(model)
        assert {"matches", "build"} <= functions, (
            f"{family.name}/model.py must define matches() and build()"
        )


def test_model_config_from_dir_bindings_are_concrete() -> None:
    violations: list[str] = []
    for model in sorted(FAMILIES_ROOT.glob("*/model.py")):
        tree = ast.parse(model.read_text(encoding="utf-8"), filename=str(model))
        calls_from_dir = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "from_dir"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "ModelConfig"
            for node in ast.walk(tree)
        )
        if not calls_from_dir:
            continue

        bindings: list[tuple[Path, str]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "ModelConfig":
                bindings.append((model, node.name))
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if (alias.asname or alias.name) == "ModelConfig":
                        bindings.append((_imported_module_path(model, node), alias.name))

        if len(bindings) != 1:
            violations.append(
                f"{model.relative_to(REPO_ROOT)}: expected one ModelConfig binding, "
                f"found {len(bindings)}"
            )
            continue
        source, class_name = bindings[0]
        if not source.is_file() or "from_dir" not in _class_methods(source, class_name):
            violations.append(
                f"{model.relative_to(REPO_ROOT)}: ModelConfig binding "
                f"{source.relative_to(REPO_ROOT) if source.is_file() else source} "
                "does not define from_dir"
            )

    assert not violations, "\n".join(violations)


def test_model_owners_have_no_builder_forwarding_shims() -> None:
    forwarding_imports = {
        "standard_decoder_builder.py": "from .default_decoder import",
        "dual_profile_decoder_tp_builder.py": ("from .default_dual_profile_decoder_tp import"),
    }
    violations = []
    for filename, marker in forwarding_imports.items():
        for path in FAMILIES_ROOT.glob(f"*/{filename}"):
            if marker in path.read_text(encoding="utf-8"):
                violations.append(path.relative_to(REPO_ROOT).as_posix())

    assert not violations, f"obsolete model-owned builder forwarding shims: {violations}"


def test_model_owned_functions_have_no_duplicate_decorators() -> None:
    violations: list[str] = []
    for module in sorted(FAMILIES_ROOT.rglob("*.py")):
        if "tests" in module.relative_to(FAMILIES_ROOT).parts:
            continue
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            decorators = [
                ast.dump(decorator, include_attributes=False) for decorator in node.decorator_list
            ]
            duplicates = sorted(
                decorator for decorator in set(decorators) if decorators.count(decorator) > 1
            )
            if duplicates:
                relative = module.relative_to(REPO_ROOT).as_posix()
                violations.append(f"{relative}:{node.lineno}:{node.name}")

    assert not violations, f"duplicate model-owned decorators: {violations}"


def test_dense_kv_caches_have_no_deprecated_mask_forwarder() -> None:
    marker = "Kept for backward compatibility with tests that call this directly"
    violations = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in FAMILIES_ROOT.glob("*/runtime/kv_cache.h")
        if marker in path.read_text(encoding="utf-8")
    ]

    assert not violations, f"deprecated dense KV mask forwarders: {violations}"


def test_family_modules_do_not_restore_retired_full_rope_table_apis() -> None:
    offenders: dict[Path, set[str]] = {}
    for module in sorted(FAMILIES_ROOT.rglob("*.py")):
        retired = _top_level_functions(module) & RETIRED_FULL_ROPE_TABLE_APIS
        if retired:
            offenders[module.relative_to(REPO_ROOT)] = retired

    assert not offenders, (
        "family modules must use native half-dimension RoPE tables instead of "
        f"restoring retired full-table helpers: {offenders}"
    )


def test_model_entries_are_direct_module_functions() -> None:
    for path in sorted(FAMILIES_ROOT.rglob("model.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert not any(
            isinstance(node, ast.ClassDef) and node.name.endswith("Model") for node in tree.body
        ), path
        assert not any(
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(
                isinstance(target, ast.Name) and target.id == "_model"
                for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            )
            for node in tree.body
        ), path

    for family in _family_dirs():
        path = family / "model.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        functions = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        matches = functions["matches"].args
        assert [arg.arg for arg in (*matches.posonlyargs, *matches.args)] == ["config"], path
        assert matches.vararg is None and matches.kwarg is None, path

        build = functions["build"].args
        assert [arg.arg for arg in (*build.posonlyargs, *build.args)] == [
            "model_dir",
            "output_path",
        ], path
        assert build.vararg is None and build.kwarg is not None, path
        assert build.kwarg.arg == "options", path


def test_model_consumers_do_not_restore_object_forwarding_or_missing_imports() -> None:
    module_bindings = {
        f"tensorrt_model_connect.models.{family.name}.model": _module_bindings(family / "model.py")
        for family in _family_dirs()
    }
    violations: list[str] = []
    consumer_roots = (
        REPO_ROOT / "tests",
        REPO_ROOT / "tools",
        REPO_ROOT / "scripts",
        FAMILIES_ROOT,
    )
    seen: set[Path] = set()
    for root in consumer_roots:
        for path in sorted(root.rglob("*.py")):
            if path in seen or "__pycache__" in path.parts:
                continue
            seen.add(path)
            if path.resolve() == Path(__file__).resolve():
                continue
            source = path.read_text(encoding="utf-8")
            if "PLUGIN_CLASS" in source:
                violations.append(f"{path}: retired PLUGIN_CLASS forwarding")
            if "tensorrt_model_connect.models." not in source or ".model" not in source:
                continue
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                available = module_bindings.get(node.module or "")
                if available is None:
                    continue
                for alias in node.names:
                    if alias.name != "*" and alias.name not in available:
                        violations.append(
                            f"{path}:{node.lineno}: {node.module} has no {alias.name!r} binding"
                        )

    assert not violations, "\n".join(violations)


def test_family_models_do_not_call_back_into_engine_builder() -> None:
    forbidden = "tensorrt_model_connect.engine_builder"
    for family in _family_dirs():
        model = family / "model.py"
        if not model.is_file():
            continue
        tree = ast.parse(model.read_text(encoding="utf-8"), filename=str(model))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(alias.name != forbidden for alias in node.names), family.name
            elif isinstance(node, ast.ImportFrom):
                assert node.module != forbidden, family.name


def test_engine_builder_is_only_a_resolver_and_dispatcher() -> None:
    source = ENGINE_BUILDER.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(ENGINE_BUILDER))
    functions = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_build_diffusion_bundle" not in functions
    assert "_build_native_impl" not in functions
    assert "_try_build_optimized_runtime" not in functions
    assert "build_bundle" not in functions
    assert "find_plugin" not in source
    assert "plugin.build" not in source


def test_manifests_have_no_retired_builder_routing_fields() -> None:
    manifests = [family / "MODEL.toml" for family in _family_dirs()]
    manifests.extend(sorted(E2E_MODELS_ROOT.glob("*/MODEL.toml")))
    for path in manifests:
        manifest = path.read_text(encoding="utf-8")
        for field in ("module", "plugin", "default_build_route"):
            assert re.search(rf"(?m)^{field}\s*=", manifest) is None, path
