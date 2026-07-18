# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal package smoke tests for model-owned runtime adapters."""

from __future__ import annotations

import fnmatch
import importlib.util
import shutil
import sys
import types
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _load_conan_recipe(monkeypatch: pytest.MonkeyPatch):
    """Load the recipe with the subset of Conan used by package()."""

    conan_module = types.ModuleType("conan")
    errors_module = types.ModuleType("conan.errors")
    tools_module = types.ModuleType("conan.tools")
    cmake_module = types.ModuleType("conan.tools.cmake")
    files_module = types.ModuleType("conan.tools.files")

    class ConanFile:
        pass

    class ConanException(Exception):
        pass

    class UnusedCMake:
        def __init__(self, *_args, **_kwargs):
            raise AssertionError("package() must not configure or build an adapter")

    def copy(
        _recipe,
        pattern: str,
        *,
        src: str,
        dst: str,
        keep_path: bool = True,
        excludes: tuple[str, ...] = (),
    ) -> None:
        source_root = Path(src)
        if not source_root.is_dir():
            return
        for source in source_root.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(source_root)
            relative_name = relative.as_posix()
            if not (
                fnmatch.fnmatch(source.name, pattern)
                or fnmatch.fnmatch(relative_name, pattern)
            ):
                continue
            if any(
                fnmatch.fnmatch(source.name, excluded)
                or fnmatch.fnmatch(relative_name, excluded)
                for excluded in excludes
            ):
                continue
            destination = Path(dst) / (relative if keep_path else source.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    conan_module.ConanFile = ConanFile
    errors_module.ConanException = ConanException
    cmake_module.CMake = UnusedCMake
    cmake_module.CMakeDeps = UnusedCMake
    cmake_module.CMakeToolchain = UnusedCMake
    cmake_module.cmake_layout = lambda *_args, **_kwargs: None
    files_module.copy = copy

    monkeypatch.setitem(sys.modules, "conan", conan_module)
    monkeypatch.setitem(sys.modules, "conan.errors", errors_module)
    monkeypatch.setitem(sys.modules, "conan.tools", tools_module)
    monkeypatch.setitem(sys.modules, "conan.tools.cmake", cmake_module)
    monkeypatch.setitem(sys.modules, "conan.tools.files", files_module)

    spec = importlib.util.spec_from_file_location(
        "_trtmc_test_conanfile", REPOSITORY_ROOT / "conanfile.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_native_build(build: Path) -> None:
    for relative in (
        "trtmc",
        "libtrtmc_core.so",
        "libtrtmc_backend_trt.so",
        "models/example/libtrtmc_model_example.so",
    ):
        path = build / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())


def _package(recipe_module, source: Path, tmp_path: Path) -> Path:
    build = tmp_path / "build"
    package = tmp_path / "package"
    _fake_native_build(build)
    recipe = recipe_module.TensorRTModelConnectConan()
    recipe.source_folder = str(source)
    recipe.build_folder = str(build)
    recipe.package_folder = str(package)
    recipe.package()
    return package / "tensorrt_model_connect"


def test_package_stages_a_model_owned_adapter_as_inert_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    source = tmp_path / "source"
    builder = source / "python/tensorrt_model_connect/families/model_a/runtime_a"
    runtime = source / "src/runtime/models/model_a/runtime_a"
    builder.mkdir(parents=True)
    runtime.mkdir(parents=True)
    (builder / "IMPLEMENTATION.toml").write_text("not valid TOML [", encoding="utf-8")
    (builder / "adapter.py").write_text("# adapter\n", encoding="utf-8")
    profile = builder / "profiles" / "profile.toml"
    profile.parent.mkdir()
    profile.write_text("profile = true\n", encoding="utf-8")
    dependency = builder / "dependencies" / "vendor" / "libvendor.so"
    dependency.parent.mkdir(parents=True)
    dependency.write_bytes(b"must remain lazy")
    (runtime / "CMakeLists.txt").write_text("# runtime\n", encoding="utf-8")
    (runtime / "adapter.cpp").write_text("// runtime\n", encoding="utf-8")
    for relative in (
        "src/runtime/providers/optimized_runtime_factory.h",
        "include/trtmc/pipeline.h",
    ):
        header = source / relative
        header.parent.mkdir(parents=True, exist_ok=True)
        header.write_text("// sdk\n", encoding="utf-8")

    module = _package(recipe_module, source, tmp_path)

    packaged = module / "families" / "model_a" / "runtime_a"
    assert {
        path.relative_to(packaged).as_posix() for path in packaged.rglob("*") if path.is_file()
    } == {
        "IMPLEMENTATION.toml",
        "adapter.py",
        "profiles/profile.toml",
        "runtime/CMakeLists.txt",
        "runtime/adapter.cpp",
    }
    assert (packaged / "IMPLEMENTATION.toml").read_text(encoding="utf-8") == "not valid TOML ["
    sdk = module / "runtime_provider" / "_sdk" / "include"
    assert (sdk / "runtime" / "providers" / "optimized_runtime_factory.h").is_file()
    assert (sdk / "trtmc" / "pipeline.h").is_file()
    assert (module / "bin" / "trtmc").is_file()


def test_package_rejects_builder_without_matching_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    source = tmp_path / "source"
    builder = source / "python/tensorrt_model_connect/families/model_a/runtime_a"
    builder.mkdir(parents=True)
    (builder / "IMPLEMENTATION.toml").write_text("not parsed [", encoding="utf-8")

    with pytest.raises(
        recipe_module.ConanException,
        match="model_a/runtime_a has no matching runtime source directory",
    ):
        _package(recipe_module, source, tmp_path)
