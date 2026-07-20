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


def _write_private_sdk(source: Path) -> None:
    for relative in (
        "src/runtime/providers/optimized_runtime_factory.h",
        "include/trtmc/pipeline.h",
    ):
        header = source / relative
        header.parent.mkdir(parents=True, exist_ok=True)
        header.write_text("// sdk\n", encoding="utf-8")


def _write_nested_adapter(
    source: Path,
    *,
    profile: str,
    tag: str,
    with_runtime: bool = True,
    with_excluded_files: bool = False,
) -> tuple[Path, Path]:
    relative = Path("example_runtime_adapter") / profile
    builder = source / "python/tensorrt_model_connect/families/model_a" / relative
    runtime = source / "src/runtime/models/model_a" / relative
    builder.mkdir(parents=True)
    (builder / "IMPLEMENTATION.toml").write_text(
        f'implementation = "{tag}"\n', encoding="utf-8"
    )
    (builder / "adapter.py").write_text(f'# adapter-{tag}\n', encoding="utf-8")
    (builder / "dependency.lock").write_text(f"dependency-{tag}\n", encoding="utf-8")
    profile_file = builder / "profiles" / "target.toml"
    profile_file.parent.mkdir()
    profile_file.write_text(f'profile = "{tag}"\n', encoding="utf-8")

    if with_runtime:
        runtime.mkdir(parents=True)
        (runtime / "CMakeLists.txt").write_text(f"# runtime-{tag}\n", encoding="utf-8")
        (runtime / "adapter.cpp").write_text(f"// runtime-{tag}\n", encoding="utf-8")

    if with_excluded_files:
        excluded = (
            builder / "dependencies" / "vendor" / "libvendor.so",
            builder / ".runtime-build" / "adapter.o",
            builder / "tests" / "test_private.py",
            builder / "__pycache__" / "adapter.pyc",
        )
        for path in excluded:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"must remain lazy")

    return builder, runtime


def _inventory(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }


def test_package_stages_a_nested_model_owned_adapter_as_inert_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    source = tmp_path / "source"
    _write_nested_adapter(source, profile="profile_a", tag="a")
    _write_private_sdk(source)

    module = _package(recipe_module, source, tmp_path)

    packaged = module / "families" / "model_a" / "example_runtime_adapter" / "profile_a"
    assert _inventory(packaged) == {
        "IMPLEMENTATION.toml",
        "adapter.py",
        "dependency.lock",
        "profiles/target.toml",
        "runtime/CMakeLists.txt",
        "runtime/adapter.cpp",
    }
    assert (packaged / "IMPLEMENTATION.toml").read_text(encoding="utf-8") == (
        'implementation = "a"\n'
    )
    sdk = module / "runtime_provider" / "_sdk" / "include"
    assert (sdk / "runtime" / "providers" / "optimized_runtime_factory.h").is_file()
    assert (sdk / "trtmc" / "pipeline.h").is_file()
    assert (module / "bin" / "trtmc").is_file()


def test_package_keeps_two_profiles_leaf_local_without_collisions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    source = tmp_path / "source"
    _write_nested_adapter(source, profile="profile_a", tag="a")
    _write_nested_adapter(source, profile="profile_b", tag="b")
    _write_private_sdk(source)

    module = _package(recipe_module, source, tmp_path)
    adapter = module / "families" / "model_a" / "example_runtime_adapter"

    assert sorted(path.parent.name for path in adapter.glob("*/IMPLEMENTATION.toml")) == [
        "profile_a",
        "profile_b",
    ]
    assert (adapter / "profile_a" / "adapter.py").read_text(encoding="utf-8") == "# adapter-a\n"
    assert (adapter / "profile_b" / "adapter.py").read_text(encoding="utf-8") == "# adapter-b\n"
    assert (adapter / "profile_a" / "runtime" / "adapter.cpp").read_text(
        encoding="utf-8"
    ) == "// runtime-a\n"
    assert (adapter / "profile_b" / "runtime" / "adapter.cpp").read_text(
        encoding="utf-8"
    ) == "// runtime-b\n"


def test_package_excludes_lazy_dependencies_build_outputs_and_private_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    source = tmp_path / "source"
    _write_nested_adapter(
        source,
        profile="profile_a",
        tag="a",
        with_excluded_files=True,
    )
    _write_private_sdk(source)

    module = _package(recipe_module, source, tmp_path)
    packaged = module / "families" / "model_a" / "example_runtime_adapter" / "profile_a"

    assert _inventory(packaged) == {
        "IMPLEMENTATION.toml",
        "adapter.py",
        "dependency.lock",
        "profiles/target.toml",
        "runtime/CMakeLists.txt",
        "runtime/adapter.cpp",
    }


def test_package_rejects_nested_builder_without_matching_nested_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    source = tmp_path / "source"
    _write_nested_adapter(
        source,
        profile="profile_a",
        tag="a",
        with_runtime=False,
    )

    with pytest.raises(
        recipe_module.ConanException,
        match=(
            "model_a/example_runtime_adapter/profile_a has no matching runtime source directory"
        ),
    ):
        _package(recipe_module, source, tmp_path)


@pytest.mark.parametrize(
    "relative_manifest",
    (
        "example_runtime_adapter/IMPLEMENTATION.toml",
        "example_runtime_adaptor/profile_a/IMPLEMENTATION.toml",
        "example_runtime_adapter/profile_a/extra/IMPLEMENTATION.toml",
    ),
)
def test_package_ignores_noncanonical_manifest_layouts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_manifest: str,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    source = tmp_path / "source"
    manifest = (
        source
        / "python/tensorrt_model_connect/families/model_a"
        / relative_manifest
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text("noncanonical = true\n", encoding="utf-8")
    _write_private_sdk(source)

    module = _package(recipe_module, source, tmp_path)

    assert not (module / "families/model_a").exists()
    assert (module / "bin/trtmc").is_file()


def test_package_does_not_recursively_interpret_deep_vendor_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    source = tmp_path / "source"
    builder, _runtime = _write_nested_adapter(
        source,
        profile="profile_a",
        tag="a",
        with_excluded_files=True,
    )
    vendor_manifest = builder / "dependencies/vendor/third_party/IMPLEMENTATION.toml"
    vendor_manifest.parent.mkdir(parents=True, exist_ok=True)
    vendor_manifest.write_text("vendor = true\n", encoding="utf-8")
    _write_private_sdk(source)

    module = _package(recipe_module, source, tmp_path)
    packaged = module / "families/model_a/example_runtime_adapter/profile_a"

    assert (packaged / "IMPLEMENTATION.toml").is_file()
    assert not (packaged / "dependencies").exists()


@pytest.mark.parametrize("component", ("provider", "profile", "manifest"))
def test_package_rejects_symlinked_builder_capsule_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    component: str,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    source = tmp_path / "source"
    family = source / "python/tensorrt_model_connect/families/model_a"
    provider = family / "example_runtime_adapter"
    profile = provider / "profile_a"
    manifest = profile / "IMPLEMENTATION.toml"

    if component == "provider":
        target = source / "outside-provider"
        target_profile = target / "profile_a"
        target_profile.mkdir(parents=True)
        (target_profile / "IMPLEMENTATION.toml").write_text("linked = true\n", encoding="utf-8")
        family.mkdir(parents=True)
        provider.symlink_to(target, target_is_directory=True)
    elif component == "profile":
        target = source / "outside-profile"
        target.mkdir(parents=True)
        (target / "IMPLEMENTATION.toml").write_text("linked = true\n", encoding="utf-8")
        provider.mkdir(parents=True)
        profile.symlink_to(target, target_is_directory=True)
    else:
        target = source / "outside-manifest.toml"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("linked = true\n", encoding="utf-8")
        profile.mkdir(parents=True)
        manifest.symlink_to(target)

    with pytest.raises(
        recipe_module.ConanException,
        match=rf"build adapter {component}.*must not be a symlink",
    ):
        _package(recipe_module, source, tmp_path)


def test_package_rejects_symlinked_builder_family_alias_within_family_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    source = tmp_path / "source"
    families = source / "python/tensorrt_model_connect/families"
    sibling_family = families / "model_b"
    manifest = sibling_family / "example_runtime_adapter/profile_a/IMPLEMENTATION.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("aliased = true\n", encoding="utf-8")
    (families / "model_a").symlink_to(sibling_family, target_is_directory=True)

    with pytest.raises(
        recipe_module.ConanException,
        match=r"build adapter family directory must not be a symlink",
    ):
        _package(recipe_module, source, tmp_path)


def test_package_rejects_families_root_resolving_outside_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    source = tmp_path / "source"
    outside_families = tmp_path / "outside-families"
    manifest = outside_families / "model_a/example_runtime_adapter/profile_a/IMPLEMENTATION.toml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("escaped = true\n", encoding="utf-8")
    family_parent = source / "python/tensorrt_model_connect"
    family_parent.mkdir(parents=True)
    (family_parent / "families").symlink_to(outside_families, target_is_directory=True)

    with pytest.raises(
        recipe_module.ConanException,
        match=r"Model family packaging root escapes or cannot be resolved below",
    ):
        _package(recipe_module, source, tmp_path)


@pytest.mark.parametrize("component", ("family", "provider", "profile"))
def test_package_rejects_symlinked_runtime_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    component: str,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    source = tmp_path / "source"
    _write_nested_adapter(
        source,
        profile="profile_a",
        tag="a",
        with_runtime=False,
    )
    runtime_root = source / "src/runtime/models"
    family = runtime_root / "model_a"
    provider = family / "example_runtime_adapter"
    profile = provider / "profile_a"

    if component == "family":
        target = source / "outside-runtime-family"
        (target / "example_runtime_adapter/profile_a").mkdir(parents=True)
        runtime_root.mkdir(parents=True)
        family.symlink_to(target, target_is_directory=True)
    elif component == "provider":
        target = source / "outside-runtime-provider"
        (target / "profile_a").mkdir(parents=True)
        family.mkdir(parents=True)
        provider.symlink_to(target, target_is_directory=True)
    else:
        target = source / "outside-runtime-profile"
        target.mkdir(parents=True)
        provider.mkdir(parents=True)
        profile.symlink_to(target, target_is_directory=True)

    with pytest.raises(
        recipe_module.ConanException,
        match=rf"runtime {component} directory must not be a symlink",
    ):
        _package(recipe_module, source, tmp_path)


def test_package_rejects_runtime_root_resolving_outside_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    source = tmp_path / "source"
    _write_nested_adapter(
        source,
        profile="profile_a",
        tag="a",
        with_runtime=False,
    )
    outside_runtime_root = tmp_path / "outside-runtime-models"
    (outside_runtime_root / "model_a/example_runtime_adapter/profile_a").mkdir(parents=True)
    runtime_parent = source / "src/runtime"
    runtime_parent.mkdir(parents=True)
    (runtime_parent / "models").symlink_to(outside_runtime_root, target_is_directory=True)

    with pytest.raises(
        recipe_module.ConanException,
        match=r"runtime packaging root escapes or cannot be resolved below",
    ):
        _package(recipe_module, source, tmp_path)
