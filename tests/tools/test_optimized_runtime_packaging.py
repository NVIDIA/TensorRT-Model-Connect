# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package-stage contracts for installable optimized-runtime capsules."""

from __future__ import annotations

import fnmatch
import importlib.util
import shutil
import sys
import tarfile
import tomllib
import types
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _compact_identifier(value: str) -> str:
    """Normalize a manifest identifier for source-level dependency checks."""

    return "".join(character for character in value.lower() if character.isalnum())


def _load_conan_recipe(monkeypatch: pytest.MonkeyPatch):
    """Load the real recipe with a filesystem-faithful Conan API test double."""

    conan_module = types.ModuleType("conan")
    errors_module = types.ModuleType("conan.errors")
    tools_module = types.ModuleType("conan.tools")
    cmake_module = types.ModuleType("conan.tools.cmake")
    files_module = types.ModuleType("conan.tools.files")

    class ConanFile:
        pass

    class ConanException(Exception):
        pass

    class _UnusedCMakeHelper:
        def __init__(self, *_args, **_kwargs):
            pass

    def _unused_layout(*_args, **_kwargs) -> None:
        return None

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
                fnmatch.fnmatch(source.name, pattern) or fnmatch.fnmatch(relative_name, pattern)
            ):
                continue
            if any(
                fnmatch.fnmatch(source.name, excluded) or fnmatch.fnmatch(relative_name, excluded)
                for excluded in excludes
            ):
                continue
            destination = Path(dst) / (relative if keep_path else source.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    conan_module.ConanFile = ConanFile
    errors_module.ConanException = ConanException
    cmake_module.CMake = _UnusedCMakeHelper
    cmake_module.CMakeDeps = _UnusedCMakeHelper
    cmake_module.CMakeToolchain = _UnusedCMakeHelper
    cmake_module.cmake_layout = _unused_layout
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
    (build / "models" / "example").mkdir(parents=True)
    for relative in (
        "trtmc",
        "libtrtmc_core.so",
        "libtrtmc_backend_trt.so",
        "models/example/libtrtmc_model_example.so",
    ):
        path = build / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"fake native package payload: {relative}".encode())


def _fake_package_source(source: Path, *, with_adapter: bool) -> None:
    source.mkdir(parents=True)
    for relative in (
        "src/runtime/providers/optimized_runtime_factory.h",
        "include/trtmc/pipeline.h",
    ):
        header = source / relative
        header.parent.mkdir(parents=True, exist_ok=True)
        header.write_text(f"// fake private SDK source: {relative}\n", encoding="utf-8")
    if not with_adapter:
        return
    builder = source / "python" / "tensorrt_model_connect" / "families" / "model_a" / "runtime_a"
    builder.mkdir(parents=True)
    (builder / "IMPLEMENTATION.toml").write_text(
        'schema_version = 1\ndownstream_runtime = "runtime-a"\n',
        encoding="utf-8",
    )
    (builder / "adapter.py").write_text("# model-owned builder adapter\n", encoding="utf-8")
    (builder / "README.md").write_text("development documentation\n", encoding="utf-8")
    dependency_source = builder / "dependencies" / "runtime-a" / "src" / "runtime.cpp"
    dependency_source.parent.mkdir(parents=True)
    dependency_source.write_text("// downstream source must remain lazy\n", encoding="utf-8")
    (dependency_source.parent / "libdependency.so").write_bytes(b"local dependency output")
    runtime = source / "src" / "runtime" / "models" / "model_a" / "runtime_a"
    runtime.mkdir(parents=True)
    (runtime / "CMakeLists.txt").write_text("# model-owned runtime build\n", encoding="utf-8")
    (runtime / "adapter.cpp").write_text("// model-owned runtime adapter\n", encoding="utf-8")


def test_full_repository_package_stages_model_owned_adapters_for_installed_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    build = tmp_path / "build"
    package = tmp_path / "package"
    _fake_native_build(build)

    recipe = recipe_module.TensorRTModelConnectConan()
    recipe.source_folder = str(REPOSITORY_ROOT)
    recipe.build_folder = str(build)
    recipe.package_folder = str(package)
    recipe.package()

    package_module = package / "tensorrt_model_connect"
    source_adapters = recipe_module._model_owned_adapters(REPOSITORY_ROOT)
    assert source_adapters
    for family, adapter, builder_source, runtime_source in source_adapters:
        packaged_adapter = package_module / "families" / family / adapter
        expected_files = recipe_module._inert_adapter_files(builder_source) | {
            f"runtime/{relative}" for relative in recipe_module._inert_adapter_files(runtime_source)
        }
        assert recipe_module._inert_adapter_files(packaged_adapter) == expected_files
        assert (packaged_adapter / "IMPLEMENTATION.toml").is_file()
        assert (packaged_adapter / "runtime").is_dir()
    assert not (package_module / "third_party").exists()
    assert not (package_module / "optimized_runtimes").exists()
    assert not list((package_module / "families").glob("**/dependencies/**"))
    assert not list((package_module / "families").glob("**/tests/**"))
    assert not list((package_module / "families").rglob("README.md"))
    assert not list((package_module / "families").rglob("*.so*"))
    assert not list((package_module / "families").rglob("engine.dir"))
    private_sdk = package_module / "optimized_runtime" / "_sdk" / "include"
    assert {
        path.relative_to(private_sdk).as_posix()
        for path in private_sdk.rglob("*")
        if path.is_file()
    } == {
        "runtime/providers/optimized_runtime_factory.h",
        "trtmc/pipeline.h",
    }


@pytest.mark.parametrize("with_adapter", (False, True))
def test_conan_package_supports_native_only_and_excludes_adapter_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_adapter: bool,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    source = tmp_path / "source"
    build = tmp_path / "build"
    package = tmp_path / "package"
    _fake_package_source(source, with_adapter=with_adapter)
    _fake_native_build(build)

    recipe = recipe_module.TensorRTModelConnectConan()
    recipe.source_folder = str(source)
    recipe.build_folder = str(build)
    recipe.package_folder = str(package)
    recipe.package()

    package_module = package / "tensorrt_model_connect"
    packaged_adapter = package_module / "families" / "model_a" / "runtime_a"
    expected_adapter_files = (
        {
            "IMPLEMENTATION.toml",
            "adapter.py",
            "runtime/CMakeLists.txt",
            "runtime/adapter.cpp",
        }
        if with_adapter
        else set()
    )
    assert recipe_module._inert_adapter_files(packaged_adapter) == expected_adapter_files
    assert not list((package_module / "families").glob("**/dependencies/**"))
    assert (package_module / "bin" / "trtmc").is_file()
    private_sdk = package_module / "optimized_runtime" / "_sdk" / "include"
    assert {
        path.relative_to(private_sdk).as_posix()
        for path in private_sdk.rglob("*")
        if path.is_file()
    } == {
        "runtime/providers/optimized_runtime_factory.h",
        "trtmc/pipeline.h",
    }


def test_conan_package_rejects_generated_runtime_payloads_generically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    source = tmp_path / "source"
    build = tmp_path / "build"
    package = tmp_path / "package"
    _fake_package_source(source, with_adapter=True)
    _fake_native_build(build)
    generated = source / "src" / "runtime" / "models" / "model_a" / "runtime_a" / "runtime.plan"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_bytes(b"generated runtime payload")

    recipe = recipe_module.TensorRTModelConnectConan()
    recipe.source_folder = str(source)
    recipe.build_folder = str(build)
    recipe.package_folder = str(package)

    with pytest.raises(recipe_module.ConanException, match="generated runtime artifacts"):
        recipe.package()


@pytest.mark.parametrize(
    "relative",
    (
        "python/tensorrt_model_connect/families/model_a/runtime_a/evidence/a100.json",
        "python/tensorrt_model_connect/families/model_a/runtime_a/qualifications/result.json",
        "src/runtime/models/model_a/runtime_a/results/performance.json",
    ),
)
def test_conan_package_rejects_model_adapter_proof_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    source = tmp_path / "source"
    build = tmp_path / "build"
    package = tmp_path / "package"
    _fake_package_source(source, with_adapter=True)
    _fake_native_build(build)
    proof = source / relative
    proof.parent.mkdir(parents=True, exist_ok=True)
    proof.write_text("{}\n", encoding="utf-8")

    recipe = recipe_module.TensorRTModelConnectConan()
    recipe.source_folder = str(source)
    recipe.build_folder = str(build)
    recipe.package_folder = str(package)

    with pytest.raises(recipe_module.ConanException, match="evidence, test, or build"):
        recipe.package()


def test_conan_package_rejects_nested_model_adapter_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    source = tmp_path / "source"
    build = tmp_path / "build"
    package = tmp_path / "package"
    _fake_package_source(source, with_adapter=True)
    _fake_native_build(build)
    adapter = source / "python" / "tensorrt_model_connect" / "families" / "model_a" / "runtime_a"
    target = adapter / "adapter.py"
    (adapter / "linked-adapter.py").symlink_to(target.name)

    recipe = recipe_module.TensorRTModelConnectConan()
    recipe.source_folder = str(source)
    recipe.build_folder = str(build)
    recipe.package_folder = str(package)

    with pytest.raises(recipe_module.ConanException, match="must not contain symlinks"):
        recipe.package()


def test_source_distribution_keeps_model_adapter_dependencies_lazy() -> None:
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    sdist = pyproject.split("[tool.conan-py-build.sdist]", maxsplit=1)[1]
    assert '"optimized_runtimes"' not in sdist
    assert '"dependencies"' in sdist
    assert '"third_party/stb"' in sdist
    assert '"third_party",' not in sdist

    conanfile = (REPOSITORY_ROOT / "conanfile.py").read_text(encoding="utf-8")
    requirements = conanfile.split("def requirements", maxsplit=1)[1].split(
        "def layout", maxsplit=1
    )[0]
    build = conanfile.split("def build(self)", maxsplit=1)[1].split(
        "def package(self)", maxsplit=1
    )[0]
    compact_requirements = _compact_identifier(requirements)
    compact_build = _compact_identifier(build)
    assert "optimizedruntimes" not in compact_requirements
    assert "optimizedruntimes" not in compact_build
    assert "thirdparty" not in compact_requirements
    assert "thirdparty" not in compact_build


def _fake_sdist_backend(source: Path):
    class Backend:
        @staticmethod
        def build_sdist(sdist_directory: str, _config_settings) -> str:
            filename = "tensorrt_model_connect-0.1.0.tar.gz"
            archive_path = Path(sdist_directory) / filename
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive_path, mode="w:gz") as archive:
                archive.add(source, arcname="tensorrt_model_connect-0.1.0")
            return filename

    return Backend()


def _fake_sdist_source(root: Path, *, include_runtime: bool = True) -> Path:
    source = root / "source"
    builder = source / "python" / "tensorrt_model_connect" / "families" / "model_a" / "runtime_a"
    runtime = source / "src" / "runtime" / "models" / "model_a" / "runtime_a"
    builder.mkdir(parents=True)
    (builder / "IMPLEMENTATION.toml").write_text("schema_version = 1\n", encoding="utf-8")
    (builder / "adapter.py").write_text("# model-owned builder\n", encoding="utf-8")
    if include_runtime:
        runtime.mkdir(parents=True)
        (runtime / "CMakeLists.txt").write_text("# model-owned runtime build\n", encoding="utf-8")
        (runtime / "adapter.cpp").write_text("// model-owned runtime\n", encoding="utf-8")
    return source


def test_sdist_archive_validation_accepts_only_model_adapter_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import _pyproject_backend as backend

    source = _fake_sdist_source(tmp_path)
    monkeypatch.setattr(backend, "_conan_build_backend", lambda: _fake_sdist_backend(source))

    filename = backend.build_sdist(str(tmp_path / "dist"))

    assert (tmp_path / "dist" / filename).is_file()


def test_sdist_archive_validation_rejects_builder_without_runtime_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import _pyproject_backend as backend

    source = _fake_sdist_source(tmp_path, include_runtime=False)
    monkeypatch.setattr(backend, "_conan_build_backend", lambda: _fake_sdist_backend(source))

    with pytest.raises(RuntimeError, match="missing model-adapter Runtime source"):
        backend.build_sdist(str(tmp_path / "dist"))


@pytest.mark.parametrize(
    "relative",
    (
        "python/tensorrt_model_connect/families/model_a/runtime_a/dependencies/vendor/source.cpp",
        "python/tensorrt_model_connect/families/model_a/runtime_a/evidence/a100.json",
        "src/runtime/models/model_a/runtime_a/artifacts/runtime.plan",
    ),
)
def test_sdist_archive_validation_rejects_non_source_model_adapter_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    import _pyproject_backend as backend

    source = _fake_sdist_source(tmp_path)
    forbidden = source / relative
    forbidden.parent.mkdir(parents=True, exist_ok=True)
    forbidden.write_text("not source\n", encoding="utf-8")
    monkeypatch.setattr(backend, "_conan_build_backend", lambda: _fake_sdist_backend(source))

    with pytest.raises(RuntimeError, match="forbidden|generated"):
        backend.build_sdist(str(tmp_path / "dist"))


def test_full_repository_adapters_are_not_eager_build_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conanfile = (REPOSITORY_ROOT / "conanfile.py").read_text(encoding="utf-8")
    requirements = conanfile.split("def requirements", maxsplit=1)[1].split(
        "def layout", maxsplit=1
    )[0]
    build = conanfile.split("def build(self)", maxsplit=1)[1].split(
        "def package(self)", maxsplit=1
    )[0]
    compact_requirements = _compact_identifier(requirements)
    compact_build = _compact_identifier(build)

    adapters = _load_conan_recipe(monkeypatch)._model_owned_adapters(REPOSITORY_ROOT)
    assert adapters
    for _family, _adapter, builder_source, _runtime_source in adapters:
        with (builder_source / "IMPLEMENTATION.toml").open("rb") as manifest_file:
            manifest = tomllib.load(manifest_file)
        downstream = _compact_identifier(str(manifest["downstream_runtime"]))
        assert downstream not in compact_requirements
        assert downstream not in compact_build
