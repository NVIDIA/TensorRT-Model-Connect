# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal package smoke tests for model-owned runtime adapters."""

from __future__ import annotations

import fnmatch
import importlib.util
import json
import shutil
import sys
import tarfile
import types
from pathlib import Path

import pytest

from _pyproject_backend import _append_benchmark_catalog_to_sdist


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _manifest_audio_assets(manifest: Path) -> tuple[Path, ...]:
    """Return every local transcription input declared by one manifest."""

    raw = json.loads(manifest.read_text(encoding="utf-8"))
    family = manifest.parent.parent.resolve()
    source_prefix = Path("tests/e2e/models") / family.name
    assets: set[Path] = set()
    for testcase in raw.get("testcases", []):
        declared = testcase.get("test_input_audio") if isinstance(testcase, dict) else None
        if declared is None:
            continue
        assert isinstance(declared, str) and declared
        path = Path(declared).expanduser()
        candidate = path if path.is_absolute() else family / path
        if (
            not candidate.is_file()
            and not path.is_absolute()
            and path.is_relative_to(source_prefix)
        ):
            candidate = family / path.relative_to(source_prefix)
        resolved = candidate.resolve()
        assert resolved.is_relative_to(family) and resolved.is_file()
        assets.add(resolved)
    return tuple(sorted(assets))


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
        "trtmc_benchmark_worker",
        "libtrtmc_core.so",
        "libtrtmc_backend_trt.so",
        "models/example/libtrtmc_model_example.so",
    ):
        path = build / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())


def _package(
    recipe_module,
    source: Path,
    tmp_path: Path,
    *,
    include_sam2_header: bool = True,
    runpaths: list[tuple[Path, str]] | None = None,
) -> Path:
    build = tmp_path / "build"
    package = tmp_path / "package"
    _fake_native_build(build)
    benchmark_script = source / "scripts/trtmc-bench"
    benchmark_script.parent.mkdir(parents=True, exist_ok=True)
    benchmark_script.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    if include_sam2_header:
        sam2_header = source / "include/trtmc/models/sam2_video.h"
        sam2_header.parent.mkdir(parents=True, exist_ok=True)
        if not sam2_header.exists():
            sam2_header.write_text("// SAM2 public C ABI fixture\n", encoding="utf-8")
    catalog_family = source / "tests/e2e/models/example"
    (catalog_family / "manifests").mkdir(parents=True)
    (catalog_family / "MODEL.toml").write_text(
        'id = "example"\ntest_manifests = ["manifests/example.json"]\n',
        encoding="utf-8",
    )
    (catalog_family / "manifests/example.json").write_text(
        '{"fp8_scales": "data/fp8-scales.json", '
        '"testcases": [{"test_image": "data/test_img.jpeg", '
        '"prompt_file": "data/prompt.txt", '
        '"test_input_audio": "data/transcription.wav"}]}\n',
        encoding="utf-8",
    )
    (catalog_family / "data").mkdir()
    (catalog_family / "data/Recording.wav").write_bytes(b"RIFF-test-audio")
    (catalog_family / "data/fp8-scales.json").write_text("{}\n", encoding="utf-8")
    (catalog_family / "data/test_img.jpeg").write_bytes(b"test-image")
    (catalog_family / "data/prompt.txt").write_text("test prompt\n", encoding="utf-8")
    (catalog_family / "data/transcription.wav").write_bytes(b"RIFF-transcription-audio")
    recipe = recipe_module.TensorRTModelConnectConan()
    recipe.source_folder = str(source)
    recipe.build_folder = str(build)
    recipe.package_folder = str(package)
    set_runpath = recipe_module._set_wheel_runpath
    try:
        # This source-staging fixture uses text placeholders instead of ELF
        # build outputs. RUNPATH behavior has its own focused assertion below.
        if runpaths is None:
            recipe_module._set_wheel_runpath = lambda _path, _runpath: None
        else:
            recipe_module._set_wheel_runpath = lambda path, runpath: runpaths.append(
                (Path(path), runpath)
            )
        recipe.package()
    finally:
        recipe_module._set_wheel_runpath = set_runpath
    return package / "tensorrt_model_connect"


def test_sam2_native_builder_package_opt_in_controls_cmake_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRTMC_CONAN_ENABLE_SAM2_NATIVE_BUILDER", raising=False)
    recipe_module = _load_conan_recipe(monkeypatch)
    generated: list[dict[str, object]] = []

    class RecordingDeps:
        def __init__(self, _recipe) -> None:
            pass

        def generate(self) -> None:
            pass

    class RecordingToolchain:
        def __init__(self, _recipe) -> None:
            self.cache_variables: dict[str, object] = {}

        def generate(self) -> None:
            generated.append(dict(self.cache_variables))

    monkeypatch.setattr(recipe_module, "CMakeDeps", RecordingDeps)
    monkeypatch.setattr(recipe_module, "CMakeToolchain", RecordingToolchain)
    recipe = recipe_module.TensorRTModelConnectConan()

    recipe.generate()
    monkeypatch.setenv("TRTMC_CONAN_ENABLE_SAM2_NATIVE_BUILDER", "1")
    recipe.generate()

    assert [item["TRTMC_BUILD_SAM2_NATIVE_BUILDER"] for item in generated] == [False, True]


def test_package_keeps_sam2_native_builder_out_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    source = tmp_path / "source"
    build = tmp_path / "build"
    build.mkdir()
    (build / "sam2_native_builder").write_bytes(b"builder")

    module = _package(recipe_module, source, tmp_path)

    assert not (module / "bin/sam2_native_builder").exists()


def test_package_stages_only_the_opt_in_sam2_native_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRTMC_CONAN_ENABLE_SAM2_NATIVE_BUILDER", "true")
    recipe_module = _load_conan_recipe(monkeypatch)
    source = tmp_path / "source"
    build = tmp_path / "build"
    build.mkdir()
    (build / "sam2_native_builder").write_bytes(b"builder")
    (build / "sam2_native_benchmark").write_bytes(b"benchmark")
    runpaths: list[tuple[Path, str]] = []

    module = _package(recipe_module, source, tmp_path, runpaths=runpaths)

    packaged_builder = module / "bin/sam2_native_builder"
    assert packaged_builder.read_bytes() == b"builder"
    assert packaged_builder.stat().st_mode & 0o111 == 0o111
    assert not (module / "bin/sam2_native_benchmark").exists()
    assert (
        packaged_builder,
        "$ORIGIN:$ORIGIN/../../tensorrt_libs:/usr/local/cuda/lib64",
    ) in runpaths


def test_package_rejects_a_missing_opt_in_sam2_native_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRTMC_CONAN_ENABLE_SAM2_NATIVE_BUILDER", "yes")
    recipe_module = _load_conan_recipe(monkeypatch)

    with pytest.raises(
        recipe_module.ConanException,
        match="opt-in SAM2 native builder was not staged",
    ):
        _package(recipe_module, tmp_path / "source", tmp_path)


def test_package_fails_closed_when_sam2_public_header_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)

    with pytest.raises(
        recipe_module.ConanException,
        match="SAM2 public C ABI header was not staged",
    ):
        _package(
            recipe_module,
            tmp_path / "source",
            tmp_path,
            include_sam2_header=False,
        )


def test_package_stages_only_declared_existing_runtime_model_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    source = tmp_path / "source"
    runtime = source / "src/runtime/models/model_a"
    runtime.mkdir(parents=True)
    (runtime / "MODEL.toml").write_text(
        'id = "model_a"\n'
        'runtime_optional_data_files = ["publication/release-metadata.json", '
        '"publication/not-published-yet.json"]\n',
        encoding="utf-8",
    )
    published = runtime / "publication/release-metadata.json"
    published.parent.mkdir()
    published.write_text('{"public":true}\n', encoding="utf-8")
    (runtime / "undeclared.bundle").write_bytes(b"must not be packaged")

    module = _package(recipe_module, source, tmp_path)

    model_data = module / "model_data/model_a"
    assert (model_data / "publication/release-metadata.json").read_bytes() == (
        published.read_bytes()
    )
    assert not (model_data / "publication/not-published-yet.json").exists()
    assert not (model_data / "undeclared.bundle").exists()


def test_repository_predeclares_only_the_public_sam2_qualification_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    declarations = recipe_module._runtime_optional_data_files(REPOSITORY_ROOT)

    assert [
        (model, relative.as_posix(), source.relative_to(REPOSITORY_ROOT).as_posix())
        for model, relative, source in declarations
    ] == [
        (
            "sam2",
            "sam2-l4-trt11.1-contract5-0001.qualification-record.json",
            "src/runtime/models/sam2/sam2-l4-trt11.1-contract5-0001.qualification-record.json",
        ),
        (
            "sam2",
            "sam2-l4-trt11.1-contract5-0001.qualification-audit.json",
            "src/runtime/models/sam2/sam2-l4-trt11.1-contract5-0001.qualification-audit.json",
        ),
    ]
    assert all(relative.suffix != ".bundle" for _, relative, _ in declarations)


def test_package_rejects_unsafe_runtime_model_data_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    source = tmp_path / "source"
    runtime = source / "src/runtime/models/model_a"
    runtime.mkdir(parents=True)
    (runtime / "MODEL.toml").write_text(
        'id = "model_a"\nruntime_optional_data_files = ["../record.json"]\n',
        encoding="utf-8",
    )

    with pytest.raises(recipe_module.ConanException, match="unsafe relative path"):
        _package(recipe_module, source, tmp_path)


def test_wheel_runpath_rewrite_invokes_patchelf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    binary = tmp_path / "libtrtmc_core.so"
    binary.write_bytes(b"ELF fixture")
    calls = []
    monkeypatch.setattr(
        recipe_module.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs))
    )

    recipe_module._set_wheel_runpath(binary, "$ORIGIN:/usr/local/cuda/lib64")

    assert calls == [
        (
            (["patchelf", "--set-rpath", "$ORIGIN:/usr/local/cuda/lib64", str(binary)],),
            {"check": True, "capture_output": True, "text": True},
        )
    ]


def test_conan_wheel_script_directory_uses_selected_package_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRTMC_PACKAGE_VERSION", "0.1.0+trt111")
    recipe_module = _load_conan_recipe(monkeypatch)
    source = tmp_path / "source"

    module = _package(recipe_module, source, tmp_path)

    scripts = module.parent / "tensorrt_model_connect-0.1.0+trt111.data/scripts"
    assert (scripts / "trtmc").is_file()
    assert (scripts / "trtmc-bench").is_file()
    assert (scripts / "libtrtmc_core.so").is_file()


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
    assert (sdk / "trtmc" / "models" / "sam2_video.h").is_file()
    assert (module / "bin" / "trtmc").is_file()
    assert (module / "bin" / "trtmc_benchmark_worker").is_file()
    benchmark_script = module.parent / "tensorrt_model_connect-0.1.0.data/scripts/trtmc-bench"
    assert benchmark_script.read_bytes().startswith(b"#!python\n")
    catalog = module / "benchmark" / "_catalog" / "example"
    assert (catalog / "MODEL.toml").is_file()
    assert (catalog / "manifests" / "example.json").is_file()
    assert (catalog / "data/Recording.wav").is_file()
    assert (catalog / "data/fp8-scales.json").is_file()
    assert (catalog / "data/test_img.jpeg").is_file()
    assert (catalog / "data/prompt.txt").is_file()
    assert (catalog / "data/transcription.wav").is_file()


def test_package_stages_the_complete_canonical_benchmark_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe_module = _load_conan_recipe(monkeypatch)
    package = tmp_path / "tensorrt_model_connect"

    recipe_module._stage_benchmark_catalog(
        recipe_module.TensorRTModelConnectConan(), REPOSITORY_ROOT, package
    )

    source = REPOSITORY_ROOT / "tests/e2e/models"
    installed = package / "benchmark/_catalog"
    assert len(list(installed.glob("*/MODEL.toml"))) == len(list(source.glob("*/MODEL.toml")))
    assert len(list(installed.glob("*/manifests/*.json"))) == len(
        list(source.glob("*/manifests/*.json"))
    )
    assert len(list(installed.glob("*/data/Recording.wav"))) == len(
        list(source.glob("*/data/Recording.wav"))
    )
    assert (installed / "gpt2/manifests/distilgpt2.json").is_file()
    assert (installed / "whisper/data/Recording.wav").is_file()
    assert (installed / "whisper/data/librispeech-test-clean-6930-75918-0003.wav").is_file()
    assert (installed / "flux/data/flux2-fp8-scales.json").is_file()
    assert (installed / "qwen_image/data/test_img.jpeg").is_file()
    assert (installed / "sana_wm/assets/demo_0.png").is_file()
    assert (installed / "sana_wm/assets/demo_0.txt").is_file()
    missing_audio_assets = [
        asset.relative_to(source).as_posix()
        for manifest in source.glob("*/manifests/*.json")
        for asset in _manifest_audio_assets(manifest)
        if not (installed / asset.relative_to(source)).is_file()
    ]
    assert not missing_audio_assets


def test_sdist_appends_only_the_minimal_benchmark_catalog(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    pyproject = project / "pyproject.toml"
    pyproject.write_text("[build-system]\n", encoding="utf-8")
    family = project / "tests/e2e/models/example"
    (family / "manifests").mkdir(parents=True)
    (family / "MODEL.toml").write_text('id = "example"\n', encoding="utf-8")
    (family / "manifests/example.json").write_text(
        '{"fp8_scales": "data/fp8-scales.json", '
        '"testcases": [{"test_image": "data/test_img.jpeg", '
        '"prompt_file": "data/prompt.txt", '
        '"test_input_audio": "data/transcription.wav"}]}\n',
        encoding="utf-8",
    )
    (family / "data").mkdir()
    (family / "data/Recording.wav").write_bytes(b"RIFF-test-audio")
    (family / "data/fp8-scales.json").write_text("{}\n", encoding="utf-8")
    (family / "data/test_img.jpeg").write_bytes(b"test-image")
    (family / "data/prompt.txt").write_text("test prompt\n", encoding="utf-8")
    (family / "data/transcription.wav").write_bytes(b"RIFF-transcription-audio")
    (family / "data/not-a-benchmark-input.bin").write_bytes(b"large fixture")
    archive = tmp_path / "example-0.1.0.tar.gz"
    with tarfile.open(archive, "w:gz") as destination:
        destination.add(pyproject, arcname="example-0.1.0/pyproject.toml")
    monkeypatch.chdir(project)

    _append_benchmark_catalog_to_sdist(archive)

    with tarfile.open(archive, "r:gz") as source:
        names = set(source.getnames())
    prefix = "example-0.1.0/tests/e2e/models/example"
    assert f"{prefix}/MODEL.toml" in names
    assert f"{prefix}/manifests/example.json" in names
    assert f"{prefix}/data/Recording.wav" in names
    assert f"{prefix}/data/fp8-scales.json" in names
    assert f"{prefix}/data/test_img.jpeg" in names
    assert f"{prefix}/data/prompt.txt" in names
    assert f"{prefix}/data/transcription.wav" in names
    assert f"{prefix}/data/not-a-benchmark-input.bin" not in names


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
