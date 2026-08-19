# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static contract for one self-contained source root per model family."""

from __future__ import annotations

from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "python" / "tensorrt_model_connect" / "models"
LEGACY_ROOTS = (
    REPO_ROOT / "python" / "tensorrt_model_connect" / "families",
    REPO_ROOT / "src" / "runtime" / "models",
    REPO_ROOT / "tests" / "e2e" / "models",
    REPO_ROOT / "tests" / "cpp" / "models",
    REPO_ROOT / "tools" / "families",
)


def _owners() -> list[Path]:
    return sorted(
        path for path in MODELS_ROOT.iterdir() if path.is_dir() and (path / "MODEL.toml").is_file()
    )


def _manifest(owner: Path) -> dict[str, object]:
    with (owner / "MODEL.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_only_the_unified_model_owner_root_exists() -> None:
    assert MODELS_ROOT.is_dir()
    assert not [path for path in LEGACY_ROOTS if path.exists()]
    assert _owners()


def test_each_owner_contains_its_builder_runtime_and_tests() -> None:
    for owner in _owners():
        manifest = _manifest(owner)
        assert manifest.get("id") == owner.name
        assert list(owner.rglob("MODEL.toml")) == [owner / "MODEL.toml"]
        assert (owner / "model.py").is_file()
        assert (owner / "runtime" / "plugin.cpp").is_file()

        e2e_tests = sorted((owner / "tests").glob("test_*_e2e.py"))
        assert len(e2e_tests) == 1, owner.name


def test_descriptor_paths_resolve_inside_their_owner() -> None:
    for owner in _owners():
        manifest = _manifest(owner)

        declared_manifests = {
            owner / str(relative) for relative in manifest.get("test_manifests", [])
        }
        discovered_manifests = set((owner / "tests" / "manifests").glob("*.json"))
        assert declared_manifests == discovered_manifests, owner.name

        for entry in manifest.get("runtime_plugins", []):
            source, separator, symbol = str(entry).partition("|")
            assert separator and source and symbol, (owner.name, entry)
            assert (owner / "runtime" / source).is_file(), (owner.name, source)

        for entry in manifest.get("runtime_config_schemas", []):
            source, separator, symbol = str(entry).partition("|")
            assert separator and source and symbol, (owner.name, entry)
            assert (owner / "runtime" / source).is_file(), (owner.name, source)

        runtime_cmake = owner / "runtime" / "CMakeLists.txt"
        assert runtime_cmake.is_file(), owner.name
        build = runtime_cmake.read_text(encoding="utf-8")
        assert f"add_library(trtmc_model_{owner.name} SHARED" in build, owner.name
        for source in (owner / "runtime").iterdir():
            if source.suffix in {".cpp", ".cu"}:
                assert source.name in build, (owner.name, source.name)
        for source in (owner / "tests" / "cpp").glob("test_*.cpp"):
            assert f"trtmc_add_test({source.stem}" in build, (owner.name, source.name)


def test_descriptors_have_no_legacy_layout_or_runtime_identity() -> None:
    for owner in _owners():
        path = owner / "MODEL.toml"
        source = path.read_text(encoding="utf-8")
        manifest = tomllib.loads(source)
        assert "runtime_library" not in manifest, owner.name
        assert "legacy_runtime_strategy_aliases" not in manifest, owner.name
        assert "runtime_tests" not in source, owner.name
        assert "runtime_link_libraries" not in source, owner.name
        assert "gnu_warning_suppressed_sources" not in source, owner.name
        assert "families/" not in source, owner.name


def test_optional_kernel_projects_are_owner_local_and_discovered_generically() -> None:
    cmake = (REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    kernel_projects = sorted(MODELS_ROOT.glob("*/runtime/kernels/CMakeLists.txt"))

    assert kernel_projects
    assert '"${TRTMC_MODELS_ROOT}/*/runtime/kernels/CMakeLists.txt"' not in cmake
    assert "src/runtime/domains/diffusion/kernels" not in cmake
    for project in kernel_projects:
        assert project.parents[2] in _owners()
        owner_build = project.parent.parent / "CMakeLists.txt"
        assert "add_subdirectory(kernels)" in owner_build.read_text(encoding="utf-8")


def test_root_cmake_only_discovers_aggregates_and_installs_model_targets() -> None:
    cmake = (REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "add_subdirectory(" in cmake
    assert "add_dependencies(trtmc_model_plugins trtmc_model_${_trtmc_model})" in cmake
    assert "install(TARGETS trtmc_model_${_trtmc_model}" in cmake
    assert '"${_trtmc_model_runtime_root}/*.cpp"' not in cmake
    assert "TRTMC_MODEL_${_trtmc_model_var}_LINK_LIBRARIES" not in cmake
    assert "trtmc_add_model_manifest_tests" not in cmake
