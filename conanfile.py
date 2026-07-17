# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import stat
from pathlib import Path

from conan import ConanFile
from conan.errors import ConanException
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


_ADAPTER_PACKAGE_EXCLUDES = (
    "__pycache__/*",
    "**/__pycache__/**",
    "*.pyc",
    ".runtime-build/*",
    "**/.runtime-build/**",
    "dependencies/*",
    "**/dependencies/**",
    "tests/*",
    "**/tests/**",
    "README.md",
)

_FORBIDDEN_ADAPTER_DIRECTORIES = frozenset(
    {
        ".runtime-build",
        "artifacts",
        "build",
        "evidence",
        "qualification",
        "qualifications",
        "results",
        "tests",
    }
)


def _inert_adapter_files(root: Path) -> set[str]:
    """Return model-adapter source files that must survive package staging."""

    if not root.is_dir():
        return set()
    files: set[str] = set()
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root)
        if not candidate.is_file():
            continue
        if (
            "__pycache__" in relative.parts
            or ".runtime-build" in relative.parts
            or "dependencies" in relative.parts
            or "tests" in relative.parts
        ):
            continue
        if candidate.suffix == ".pyc" or candidate.name == "README.md":
            continue
        files.add(relative.as_posix())
    return files


def _validate_adapter_source(root: Path) -> None:
    """Reject non-source data that must never enter a model adapter package."""

    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root)
        if "dependencies" in relative.parts:
            continue
        if candidate.is_symlink():
            raise ConanException(f"Model-owned adapters must not contain symlinks: {candidate}")
        forbidden = sorted(_FORBIDDEN_ADAPTER_DIRECTORIES.intersection(relative.parts))
        if forbidden:
            raise ConanException(
                "Model-owned adapters must not contain evidence, test, or build directories: "
                f"{candidate} ({', '.join(forbidden)})"
            )
        if candidate.is_file() and _is_generated_runtime_payload(candidate):
            raise ConanException(
                f"Model-owned adapters must not contain generated runtime artifacts: {candidate}"
            )


def _model_owned_adapters(source_folder: str | Path) -> tuple[tuple[str, str, Path, Path], ...]:
    """Discover family-owned builder/runtime pairs by their source layout."""

    source = Path(source_folder)
    families_root = source / "python" / "tensorrt_model_connect" / "families"
    runtime_root = source / "src" / "runtime" / "models"
    adapters: list[tuple[str, str, Path, Path]] = []
    if not families_root.is_dir():
        return ()
    for manifest in sorted(families_root.glob("*/*/IMPLEMENTATION.toml")):
        builder = manifest.parent
        family = builder.parent.name
        adapter = builder.name
        runtime = runtime_root / family / adapter
        if manifest.is_symlink() or builder.is_symlink() or runtime.is_symlink():
            raise ConanException(f"Model-owned adapter paths must not be symbolic links: {builder}")
        if not runtime.is_dir():
            raise ConanException(
                "Model-owned optimized-runtime builder has no matching runtime source: "
                f"{builder} -> {runtime}"
            )
        adapters.append((family, adapter, builder, runtime))
    return tuple(adapters)


def _is_generated_runtime_payload(path: Path) -> bool:
    """Identify binary build outputs that never belong in an inert capsule."""

    name = path.name.lower()
    generated_suffixes = (
        ".so",
        ".dll",
        ".dylib",
        ".engine",
        ".onnx",
        ".plan",
        ".safetensors",
    )
    return any(name.endswith(suffix) or f"{suffix}." in name for suffix in generated_suffixes)


class TensorRTModelConnectConan(ConanFile):
    name = "tensorrt-model-connect"
    version = "0.1.0"
    package_type = "application"

    settings = "os", "compiler", "build_type", "arch"

    def requirements(self) -> None:
        self.requires("nlohmann_json/3.11.3")

    def layout(self) -> None:
        cmake_layout(self)

    def generate(self) -> None:
        deps = CMakeDeps(self)
        deps.generate()

        toolchain = CMakeToolchain(self)
        toolchain.cache_variables["TRTMC_BUILD_TESTS"] = _env_flag(
            "TRTMC_CONAN_ENABLE_TEST_TARGETS"
        )
        toolchain.cache_variables["TRTMC_BUILD_BENCHMARKS"] = False
        toolchain.cache_variables["TRTMC_ENABLE_LIBTORCH_MULTINOMIAL"] = False

        for name in (
            "TRTMC_TRT_INCLUDE_DIR",
            "TRTMC_TRT_LIBRARY",
            "TRTMC_CUDA_INCLUDE_DIR",
            "TRTMC_CUDART_LIBRARY",
        ):
            value = os.environ.get(name)
            if value:
                toolchain.cache_variables[name] = value

        toolchain.generate()

    def build(self) -> None:
        cmake = CMake(self)
        cmake.configure()
        targets = os.environ.get("TRTMC_CONAN_BUILD_TARGETS", "").split()
        if targets:
            for target in targets:
                cmake.build(target=target)
        else:
            cmake.build()
            cmake.build(target="trtmc_model_plugins")

    def package(self) -> None:
        package_bin = Path(self.package_folder) / "tensorrt_model_connect" / "bin"
        package_module = Path(self.package_folder) / "tensorrt_model_connect"
        wheel_data_scripts = (
            Path(self.package_folder)
            / f"{self.name.replace('-', '_')}-{self.version}.data"
            / "scripts"
        )
        copy(self, "trtmc", src=self.build_folder, dst=str(package_bin), keep_path=False)
        copy(self, "trtmc", src=self.build_folder, dst=str(wheel_data_scripts), keep_path=False)
        for destination in (package_bin, wheel_data_scripts):
            copy(
                self,
                "libtrtmc_core.so*",
                src=self.build_folder,
                dst=str(destination),
                keep_path=False,
            )
        copy(
            self,
            "libtrtmc_backend_trt*.so*",
            src=self.build_folder,
            dst=str(package_bin),
            keep_path=False,
        )
        for model_plugin in sorted(
            (Path(self.build_folder) / "models").rglob("libtrtmc_model_*.so*")
        ):
            copy(
                self,
                model_plugin.name,
                src=str(model_plugin.parent),
                dst=str(package_bin),
                keep_path=False,
            )

        # Each model family owns its build adapter and matching runtime source.
        # They remain inert package data until that family selects the adapter;
        # no downstream runtime is built or linked while packaging Model Connect.
        packaged_adapters: list[Path] = []
        for family, adapter, builder_source, runtime_source in _model_owned_adapters(
            self.source_folder
        ):
            _validate_adapter_source(builder_source)
            _validate_adapter_source(runtime_source)

            packaged_adapter = package_module / "families" / family / adapter
            copy(
                self,
                "*",
                src=str(builder_source),
                dst=str(packaged_adapter),
                excludes=_ADAPTER_PACKAGE_EXCLUDES,
            )
            copy(
                self,
                "*",
                src=str(runtime_source),
                dst=str(packaged_adapter / "runtime"),
                excludes=_ADAPTER_PACKAGE_EXCLUDES,
            )
            expected_files = _inert_adapter_files(builder_source) | {
                f"runtime/{relative}" for relative in _inert_adapter_files(runtime_source)
            }
            packaged_files = _inert_adapter_files(packaged_adapter)
            if packaged_files != expected_files:
                missing = sorted(expected_files - packaged_files)
                extra = sorted(packaged_files - expected_files)
                raise ConanException(
                    "Model-owned adapter package staging was incomplete: "
                    f"family={family}, adapter={adapter}, missing={missing}, extra={extra}"
                )
            if not (packaged_adapter / "IMPLEMENTATION.toml").is_file():
                raise ConanException(
                    "Model-owned adapter manifest was not staged into the package: "
                    f"{family}/{adapter}"
                )
            packaged_adapters.append(packaged_adapter)

        private_sdk = package_module / "optimized_runtime" / "_sdk" / "include"
        copy(
            self,
            "optimized_runtime_factory.h",
            src=str(Path(self.source_folder) / "src" / "runtime" / "providers"),
            dst=str(private_sdk / "runtime" / "providers"),
            keep_path=False,
        )
        copy(
            self,
            "pipeline.h",
            src=str(Path(self.source_folder) / "include" / "trtmc"),
            dst=str(private_sdk / "trtmc"),
            keep_path=False,
        )
        expected_private_sdk = {
            "runtime/providers/optimized_runtime_factory.h",
            "trtmc/pipeline.h",
        }
        packaged_private_sdk = {
            path.relative_to(private_sdk).as_posix()
            for path in private_sdk.rglob("*")
            if path.is_file()
        }
        if packaged_private_sdk != expected_private_sdk:
            raise ConanException(
                "The package-private optimized-runtime compile SDK was incomplete: "
                f"expected={sorted(expected_private_sdk)}, actual={sorted(packaged_private_sdk)}"
            )
        generated_adapter_payloads = [
            path
            for adapter_root in packaged_adapters
            for path in adapter_root.rglob("*")
            if path.is_file() and _is_generated_runtime_payload(path)
        ]
        if generated_adapter_payloads:
            raise ConanException(
                "Inert model-owned adapters contain generated runtime artifacts: "
                + ", ".join(str(path) for path in generated_adapter_payloads)
            )

        native = package_bin / "trtmc"
        installed_script = wheel_data_scripts / "trtmc"
        package_cores = sorted(package_bin.glob("libtrtmc_core.so*"))
        script_cores = sorted(wheel_data_scripts.glob("libtrtmc_core.so*"))
        backends = sorted(package_bin.glob("libtrtmc_backend_trt*.so*"))
        model_plugins = sorted(package_bin.glob("libtrtmc_model_*.so*"))
        if not native.is_file():
            raise ConanException("TRTMC native executable was not staged into the wheel package")
        if not installed_script.is_file():
            raise ConanException("TRTMC native executable was not staged as the wheel script")
        if not package_cores:
            raise ConanException("TRTMC core DSO was not staged into the wheel package")
        if not script_cores:
            raise ConanException("TRTMC core DSO was not staged beside the wheel script")
        if not backends:
            raise ConanException("TRTMC TensorRT backend DSO was not staged into the wheel package")
        if not model_plugins:
            raise ConanException("TRTMC model plugin DSOs were not staged into the wheel package")

        for executable in (native, installed_script):
            mode = executable.stat().st_mode
            executable.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
