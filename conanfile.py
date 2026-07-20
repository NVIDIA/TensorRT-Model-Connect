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
)


def _resolved_packaging_path(
    path: Path,
    root: Path,
    *,
    description: str,
    directory: bool,
) -> Path:
    """Resolve one packaging input without accepting links or root escapes."""

    if path.is_symlink():
        raise ConanException(f"{description} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ConanException(f"Unable to resolve {description} {path}: {exc}") from exc
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ConanException(
            f"{description} resolves outside its packaging root {root}: {path} -> {resolved}"
        ) from exc
    expected_type = resolved.is_dir() if directory else resolved.is_file()
    if not expected_type:
        kind = "directory" if directory else "regular file"
        raise ConanException(f"{description} must be a {kind}: {path}")
    return resolved


def _model_owned_adapters(source_folder: str | Path) -> tuple[tuple[str, Path, Path, Path], ...]:
    """Locate model-owned adapter source trees without interpreting their contents."""

    source = Path(source_folder)
    families_root = source / "python" / "tensorrt_model_connect" / "families"
    runtime_root = source / "src" / "runtime" / "models"
    adapters: list[tuple[str, Path, Path, Path]] = []
    if not families_root.is_dir():
        return ()

    canonical_manifests = sorted(
        families_root.glob("*/*_adapter/*/IMPLEMENTATION.toml"),
        key=lambda path: str(path),
    )
    if not canonical_manifests:
        return ()

    try:
        source_root = source.resolve(strict=True)
        families_root_resolved = families_root.resolve(strict=True)
        families_root_resolved.relative_to(source_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConanException(
            f"Model family packaging root escapes or cannot be resolved below {source}: "
            f"{families_root}"
        ) from exc

    for manifest in canonical_manifests:
        builder = manifest.parent
        relative = builder.relative_to(families_root)
        family = relative.parts[0]
        adapter_profile = Path(*relative.parts[1:])
        provider = builder.parent
        _resolved_packaging_path(
            provider.parent,
            families_root_resolved,
            description="Model-owned build adapter family directory",
            directory=True,
        )
        _resolved_packaging_path(
            provider,
            families_root_resolved,
            description="Model-owned build adapter provider directory",
            directory=True,
        )
        builder_resolved = _resolved_packaging_path(
            builder,
            families_root_resolved,
            description="Model-owned build adapter profile directory",
            directory=True,
        )
        _resolved_packaging_path(
            manifest,
            families_root_resolved,
            description="Model-owned build adapter manifest",
            directory=False,
        )

        runtime = runtime_root / family / adapter_profile
        if not runtime.is_dir():
            raise ConanException(
                "Model-owned build adapter "
                f"{family}/{adapter_profile.as_posix()} has no matching runtime source directory: "
                f"{runtime}"
            )
        try:
            runtime_root_resolved = runtime_root.resolve(strict=True)
            runtime_root_resolved.relative_to(source_root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ConanException(
                f"Model-owned runtime packaging root escapes or cannot be resolved below {source}: "
                f"{runtime_root}"
            ) from exc
        runtime_family = runtime_root / family
        runtime_provider = runtime_family / adapter_profile.parts[0]
        for runtime_directory, description in (
            (runtime_family, "Model-owned runtime family directory"),
            (runtime_provider, "Model-owned runtime provider directory"),
        ):
            _resolved_packaging_path(
                runtime_directory,
                runtime_root_resolved,
                description=description,
                directory=True,
            )
        runtime_resolved = _resolved_packaging_path(
            runtime,
            runtime_root_resolved,
            description="Model-owned runtime profile directory",
            directory=True,
        )
        adapters.append((family, adapter_profile, builder_resolved, runtime_resolved))
    return tuple(adapters)


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
        for family, adapter_profile, builder_source, runtime_source in _model_owned_adapters(
            self.source_folder
        ):
            packaged_adapter = package_module / "families" / family / adapter_profile
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

        private_sdk = package_module / "runtime_provider" / "_sdk" / "include"
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
