# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

from conan import ConanFile
from conan.errors import ConanException
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy


def _set_runpath(path: Path, runpath: str) -> None:
    try:
        subprocess.run(
            ["patchelf", "--set-rpath", runpath, str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ConanException(f"cannot set RUNPATH on {path.name}: {error}") from error


def _needed_libraries(path: Path) -> tuple[str, ...]:
    try:
        result = subprocess.run(
            ["patchelf", "--print-needed", str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ConanException(f"cannot read dependencies from {path.name}: {error}") from error
    return tuple(result.stdout.splitlines())


def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class TensorRTModelConnectConan(ConanFile):
    name = "tensorrt-model-connect"
    version = "0.1.0"
    package_type = "application"

    settings = "os", "compiler", "build_type", "arch"

    def _windows(self) -> bool:
        return str(self.settings.os) == "Windows"

    def layout(self) -> None:
        cmake_layout(self)
        # CMakeToolchain derives install directories from the package layout.
        self.cpp.package.libdirs = ["bin"]

    def requirements(self) -> None:
        # Linux images provide nlohmann-json3-dev; MSVC builds take it from Conan.
        if self._windows():
            self.requires("nlohmann_json/3.11.3")

    def generate(self) -> None:
        toolchain = CMakeToolchain(self)
        toolchain.cache_variables["TRTMC_BUILD_TESTS"] = False
        for name in (
            "TRT_ROOT",
            "CMAKE_CUDA_ARCHITECTURES",
            "TRTMC_FAMILIES",
        ):
            value = os.environ.get(name)
            if value:
                toolchain.cache_variables[name] = value
        if self._windows():
            # The Windows port covers the native runtime, CLI, and model
            # families; the server, BYOK bridge, and examples stay ELF-only.
            for option in ("TRTMC_BUILD_SERVER", "TRTMC_ENABLE_BYOK", "TRTMC_BUILD_EXAMPLES"):
                toolchain.cache_variables[option] = False
            CMakeDeps(self).generate()
        toolchain.generate()

    def build(self) -> None:
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def _package_windows(self) -> None:
        source = Path(self.source_folder)
        build = Path(self.build_folder)
        module_bin = Path(self.package_folder) / "tensorrt_model_connect" / "bin"
        # Windows has no RUNPATH: the executable, runtime DLLs, backend, and
        # family DLLs share one directory, which is also the runtime root.
        copy(self, "trtmc.exe", src=str(build), dst=str(module_bin), keep_path=False)
        copy(self, "*.dll", src=str(build), dst=str(module_bin), keep_path=False)
        selected = [name for name in os.environ.get("TRTMC_FAMILIES", "").split(";") if name]
        expected = set(selected) or {
            path.parent.name for path in (source / "families").glob("*/model.py")
        }
        packaged = {
            path.stem.removeprefix("trtmc_model_") for path in module_bin.glob("trtmc_model_*.dll")
        }
        required = (
            "trtmc.exe",
            "trtmc_core.dll",
            "trtmc_runtime.dll",
            "trtmc_c.dll",
            "trtmc_backend_trt.dll",
        )
        if not all((module_bin / name).is_file() for name in required):
            raise ConanException("native Windows runtime package is incomplete")
        if packaged != expected:
            raise ConanException(
                f"family DLL set does not match: missing={sorted(expected - packaged)}, "
                f"extra={sorted(packaged - expected)}"
            )

    def package(self) -> None:
        if self._windows():
            self._package_windows()
            return
        source = Path(self.source_folder)
        build = Path(self.build_folder)
        package = Path(self.package_folder)
        module_bin = package / "tensorrt_model_connect" / "bin"

        subprocess.run(
            [
                "cmake",
                "--install",
                str(build),
                "--prefix",
                str(module_bin.parent),
                "--component",
                "sdk",
            ],
            check=True,
        )
        copy(self, "trtmc", src=str(build), dst=str(module_bin), keep_path=False)
        copy(self, "trtmc-server", src=str(build), dst=str(module_bin), keep_path=False)
        for library in ("libtrtmc_core.so", "libtrtmc_runtime.so"):
            copy(self, library, src=str(build), dst=str(module_bin), keep_path=False)
        copy(
            self,
            "libtrtmc_backend_trt*.so",
            src=str(build),
            dst=str(module_bin),
            keep_path=False,
        )
        copy(
            self,
            "libtrtmc_byok_tvm_ffi.so",
            src=str(build),
            dst=str(module_bin),
            keep_path=False,
        )
        for executable in ("trtmc_benchmark_worker", "trtmc_dataset_benchmark"):
            copy(self, executable, src=str(build), dst=str(module_bin), keep_path=False)
        copy(
            self,
            "libtrtmc_model_*.so",
            src=str(build),
            dst=str(module_bin),
            keep_path=False,
        )
        copy(self, "libtrtmc_cli_*.so", src=str(build), dst=str(module_bin), keep_path=False)
        expected_cli_libraries = set()
        for declaration in sorted((source / "families").glob("*/cli.json")):
            copy(
                self,
                declaration.name,
                src=str(declaration.parent),
                dst=str(module_bin / "families" / declaration.parent.name),
                keep_path=False,
            )
            if any(
                command["executor"] == "native"
                for command in json.loads(declaration.read_text(encoding="utf-8"))["commands"]
            ):
                expected_cli_libraries.add(f"libtrtmc_cli_{declaration.parent.name}.so")
        if {path.name for path in module_bin.glob("libtrtmc_cli_*.so")} != expected_cli_libraries:
            raise ConanException("family CLI adapter set does not match CLI declarations")
        catalog = package / "trtmc_benchmark" / "_catalog"
        source_suffixes = {".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".py", ".pyc"}
        for asset in sorted((source / "families").glob("*/tests/**/*")):
            if (
                not asset.is_file()
                or asset.suffix in source_suffixes
                or "__pycache__" in asset.parts
            ):
                continue
            family = asset.relative_to(source / "families").parts[0]
            relative = asset.relative_to(source / "families" / family / "tests")
            destination = catalog / family / "tests" / relative.parent
            copy(
                self,
                asset.name,
                src=str(asset.parent),
                dst=str(destination),
                keep_path=False,
            )

        expected = {path.parent.name for path in (source / "families").glob("*/model.py")}
        if not expected:
            raise ConanException("repository has no model families")
        packaged = {
            path.name.removeprefix("libtrtmc_model_").removesuffix(".so")
            for path in module_bin.glob("libtrtmc_model_*.so")
        }
        if packaged != expected:
            missing = sorted(expected - packaged)
            extra = sorted(packaged - expected)
            raise ConanException(
                f"family DSO set does not match family builders: missing={missing}, extra={extra}"
            )

        native = module_bin / "trtmc"
        native_server = module_bin / "trtmc-server"
        shared_runtime = [
            module_bin / library
            for library in (
                "libtrtmc_core.so",
                "libtrtmc_runtime.so",
                "libtrtmc_c.so",
                "libtrtmc_c.so.1",
            )
        ]
        backend = module_bin / "libtrtmc_backend_trt.so"
        backends = sorted(module_bin.glob("libtrtmc_backend_trt*.so"))
        byok = module_bin / "libtrtmc_byok_tvm_ffi.so"
        benchmark_worker = module_bin / "trtmc_benchmark_worker"
        dataset_benchmark = module_bin / "trtmc_dataset_benchmark"
        if (
            not native.is_file()
            or not native_server.is_file()
            or not all(library.is_file() for library in shared_runtime)
            or not backend.is_file()
            or not byok.is_file()
            or not benchmark_worker.is_file()
            or not dataset_benchmark.is_file()
        ):
            raise ConanException("native runtime package is incomplete")

        for executable in (native, native_server, benchmark_worker, dataset_benchmark):
            _make_executable(executable)
            _set_runpath(executable, "$ORIGIN")
        for library in shared_runtime:
            _set_runpath(library, "$ORIGIN:/usr/local/cuda/lib64")
        _set_runpath(
            byok,
            "$ORIGIN:$ORIGIN/../../tensorrt_libs:$ORIGIN/../../tvm_ffi/lib:/usr/local/cuda/lib64",
        )
        for library in (
            *backends,
            *module_bin.glob("libtrtmc_model_*.so"),
            *module_bin.glob("libtrtmc_cli_*.so"),
        ):
            runpaths = ["$ORIGIN", "$ORIGIN/../../tensorrt_libs", "/usr/local/cuda/lib64"]
            if any(
                dependency.startswith(("libtorch", "libc10"))
                for dependency in _needed_libraries(library)
            ):
                runpaths.append("$ORIGIN/../../torch/lib")
            _set_runpath(
                library,
                ":".join(runpaths),
            )
