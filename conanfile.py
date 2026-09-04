# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

from conan import ConanFile
from conan.errors import ConanException
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import copy


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


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


def _make_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class TensorRTModelConnectConan(ConanFile):
    name = "tensorrt-model-connect"
    version = "0.1.0"
    package_type = "application"

    settings = "os", "compiler", "build_type", "arch"

    def layout(self) -> None:
        cmake_layout(self)

    def generate(self) -> None:
        toolchain = CMakeToolchain(self)
        toolchain.cache_variables["TRTMC_BUILD_TESTS"] = _enabled("TRTMC_CONAN_ENABLE_TEST_TARGETS")
        for name in (
            "TRT_ROOT",
            "CMAKE_CUDA_ARCHITECTURES",
        ):
            value = os.environ.get(name)
            if value:
                toolchain.cache_variables[name] = value
        toolchain.generate()

    def build(self) -> None:
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self) -> None:
        source = Path(self.source_folder)
        build = Path(self.build_folder)
        package = Path(self.package_folder)
        module_bin = package / "tensorrt_model_connect" / "bin"
        script_bin = package / f"{self.name.replace('-', '_')}-{self.version}.data" / "scripts"

        copy(self, "trtmc", src=str(build), dst=str(module_bin), keep_path=False)
        copy(self, "trtmc", src=str(build), dst=str(script_bin), keep_path=False)
        for destination in (module_bin, script_bin):
            for library in ("libtrtmc_core.so", "libtrtmc_runtime.so"):
                copy(
                    self,
                    library,
                    src=str(build),
                    dst=str(destination),
                    keep_path=False,
                )
        copy(
            self,
            "libtrtmc_backend_trt.so",
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
        if not expected or packaged != expected:
            missing = sorted(expected - packaged)
            extra = sorted(packaged - expected)
            raise ConanException(
                f"family DSO set does not match family builders: missing={missing}, extra={extra}"
            )

        native = module_bin / "trtmc"
        installed = script_bin / "trtmc"
        shared_runtime = [
            destination / library
            for destination in (module_bin, script_bin)
            for library in ("libtrtmc_core.so", "libtrtmc_runtime.so")
        ]
        backend = module_bin / "libtrtmc_backend_trt.so"
        byok = module_bin / "libtrtmc_byok_tvm_ffi.so"
        benchmark_worker = module_bin / "trtmc_benchmark_worker"
        dataset_benchmark = module_bin / "trtmc_dataset_benchmark"
        if (
            not native.is_file()
            or not installed.is_file()
            or not all(library.is_file() for library in shared_runtime)
            or not backend.is_file()
            or not byok.is_file()
            or not benchmark_worker.is_file()
            or not dataset_benchmark.is_file()
        ):
            raise ConanException("native runtime package is incomplete")

        for executable in (native, installed, benchmark_worker, dataset_benchmark):
            _make_executable(executable)
            _set_runpath(executable, "$ORIGIN")
        for library in shared_runtime:
            _set_runpath(library, "$ORIGIN:/usr/local/cuda/lib64")
        _set_runpath(
            byok,
            "$ORIGIN:$ORIGIN/../../tensorrt_libs:$ORIGIN/../../tvm_ffi/lib:"
            "/usr/local/cuda/lib64",
        )
        for library in (
            backend,
            *module_bin.glob("libtrtmc_model_*.so"),
        ):
            torch_runpath = (
                ":$ORIGIN/../../torch/lib"
                if library.name == "libtrtmc_model_sana_wm.so"
                else ""
            )
            _set_runpath(
                library,
                "$ORIGIN:$ORIGIN/../../tensorrt_libs:/usr/local/cuda/lib64" + torch_runpath,
            )
