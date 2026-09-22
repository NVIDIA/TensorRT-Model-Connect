# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

from conan import ConanFile
from conan.errors import ConanException
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
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


def _package_native_commands(build: Path, destinations: tuple[Path, ...]) -> None:
    """Ship root-level native commands, including family-owned trtmc-* binaries."""
    commands = [build / "trtmc"]
    commands.extend(
        path for path in sorted(build.glob("trtmc-*")) if not path.suffix and not path.is_dir()
    )
    for command in commands:
        try:
            mode = command.lstat().st_mode
            if not stat.S_ISREG(mode) or not mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
                raise ConanException(f"native command is not a regular executable: {command}")
            with command.open("rb") as source:
                if source.read(4) != b"\x7fELF":
                    raise ConanException(f"native command is not an ELF executable: {command}")
            for directory in destinations:
                directory.mkdir(parents=True, exist_ok=True)
                target = directory / command.name
                if target.exists() or target.is_symlink():
                    raise ConanException(f"duplicate native command destination: {target}")
                # Copy this exact build output; recursive basename matching can
                # otherwise replace it with a nested target of the same name.
                shutil.copy2(command, target)
                _make_executable(target)
                _set_runpath(target, "$ORIGIN")
        except OSError as error:
            raise ConanException(f"cannot package native command {command}: {error}") from error


class TensorRTModelConnectConan(ConanFile):
    name = "tensorrt-model-connect"
    version = "0.1.0"
    package_type = "application"

    settings = "os", "compiler", "build_type", "arch"

    def layout(self) -> None:
        cmake_layout(self)
        # CMakeToolchain derives install directories from the package layout.
        self.cpp.package.libdirs = ["bin"]

    def generate(self) -> None:
        toolchain = CMakeToolchain(self)
        toolchain.cache_variables["TRTMC_BUILD_TESTS"] = False
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
        _package_native_commands(build, (module_bin,))
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

        for executable in (benchmark_worker, dataset_benchmark):
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
