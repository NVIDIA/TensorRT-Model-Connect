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
        wheel_data_scripts = (
            Path(self.package_folder)
            / f"{self.name.replace('-', '_')}-{self.version}.data"
            / "scripts"
        )
        copy(self, "trtmc", src=self.build_folder, dst=str(package_bin), keep_path=False)
        copy(self, "trtmc", src=self.build_folder, dst=str(wheel_data_scripts), keep_path=False)
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

        native = package_bin / "trtmc"
        installed_script = wheel_data_scripts / "trtmc"
        backends = sorted(package_bin.glob("libtrtmc_backend_trt*.so*"))
        model_plugins = sorted(package_bin.glob("libtrtmc_model_*.so*"))
        if not native.is_file():
            raise ConanException("TRTMC native executable was not staged into the wheel package")
        if not installed_script.is_file():
            raise ConanException("TRTMC native executable was not staged as the wheel script")
        if not backends:
            raise ConanException("TRTMC TensorRT backend DSO was not staged into the wheel package")
        if not model_plugins:
            raise ConanException("TRTMC model plugin DSOs were not staged into the wheel package")

        for executable in (native, installed_script):
            mode = executable.stat().st_mode
            executable.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
