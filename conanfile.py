# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import sys
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

_PACKAGE_SITE_PACKAGES_RUNPATHS = (
    "$ORIGIN/../../tensorrt_libs",
    "$ORIGIN/../../tvm_ffi/lib",
    "$ORIGIN/../../nvidia/cudnn/lib",
    "$ORIGIN/../../nvidia/cu13/lib",
)
_TRT_PLUGIN_INSTALL_RUNPATH = ":".join(
    (
        "$ORIGIN",
        _PACKAGE_SITE_PACKAGES_RUNPATHS[0],
        _PACKAGE_SITE_PACKAGES_RUNPATHS[2],
        _PACKAGE_SITE_PACKAGES_RUNPATHS[3],
    )
)


def _rewrite_elf_runpath(path: Path, runpath: str) -> None:
    """Replace build-tree RUNPATH on a staged ELF with relocatable wheel paths."""

    with path.open("rb") as stream:
        elf_magic = stream.read(4)
    if elf_magic != b"\x7fELF":
        return
    patchelf = shutil.which("patchelf")
    if not patchelf:
        raise ConanException(f"patchelf is required to relocate packaged ELF: {path}")
    result = subprocess.run(
        [patchelf, "--set-rpath", runpath, str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise ConanException(
            f"failed to set relocatable RUNPATH on {path}: {detail}"
        )
    verified = subprocess.run(
        [patchelf, "--print-rpath", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if verified.returncode != 0 or verified.stdout.strip() != runpath:
        detail = (verified.stderr or verified.stdout).strip()
        raise ConanException(
            f"failed to verify relocatable RUNPATH on {path}: {detail}"
        )


def _wheel_script_dependency_runpath() -> str:
    tag = os.environ.get(
        "WHEEL_PYVER",
        f"py{sys.version_info.major}{sys.version_info.minor}",
    )
    match = re.fullmatch(r"py([0-9])([0-9]+)", tag)
    if not match:
        raise ConanException(f"cannot derive wheel site-packages path from WHEEL_PYVER={tag!r}")
    python_dir = f"python{match.group(1)}.{match.group(2)}"
    site_packages = f"$ORIGIN/../lib/{python_dir}/site-packages"
    return ":".join(
        (
            "$ORIGIN",
            f"{site_packages}/tvm_ffi/lib",
            f"{site_packages}/nvidia/cu13/lib",
        )
    )


def _model_owned_adapters(source_folder: str | Path) -> tuple[tuple[str, str, Path, Path], ...]:
    """Locate model-owned adapter source trees without interpreting their contents."""

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
        if not runtime.is_dir():
            raise ConanException(
                "Model-owned build adapter "
                f"{family}/{adapter} has no matching runtime source directory: {runtime}"
            )
        adapters.append((family, adapter, builder, runtime))
    return tuple(adapters)


def _stage_benchmark_catalog(recipe: ConanFile, source_folder: str | Path, package: Path) -> None:
    """Copy canonical E2E model descriptors into the installed Python package."""

    source = Path(source_folder) / "tests" / "e2e" / "models"
    descriptors = sorted(source.glob("*/MODEL.toml"))
    manifests = sorted(source.glob("*/manifests/*.json"))
    benchmark_assets = set(source.glob("*/data/Recording.wav"))
    if not descriptors or not manifests:
        raise ConanException(f"benchmark model catalog is empty or unavailable: {source}")
    missing_descriptors = [
        manifest for manifest in manifests if not (manifest.parent.parent / "MODEL.toml").is_file()
    ]
    if missing_descriptors:
        paths = ", ".join(str(path) for path in missing_descriptors)
        raise ConanException(f"benchmark manifests have no family MODEL.toml: {paths}")
    for manifest in manifests:
        try:
            raw = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConanException(f"cannot read benchmark manifest {manifest}: {exc}") from exc
        references = [("fp8_scales", raw.get("fp8_scales"))]
        for index, testcase in enumerate(raw.get("testcases", [])):
            if not isinstance(testcase, dict):
                continue
            for field in ("test_image", "prompt_file"):
                if field in testcase:
                    references.append((f"testcases[{index}].{field}", testcase[field]))
        for field, declared in references:
            if declared is None:
                continue
            if not isinstance(declared, str) or not declared.strip():
                raise ConanException(f"{field} in benchmark manifest {manifest} must be a path")
            family = manifest.parent.parent.resolve()
            declared_path = Path(declared)
            asset = (family / declared_path).resolve()
            source_prefix = Path("tests/e2e/models") / family.name
            if not asset.is_file() and declared_path.is_relative_to(source_prefix):
                asset = (family / declared_path.relative_to(source_prefix)).resolve()
            if not asset.is_relative_to(family) or not asset.is_file():
                raise ConanException(
                    f"{field} in benchmark manifest {manifest} is missing or outside {family}: "
                    f"{asset}"
                )
            benchmark_assets.add(asset)
    benchmark_assets = sorted(benchmark_assets)

    destination = package / "benchmark" / "_catalog"
    for source_path in (*descriptors, *manifests, *benchmark_assets):
        relative = source_path.relative_to(source)
        copy(
            recipe,
            source_path.name,
            src=str(source_path.parent),
            dst=str(destination / relative.parent),
            keep_path=False,
        )

    packaged_descriptors = sorted(destination.glob("*/MODEL.toml"))
    packaged_manifests = sorted(destination.glob("*/manifests/*.json"))
    missing_assets = [
        source_path
        for source_path in benchmark_assets
        if not (destination / source_path.relative_to(source)).is_file()
    ]
    if (
        len(packaged_descriptors) != len(descriptors)
        or len(packaged_manifests) != len(manifests)
        or missing_assets
    ):
        raise ConanException(
            "benchmark model catalog staging is incomplete: "
            f"descriptors={len(packaged_descriptors)}/{len(descriptors)}, "
            f"manifests={len(packaged_manifests)}/{len(manifests)}, "
            f"assets={len(benchmark_assets) - len(missing_assets)}/{len(benchmark_assets)}"
        )


def _set_wheel_python_shebang(script: Path) -> None:
    """Mark a wheel data script for installer-specific interpreter rewriting."""

    try:
        source = script.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConanException(f"cannot read wheel script {script}: {exc}") from exc
    first_line, separator, body = source.partition("\n")
    if not separator or not first_line.startswith("#!"):
        raise ConanException(f"wheel script has no executable shebang: {script}")
    script.write_text(f"#!python\n{body}", encoding="utf-8")


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
        toolchain.cache_variables["TRTMC_BUILD_BENCHMARKS"] = True
        toolchain.cache_variables["TRTMC_ENABLE_LIBTORCH_MULTINOMIAL"] = False
        toolchain.cache_variables[
            "TRTMC_REQUIRE_DYNAMIC_MEMORY_CALIBRATOR_NVML"
        ] = True

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
        _stage_benchmark_catalog(self, self.source_folder, package_module)
        copy(self, "trtmc", src=self.build_folder, dst=str(package_bin), keep_path=False)
        copy(self, "trtmc", src=self.build_folder, dst=str(wheel_data_scripts), keep_path=False)
        copy(
            self,
            "trtmc_benchmark_worker",
            src=self.build_folder,
            dst=str(package_bin),
            keep_path=False,
        )
        package_internal = package_bin / ".trtmc-internal"
        script_internal = wheel_data_scripts / ".trtmc-internal"
        for destination in (package_internal, script_internal):
            copy(
                self,
                "trtmc_dynamic_memory_qualify",
                src=self.build_folder,
                dst=str(destination),
                keep_path=False,
            )
        copy(
            self,
            "trtmc-bench",
            src=str(Path(self.source_folder) / "scripts"),
            dst=str(wheel_data_scripts),
            keep_path=False,
        )
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
        copy(
            self,
            "libtrtmc_trt_plugins.so*",
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
        for family, adapter, builder_source, runtime_source in _model_owned_adapters(
            self.source_folder
        ):
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
        benchmark_worker = package_bin / "trtmc_benchmark_worker"
        package_calibrator = package_internal / "trtmc_dynamic_memory_qualify"
        script_calibrator = script_internal / "trtmc_dynamic_memory_qualify"
        benchmark_script = wheel_data_scripts / "trtmc-bench"
        package_cores = sorted(package_bin.glob("libtrtmc_core.so*"))
        script_cores = sorted(wheel_data_scripts.glob("libtrtmc_core.so*"))
        backends = sorted(package_bin.glob("libtrtmc_backend_trt*.so*"))
        trt_plugins = sorted(package_bin.glob("libtrtmc_trt_plugins.so*"))
        model_plugins = sorted(package_bin.glob("libtrtmc_model_*.so*"))
        if not native.is_file():
            raise ConanException("TRTMC native executable was not staged into the wheel package")
        if not installed_script.is_file():
            raise ConanException("TRTMC native executable was not staged as the wheel script")
        if not benchmark_worker.is_file():
            raise ConanException("TRTMC benchmark worker was not staged into the wheel package")
        if not package_calibrator.is_file() or not script_calibrator.is_file():
            raise ConanException(
                "TRTMC internal dynamic-memory calibrator was not staged beside both native CLIs"
            )
        if not benchmark_script.is_file():
            raise ConanException("trtmc-bench was not staged as the wheel script")
        _set_wheel_python_shebang(benchmark_script)
        if not package_cores:
            raise ConanException("TRTMC core DSO was not staged into the wheel package")
        if not script_cores:
            raise ConanException("TRTMC core DSO was not staged beside the wheel script")
        if not backends:
            raise ConanException("TRTMC TensorRT backend DSO was not staged into the wheel package")
        if not trt_plugins:
            raise ConanException(
                "TRTMC common TensorRT plugin DSO was not staged into the wheel package"
            )
        if not model_plugins:
            raise ConanException("TRTMC model plugin DSOs were not staged into the wheel package")

        package_core_runpath = ":".join(
            (
                "$ORIGIN",
                _PACKAGE_SITE_PACKAGES_RUNPATHS[1],
                _PACKAGE_SITE_PACKAGES_RUNPATHS[3],
            )
        )
        backend_runpath = ":".join(("$ORIGIN", *_PACKAGE_SITE_PACKAGES_RUNPATHS))
        model_plugin_runpath = ":".join(
            ("$ORIGIN", _PACKAGE_SITE_PACKAGES_RUNPATHS[3])
        )
        for executable in (native, installed_script, benchmark_worker):
            _rewrite_elf_runpath(executable, "$ORIGIN")
        _rewrite_elf_runpath(package_calibrator, "$ORIGIN/..")
        _rewrite_elf_runpath(script_calibrator, "$ORIGIN/..")
        for core in package_cores:
            _rewrite_elf_runpath(core, package_core_runpath)
        for core in script_cores:
            _rewrite_elf_runpath(core, _wheel_script_dependency_runpath())
        for backend in backends:
            _rewrite_elf_runpath(backend, backend_runpath)
        for trt_plugin in trt_plugins:
            _rewrite_elf_runpath(trt_plugin, _TRT_PLUGIN_INSTALL_RUNPATH)
        for model_plugin in model_plugins:
            _rewrite_elf_runpath(model_plugin, model_plugin_runpath)

        for executable in (
            native,
            installed_script,
            benchmark_worker,
            package_calibrator,
            script_calibrator,
            benchmark_script,
        ):
            mode = executable.stat().st_mode
            executable.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
