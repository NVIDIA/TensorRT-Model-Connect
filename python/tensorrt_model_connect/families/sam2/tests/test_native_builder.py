# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only install and discovery contracts for the SAM2 native builder."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tensorrt_model_connect.families.sam2 import native_builder


REPOSITORY_ROOT = Path(__file__).resolve().parents[5]


def _write_executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def test_locator_accepts_only_the_package_owned_binary_by_default(tmp_path: Path) -> None:
    expected = _write_executable(tmp_path / "package/bin/sam2_native_builder")

    actual = native_builder.locate_native_builder(environ={}, package_root=tmp_path / "package")

    assert actual == expected


def test_locator_fails_when_opt_in_component_is_absent(tmp_path: Path) -> None:
    with pytest.raises(
        native_builder.Sam2NativeBuilderError,
        match=r"component 'sam2_native_builder'.*TRTMC_SAM2_NATIVE_BUILDER",
    ):
        native_builder.locate_native_builder(environ={}, package_root=tmp_path / "package")


def test_explicit_absolute_override_is_authoritative(tmp_path: Path) -> None:
    packaged = _write_executable(tmp_path / "package/bin/sam2_native_builder")
    configured = _write_executable(tmp_path / "configured/sam2_native_builder")

    actual = native_builder.locate_native_builder(
        environ={native_builder.NATIVE_BUILDER_ENV: str(configured)},
        package_root=packaged.parents[1],
    )

    assert actual == configured


def test_invalid_override_does_not_fall_back_to_packaged_binary(tmp_path: Path) -> None:
    packaged = _write_executable(tmp_path / "package/bin/sam2_native_builder")
    missing = tmp_path / "missing/sam2_native_builder"

    with pytest.raises(native_builder.Sam2NativeBuilderError, match=str(missing)):
        native_builder.locate_native_builder(
            environ={native_builder.NATIVE_BUILDER_ENV: str(missing)},
            package_root=packaged.parents[1],
        )


def test_locator_does_not_search_path(tmp_path: Path) -> None:
    path_binary = _write_executable(tmp_path / "path-bin/sam2_native_builder")

    with pytest.raises(native_builder.Sam2NativeBuilderError, match="installed Python package"):
        native_builder.locate_native_builder(
            environ={"PATH": str(path_binary.parent)},
            package_root=tmp_path / "package",
        )


def test_locator_rejects_relative_non_executable_and_symlink_overrides(tmp_path: Path) -> None:
    non_executable = tmp_path / "non-executable"
    non_executable.write_bytes(b"not executable")
    executable = _write_executable(tmp_path / "real-builder")
    symlink = tmp_path / "linked-builder"
    symlink.symlink_to(executable)

    with pytest.raises(native_builder.Sam2NativeBuilderError, match="must name an absolute path"):
        native_builder.locate_native_builder(
            environ={native_builder.NATIVE_BUILDER_ENV: "relative/builder"},
            package_root=tmp_path,
        )
    with pytest.raises(native_builder.Sam2NativeBuilderError, match="must name an absolute path"):
        native_builder.locate_native_builder(
            environ={native_builder.NATIVE_BUILDER_ENV: "~/builder"},
            package_root=tmp_path,
        )
    with pytest.raises(native_builder.Sam2NativeBuilderError, match="not executable"):
        native_builder.locate_native_builder(
            environ={native_builder.NATIVE_BUILDER_ENV: str(non_executable)},
            package_root=tmp_path,
        )
    with pytest.raises(native_builder.Sam2NativeBuilderError, match="must not be a symlink"):
        native_builder.locate_native_builder(
            environ={native_builder.NATIVE_BUILDER_ENV: str(symlink)},
            package_root=tmp_path,
        )


def test_root_cmake_keeps_sam2_native_builder_default_off_and_isolated() -> None:
    root_cmake = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    native_cmake = (REPOSITORY_ROOT / "cmake/sam2_native.cmake").read_text(encoding="utf-8")

    option = re.search(
        r'option\(TRTMC_BUILD_SAM2_NATIVE_BUILDER\s+"[^"]*"\s+OFF\)',
        root_cmake,
    )
    assert option is not None
    assert "if(TRTMC_BUILD_SAM2_NATIVE_BUILDER)" in root_cmake
    assert "_trtmc_install_sam2_native_builder(trtmc_sam2_native_builder)" in native_cmake


@pytest.mark.skipif(
    shutil.which("cmake") is None or shutil.which(os.environ.get("CXX", "c++")) is None,
    reason="CMake and a C++ compiler are required for the install-component contract",
)
def test_cmake_component_builds_and_installs_only_the_builder(tmp_path: Path) -> None:
    source = tmp_path / "source"
    build = tmp_path / "build"
    component_prefix = tmp_path / "component-prefix"
    unspecified_prefix = tmp_path / "unspecified-prefix"
    source.mkdir()
    (source / "main.cpp").write_text("int main() { return 0; }\n", encoding="utf-8")
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(sam2_native_install_contract LANGUAGES CXX)\n"
        "include(GNUInstallDirs)\n"
        'include("${TRTMC_SOURCE_DIR}/cmake/sam2_native.cmake")\n'
        "add_executable(trtmc_sam2_native_builder main.cpp)\n"
        "set_target_properties(trtmc_sam2_native_builder PROPERTIES "
        "OUTPUT_NAME sam2_native_builder)\n"
        "_trtmc_install_sam2_native_builder(trtmc_sam2_native_builder)\n"
        'file(WRITE "${CMAKE_BINARY_DIR}/sam2-install-metadata.txt" '
        '"${TRTMC_SAM2_NATIVE_BUILDER_INSTALL_COMPONENT}\\n" '
        '"${CPACK_COMPONENT_SAM2_NATIVE_BUILDER_DISPLAY_NAME}\\n" '
        '"${CPACK_COMPONENT_SAM2_NATIVE_BUILDER_DESCRIPTION}\\n")\n',
        encoding="utf-8",
    )

    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            f"-DTRTMC_SOURCE_DIR={REPOSITORY_ROOT}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cmake", "--build", str(build), "--target", "trtmc_sam2_native_builder"],
        check=True,
        capture_output=True,
        text=True,
    )
    component, display_name, description = (
        (build / "sam2-install-metadata.txt").read_text(encoding="utf-8").splitlines()
    )
    assert component == native_builder.NATIVE_BUILDER_INSTALL_COMPONENT
    assert display_name == "SAM2 TensorRT native-attention builder"
    assert description.startswith("Unqualified, opt-in SAM2 checkpoint-to-bundle builder")

    subprocess.run(
        [
            "cmake",
            "--install",
            str(build),
            "--prefix",
            str(component_prefix),
            "--component",
            native_builder.NATIVE_BUILDER_INSTALL_COMPONENT,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    installed = component_prefix / "bin/sam2_native_builder"
    assert installed.is_file()
    assert os.access(installed, os.X_OK)
    assert (build / "install_manifest_sam2_native_builder.txt").read_text(
        encoding="utf-8"
    ).splitlines() == [str(installed)]

    subprocess.run(
        [
            "cmake",
            "--install",
            str(build),
            "--prefix",
            str(unspecified_prefix),
            "--component",
            "Unspecified",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert not (unspecified_prefix / "bin/sam2_native_builder").exists()
