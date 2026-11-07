# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the production family selector with real CMake configuration."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.fixture
def family_project(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    shutil.copytree(Path(__file__).resolve().parents[2] / "cmake", source / "cmake")
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(family_selection LANGUAGES C)\n"
        "include(cmake/ModelFamilies.cmake)\n"
    )
    for name in ("alpha", "beta"):
        runtime = source / "families" / name / "runtime"
        runtime.mkdir(parents=True)
        (runtime / "owner.c").write_text("int owner(void) { return 0; }\n")
        (runtime / "CMakeLists.txt").write_text(
            ("find_package(SelectionDependency REQUIRED CONFIG)\n" if name == "beta" else "")
            + f"add_library(trtmc_model_{name} SHARED owner.c)\n"
            + f"add_library(trtmc_cli_{name} SHARED owner.c)\n"
        )
    return source


def configure(source, selection=None, *, dependency=False):
    command = ["cmake", "-S", str(source), "-B", str(source.parent / "build")]
    if selection is not None:
        command.append(f"-DTRTMC_FAMILIES={selection}")
    if dependency:
        package = source.parent / "dependency"
        package.mkdir()
        (package / "SelectionDependencyConfig.cmake").write_text(
            "set(SelectionDependency_FOUND TRUE)\n"
        )
        command.append(f"-DSelectionDependency_DIR={package}")
    return subprocess.run(command, text=True, capture_output=True, check=False)


def test_selected_family_does_not_configure_unavailable_dependency(family_project):
    result = configure(family_project, "alpha")
    assert result.returncode == 0, result.stdout + result.stderr
    build = family_project.parent / "build"
    subprocess.run(
        ["cmake", "--build", str(build), "--target", "trtmc_model_alpha", "trtmc_cli_alpha"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (build / "families/alpha/libtrtmc_model_alpha.so").is_file()
    assert (build / "families/alpha/libtrtmc_cli_alpha.so").is_file()
    assert not (build / "families/beta").exists()


@pytest.mark.parametrize("selection", [None, "", "beta", "alpha;beta"])
def test_required_dependency_is_not_silenced(family_project, selection):
    result = configure(family_project, selection)
    assert result.returncode != 0
    assert "SelectionDependency" in result.stderr


@pytest.mark.parametrize("selection", [None, "", "beta;alpha;alpha"])
def test_default_and_multiple_families_keep_all_owner_targets(family_project, selection):
    result = configure(family_project, selection, dependency=True)
    assert result.returncode == 0, result.stdout + result.stderr
    build = family_project.parent / "build"
    subprocess.run(["cmake", "--build", str(build)], check=True, capture_output=True, text=True)
    for family in ("alpha", "beta"):
        assert (build / f"families/{family}/libtrtmc_model_{family}.so").is_file()
        assert (build / f"families/{family}/libtrtmc_cli_{family}.so").is_file()


@pytest.mark.parametrize(
    "selection", ["unknown", "Alpha", "../alpha", "alpha/beta", "alpha;", ";alpha", "alpha;;beta"]
)
def test_invalid_family_selection_fails_before_configuring_owners(family_project, selection):
    result = configure(family_project, selection)
    assert result.returncode != 0
    assert "Unknown build family:" in result.stderr
    assert not (family_project.parent / "build/families").exists()
