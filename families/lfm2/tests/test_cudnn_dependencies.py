# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of native SDK selection, staging and legacy dependency isolation."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


HELPER = Path(__file__).resolve().parents[1] / "runtime/cudnn.cmake"
LIBRARIES = (
    "libcudnn.so.9",
    "libcudnn_graph.so.9",
    "libcudnn_heuristic.so.9",
    "libcudnn_engines_precompiled.so.9",
    "libcudnn_engines_runtime_compiled.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_cnn.so.9",
    "libcudnn_adv.so.9",
)


@pytest.fixture
def sdk(tmp_path):
    if not shutil.which("cmake") or not shutil.which("cc"):
        pytest.skip("native dependency checks require cmake and a C compiler")

    def create(*, version=92000, cuda=13000, libdir="lib"):
        root = tmp_path / f"sdk-{version}-{cuda}-{libdir}"
        include = root / "include"
        include.mkdir(parents=True)
        (include / "cudnn.h").write_text("/* SDK fixture */\n")
        (include / "cudnn_graph.h").write_text("/* SDK fixture */\n")
        (include / "cudnn_version.h").write_text(
            "#define CUDNN_MAJOR 9\n#define CUDNN_MINOR 20\n#define CUDNN_PATCHLEVEL 0\n"
        )
        libs = root / libdir
        libs.mkdir()
        source = root / "version.c"
        source.write_text(
            "#include <stddef.h>\n"
            f"size_t cudnnGetVersion(void) {{ return {version}; }}\n"
            f"size_t cudnnGetCudartVersion(void) {{ return {cuda}; }}\n"
        )
        subprocess.run(
            ["cc", "-shared", "-fPIC", str(source), "-o", str(libs / LIBRARIES[0])],
            check=True,
            capture_output=True,
        )
        for name in LIBRARIES[1:]:
            (libs / name).write_bytes(("fixture:" + name).encode())
        (root / "LICENSE.txt").write_text("Complete fixture license.\n")
        return root

    return create


def stage(root, output, *extra):
    return subprocess.run(
        [
            "cmake",
            f"-DLFM2_CUDNN_STAGE_ROOT={output}",
            f"-DLFM2_CUDNN_ROOT={root}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            *extra,
            "-P",
            str(HELPER),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("libdir", ["lib", "lib64"])
def test_complete_sdk_stages_exact_library_and_license_bytes(sdk, tmp_path, libdir):
    root = sdk(libdir=libdir)
    output = tmp_path / "stage"
    result = stage(root, output)
    assert result.returncode == 0, result.stdout + result.stderr
    for name in LIBRARIES:
        assert (output / name).read_bytes() == (root / libdir / name).read_bytes()
        assert not (output / name).is_symlink()
    assert (output / "lfm2-cudnn-LICENSE.txt").read_bytes() == (root / "LICENSE.txt").read_bytes()


@pytest.mark.parametrize("version,cuda", [(91900, 13000), (92000, 12090)])
def test_selected_library_version_and_cuda_major_must_match(sdk, tmp_path, version, cuda):
    result = stage(sdk(version=version, cuda=cuda), tmp_path / "stage")
    assert result.returncode != 0
    assert "must be 9.20.0 for CUDA 13" in result.stderr
    assert not (tmp_path / "stage").exists()


def test_partial_sdk_does_not_fall_back_to_another_prefix(sdk, tmp_path):
    root = sdk()
    other = sdk(libdir="lib64")
    (root / "lib/libcudnn_heuristic.so.9").unlink()
    result = stage(root, tmp_path / "stage", f"-DCMAKE_PREFIX_PATH={other}")
    assert result.returncode != 0
    assert "missing libcudnn_heuristic.so.9" in result.stderr


def test_sdk_library_cannot_escape_selected_prefix(sdk, tmp_path):
    root = sdk()
    library = root / "lib/libcudnn_ops.so.9"
    outside = tmp_path / library.name
    library.rename(outside)
    library.symlink_to(outside)
    result = stage(root, tmp_path / "stage")
    assert result.returncode != 0
    assert "library escapes the selected SDK" in result.stderr


def test_header_library_version_mismatch_is_rejected(sdk, tmp_path):
    root = sdk()
    (root / "include/cudnn_version.h").write_text(
        "#define CUDNN_MAJOR 9\n#define CUDNN_MINOR 19\n#define CUDNN_PATCHLEVEL 0\n"
    )
    result = stage(root, tmp_path / "stage")
    assert result.returncode != 0
    assert "requires 9.20.0 headers" in result.stderr


def test_explicit_missing_license_does_not_fall_back(sdk, tmp_path):
    result = stage(sdk(), tmp_path / "stage", f"-DLFM2_CUDNN_LICENSE_FILE={tmp_path / 'absent'}")
    assert result.returncode != 0
    assert "complete license" in result.stderr


@pytest.mark.parametrize(
    "bridge,cuda,enable,success",
    [
        (False, 13, True, True),
        (True, 12, None, True),
        (True, 12, True, False),
    ],
)
def test_disabled_or_legacy_configuration_does_not_acquire_dependencies(
    tmp_path, bridge, cuda, enable, success
):
    if not shutil.which("cmake"):
        pytest.skip("cmake is required")
    source = tmp_path / "source"
    source.mkdir()
    declarations = "add_library(trtmc_byok_tvm_ffi INTERFACE)\n" if bridge else ""
    declarations += f"set(CUDAToolkit_VERSION_MAJOR {cuda})\n"
    if enable is not None:
        declarations += f'set(LFM2_ENABLE_CUDNN_SDPA {"ON" if enable else "OFF"} CACHE BOOL "")\n'
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.20)\nproject(dependency_scope NONE)\n"
        + declarations
        + f'set(LFM2_CUDNN_ROOT "{tmp_path / "absent-sdk"}")\n'
        + f'include("{HELPER}")\n'
        + "if(TARGET lfm2_cudnn OR TARGET lfm2_cudnn_frontend_source)\n"
        + '  message(FATAL_ERROR "Disabled path created dependency targets")\nendif()\n'
    )
    result = subprocess.run(
        ["cmake", "-S", str(source), "-B", str(tmp_path / "build")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is success, result.stdout + result.stderr
    assert not (tmp_path / "build/_deps").exists()
    if not success:
        assert "requires Linux and CUDA 13" in result.stderr
