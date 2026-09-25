# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the transcription proof executable in the family GPU plan."""

from pathlib import Path
import shutil
import subprocess

import pytest

from tools.community_gpu_ci import family_plan
from families.parakeet_tdt.tests import test_e2e


def test_transcription_is_selected_with_its_pinned_checkpoint():
    plan = family_plan(Path(__file__).resolve().parents[3], "parakeet_tdt")
    assert "parakeet-tdt-0.6b-v3" in plan.testcases
    assert (
        "nvidia/parakeet-tdt-0.6b-v3",
        "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
    ) in plan.checkpoints


def test_transcription_consumer_is_built_outside_isolated_runtime(tmp_path, monkeypatch):
    if not shutil.which("cmake") or not shutil.which("cc"):
        pytest.skip("native build probe requires CMake and a C compiler")
    source = tmp_path / "source"
    source.mkdir()
    (source / "probe.c").write_text("int main(void) { return 0; }\n")
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.18)\nproject(probe C)\n"
        "add_executable(test_parakeet_tdt_sdk_cpp probe.c)\n"
        "set_target_properties(test_parakeet_tdt_sdk_cpp PROPERTIES "
        'RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")\n'
    )
    build_dir = tmp_path / "build"
    subprocess.run(["cmake", "-S", str(source), "-B", str(build_dir)], check=True)
    monkeypatch.setenv("TRTMC_NATIVE_BUILD_DIR", str(build_dir))
    binary = test_e2e._sdk_consumer_binary()
    assert binary.parent == build_dir
    subprocess.run([str(binary)], check=True)


def test_transcription_consumer_requires_native_build(monkeypatch):
    monkeypatch.delenv("TRTMC_NATIVE_BUILD_DIR", raising=False)
    with pytest.raises(AssertionError, match="TRTMC_NATIVE_BUILD_DIR"):
        test_e2e._sdk_consumer_binary()
