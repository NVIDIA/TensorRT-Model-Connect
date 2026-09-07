# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-only contracts for the standalone LeRobot ACT example."""

from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
EXAMPLE = REPO / "examples/models/lerobot_act/recorded_control"


def test_documented_runtime_root_contains_the_family_and_backend_dsos() -> None:
    cmake = (EXAMPLE / "CMakeLists.txt").read_text(encoding="utf-8")
    readme = (EXAMPLE / "README.md").read_text(encoding="utf-8")
    source = (EXAMPLE / "main.cpp").read_text(encoding="utf-8")

    runtime_dir = "${CMAKE_BINARY_DIR}/trtmc"
    assert f'add_subdirectory("${{TRTMC_SOURCE_DIR}}" "{runtime_dir}")' in cmake
    family_output = cmake.split("set_target_properties(trtmc_model_lerobot_act PROPERTIES", 1)[
        1
    ].split(")", 1)[0]
    assert f'LIBRARY_OUTPUT_DIRECTORY "{runtime_dir}"' in family_output
    dependencies = cmake.split("add_dependencies(trtmc_lerobot_act_recorded_control", 1)[1].split(
        ")", 1
    )[0]
    assert "trtmc_backend_trt" in dependencies
    assert "trtmc_model_lerobot_act" in dependencies

    assert "--runtime-root /tmp/lerobot-act-example/trtmc" in readme
    assert "load_task(options.bundle, options.runtime_root)" in source
