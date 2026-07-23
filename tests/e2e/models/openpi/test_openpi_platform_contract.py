# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed checks for the qualified OpenPI deployment platform."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_CMAKE = REPO_ROOT / "src" / "runtime" / "models" / "openpi" / "model.cmake"
MODEL_OPTIONS = MODEL_CMAKE.with_name("model_options.cmake")


def test_native_runtime_build_fails_closed_outside_qualified_platform() -> None:
    cmake = MODEL_CMAKE.read_text(encoding="utf-8")

    assert 'CMAKE_SYSTEM_NAME STREQUAL "Linux"' in cmake
    assert 'CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64"' in cmake
    assert "NV_TENSORRT_MAJOR" in cmake
    assert "NV_TENSORRT_MINOR" in cmake
    assert "NV_TENSORRT_PATCH" in cmake
    assert "NV_TENSORRT_BUILD" in cmake
    assert 'STREQUAL "11.2.0.113"' in cmake
    assert "TRTMC_OPENPI_REQUIRE_QUALIFIED_RUNTIME AND NOT" in cmake
    assert "FATAL_ERROR" in cmake


def test_openpi_plugin_targets_only_qualified_gb300_architecture() -> None:
    cmake = MODEL_CMAKE.read_text(encoding="utf-8")

    assert 'TRTMC_OPENPI_CUDA_ARCHITECTURES "103"' in cmake
    assert "must be exactly '103'" in cmake
    assert "80-real" not in cmake


def test_model_proof_disables_non_native_core_dependencies() -> None:
    root_cmake = (REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    options = MODEL_OPTIONS.read_text(encoding="utf-8")

    assert '"${PROJECT_SOURCE_DIR}/src/runtime/models/*/model_options.cmake"' in root_cmake
    assert 'if(TRTMC_MODEL_PROOF_MODEL STREQUAL "openpi")' in options
    assert "set(TRTMC_ENABLE_LIBTORCH_MULTINOMIAL OFF CACHE BOOL" in options
    assert "set(TRTMC_ENABLE_TVM_FFI OFF CACHE BOOL" in options
