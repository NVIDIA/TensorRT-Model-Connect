# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Non-GPU checks for this profile's private qualification runners."""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest


LEAF = Path(__file__).resolve().parent
REPOSITORY = Path(__file__).resolve().parents[6]


def _json_include() -> Path | None:
    configured = os.environ.get("TRTMC_NLOHMANN_JSON_INCLUDE_DIR", "").strip()
    candidates = [Path(configured)] if configured else []
    candidates.extend(REPOSITORY.glob("build*/_deps/nlohmann_json-src/include"))
    return next((path for path in candidates if (path / "nlohmann" / "json.hpp").is_file()), None)


def test_protocol_header_compiles_and_executes_without_gpu(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    json_include = _json_include()
    if compiler is None or json_include is None:
        pytest.skip("a C++ compiler and nlohmann/json headers are required")
    source = tmp_path / "protocol.cpp"
    source.write_text(
        r"""
#include "qualification_runner.h"

int main(int argc, char** argv) {
    if (argc != 2)
        return 2;
    int generated = 0;
    int synchronized = 0;
    const auto measurements = qwen_edge_qualification::measure(
        [&] {
            ++generated;
            return qwen_edge_qualification::Sample{"answer", {1, 2, 3}};
        },
        [&] { ++synchronized; });
    if (generated != 35 || synchronized != 66 || measurements.iterations.size() != 30)
        return 1;
    const qwen_edge_qualification::RuntimeVersions versions{11, 2, 0, 113, 13030};
    qwen_edge_qualification::write_result(argv[1], "contract", versions, measurements);
    return 0;
}
""".lstrip(),
        encoding="utf-8",
    )
    binary = tmp_path / "protocol"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wpedantic",
            "-Werror",
            f"-I{LEAF}",
            f"-I{json_include}",
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )
    output = tmp_path / "result.json"
    subprocess.run([str(binary), str(output)], check=True)
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["observed_tensorrt_version"] == "11.2.0.113"
    assert result["observed_cuda_runtime_version"] == 13030
    assert result["decoding_cuda_graph_captured"] is True


def test_direct_runner_uses_the_official_edge_lifecycle_and_native_ids() -> None:
    source = (LEAF / "direct_runner.cpp").read_text(encoding="utf-8")
    assert 'dlsym(handle_, "initEdgellmPlugins")' in source
    assert "gLogger.setLevel(nvinfer1::ILogger::Severity::kWARNING)" in source
    assert "if (!runtime.captureDecodingCUDAGraph(stream))" in source
    assert "runtime.handleRequest(request, response, stream)" in source
    assert "std::move(response.outputIds.front())" in source
    for symbol in (
        "getInferLibMajorVersion()",
        "getInferLibMinorVersion()",
        "getInferLibPatchVersion()",
        "getInferLibBuildVersion()",
        "cudaRuntimeGetVersion(&cuda_runtime)",
    ):
        assert symbol in source
    assert 'throw std::runtime_error("loaded TensorRT runtime is not 11.2.0.113")' in source
    assert 'throw std::runtime_error("loaded CUDA runtime is not 13030")' in source
    assert 'check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize")' in source
    assert "cudaStreamSynchronize" not in source
    header = (LEAF / "qualification_runner.h").read_text(encoding="utf-8")
    assert '{"decoding_cuda_graph_captured", true}' in header


def test_mc_runner_uses_only_the_public_cpp_pipeline_and_native_ids() -> None:
    source = (LEAF / "mc_runner.cpp").read_text(encoding="utf-8")
    assert "trtmc::load(bundle.string(), load_options)" in source
    assert "pipeline->generate(config.prompt, generation)" in source
    assert "std::move(result.token_ids)" in source
    assert "cudaDeviceSynchronize()" in source
    for symbol in (
        "getInferLibMajorVersion()",
        "getInferLibMinorVersion()",
        "getInferLibPatchVersion()",
        "getInferLibBuildVersion()",
        "cudaRuntimeGetVersion(&cuda_runtime)",
    ):
        assert symbol in source
    assert 'throw std::runtime_error("loaded TensorRT runtime is not 11.2.0.113")' in source
    assert 'throw std::runtime_error("loaded CUDA runtime is not 13030")' in source


def test_edge_build_stamp_rehashes_recipe_and_exact_products(tmp_path: Path) -> None:
    scope = runpy.run_path(str(LEAF / "test_a100_e2e.py"))
    verify_stamp = scope["_verify_edge_build_stamp"]
    canonical_sha256 = scope["_canonical_sha256"]
    build = tmp_path / "edge-build"
    products = {
        "cpp/libedgellmCore.a": b"core",
        "libNvInfer_edgellm_plugin.so.1.0": b"plugin",
        "examples/llm/llm_build": b"build-tool",
    }
    for relative_path, content in products.items():
        path = build / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    recipe = {"edge_commit": "pinned", "schema_version": 1, "targets": ["three"]}
    stamp = {
        "products": {
            path: hashlib.sha256(content).hexdigest() for path, content in products.items()
        },
        "recipe": recipe,
        "recipe_sha256": canonical_sha256(recipe),
        "schema_version": 1,
    }
    stamp_path = build / ".trtmc-edge-build-stamp.json"

    def write_stamp() -> None:
        stamp_path.write_text(
            json.dumps(stamp, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    write_stamp()
    assert verify_stamp(build) == stamp

    core = build / "cpp/libedgellmCore.a"
    core.write_bytes(b"mutated")
    with pytest.raises(AssertionError):
        verify_stamp(build)
    core.write_bytes(products["cpp/libedgellmCore.a"])

    stamp["recipe_sha256"] = "0" * 64
    write_stamp()
    with pytest.raises(AssertionError):
        verify_stamp(build)
    stamp["recipe_sha256"] = canonical_sha256(recipe)

    extra = build / "unexpected-product"
    extra.write_bytes(b"unexpected")
    stamp["products"]["unexpected-product"] = hashlib.sha256(b"unexpected").hexdigest()
    write_stamp()
    with pytest.raises(AssertionError):
        verify_stamp(build)


def test_runner_build_is_leaf_local_and_pins_every_external_boundary() -> None:
    cmake = (LEAF / "CMakeLists.txt").read_text(encoding="utf-8")
    helper = (LEAF / "build_runners.py").read_text(encoding="utf-8")
    for variable in (
        "TRTMC_EDGE_LLM_SOURCE_DIR",
        "TRTMC_EDGE_LLM_BUILD_DIR",
        "TRTMC_EDGE_LLM_PLUGIN_LIBRARY",
        "TRTMC_TENSORRT_LIBRARY",
        "TRTMC_CUDART_LIBRARY",
        "TRTMC_CUDA_DRIVER_LIBRARY",
        "TRTMC_MC_INCLUDE_DIR",
        "TRTMC_MC_CORE_LIBRARY",
    ):
        assert f'set({variable} "" CACHE' in cmake
    assert "1ac0f2b99642045125e1c5ac7b109434ba3b36c7" in cmake
    assert "11.2.0.113" in cmake
    assert 'TRTMC_CUDA_VERSION STREQUAL "13.3"' in cmake
    assert 'REGEX "^CMAKE_HOME_DIRECTORY:INTERNAL="' in cmake
    assert 'STREQUAL "libNvInfer_edgellm_plugin.so.1.0"' in cmake
    assert "add_executable(trtmc_edgellm_direct_runner" in cmake
    assert "add_executable(trtmc_edgellm_mc_runner" in cmake
    assert (
        'target_link_libraries(trtmc_edgellm_mc_runner PRIVATE\n  "${TRTMC_MC_CORE_LIBRARY}"\n  "${TRTMC_TENSORRT_LIBRARY}"'
        in cmake
    )
    assert '"-DCMAKE_BUILD_TYPE=Release"' in helper
    assert '("CC", "CMAKE_C_COMPILER")' in helper
    assert '("CXX", "CMAKE_CXX_COMPILER")' in helper
    assert '("CUDAHOSTCXX", "CMAKE_CUDA_HOST_COMPILER")' in helper


def test_a100_gate_requires_installed_and_direct_runtime_surfaces() -> None:
    source = (LEAF / "test_a100_e2e.py").read_text(encoding="utf-8")
    for required in (
        "TRTMC_BINARY",
        "TRTMC_INSTALLED_PYTHON",
        "TRTMC_EDGELLM_DIRECT_RUNNER",
        "TRTMC_EDGELLM_MC_RUNNER",
    ):
        assert f'_required_executable("{required}")' in source
    assert "TRTMC_INSTALLED_BINARY" not in source
    assert 'package / "bin" / "trtmc"' in source
    assert "assert binary == installed_binary" in source
    assert "assert core_library == installed_core" in source
    assert '["ldd", str(binary)]' in source
    assert "str(binary.parent), original_library_path" not in source
    assert '.rsplit(" (", 1)[0].strip()' in source
    assert "loaded_core != expected_core" in source
    assert "installed executable did not resolve exactly one bundled libtrtmc_core" in source
    assert "TRTMC_REQUIRE_EDGELLM_DIRECT" not in source
    assert "direct EdgeLLM parity/performance proof: not configured" not in source
    assert "installed-package proof: not configured" not in source
