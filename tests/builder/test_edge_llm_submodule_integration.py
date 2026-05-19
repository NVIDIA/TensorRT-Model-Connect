"""Contract checks for the optional TensorRT Edge-LLM source integration."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_edge_llm_submodule_is_pinned_and_documented() -> None:
    gitmodules = _read(".gitmodules")
    plan = _read("website/docs/architecture/edge-llm-submodule-integration.md")

    assert "path = third_party/tensorrt-edge-llm" in gitmodules
    assert (
        "url = https://gitlab-master.nvidia.com/TensorRT/tensorrt-edge-llm/"
        "tensorrt-edge-llm"
    ) in gitmodules
    assert "20e9ff86492e121b2fdfb9165fef4b799f97f664" in plan
    assert "git submodule update --init --recursive third_party/tensorrt-edge-llm" in plan


def test_edge_llm_cmake_uses_optional_submodule_provider_dso() -> None:
    cmake = _read("CMakeLists.txt")

    assert "TRTMC_EDGE_LLM_BUILD_DIR" in cmake
    assert "third_party/tensorrt-edge-llm" in cmake
    assert "TRTMC_ENABLE_EDGE_LLM_PROVIDER=ON" in cmake
    assert "add_library(trtmc_provider_edgellm SHARED" in cmake
    assert "target_link_libraries(trtmc_core PRIVATE" not in cmake.split(
        "# --- TensorRT Edge-LLM runtime provider (optional) ---", maxsplit=1
    )[1].split("# --- libtorch multinomial bridge (optional) ---", maxsplit=1)[0]
    assert "runtime provider into trtmc_core" not in cmake


def test_edge_llm_opt_in_ci_records_reproducibility_surface() -> None:
    workflow = _read(".github/workflows/edge-llm-provider.yml")
    script = _read(".github/scripts/run-edge-llm-provider-ci.sh")

    assert "workflow_dispatch" in workflow
    assert "submodules: recursive" in workflow
    assert "run-gha-stage.sh edge-llm-provider" in workflow
    assert "git submodule update --init --recursive" in script
    assert "edge_llm_commit=" in script
    assert "model_connect_commit=" in script
    assert "tensorrt_version=" in script
    assert "cuda_version=" in script
    assert "CMAKE_CUDA_ARCHITECTURES" in script
    assert "correctness.txt" in script
    assert "comparator=non_empty_text_output" in script
    assert "benchmark.txt" in script
    assert "cmake --build \"$edge_build\"" in script
    assert "cmake --build \"$provider_build\"" in script
    assert "trtmc-build build \"$model_dir\"" in script
    assert "python -m tensorrt_model_connect.cli" not in script
    assert "inspect \"$bundle_path\" --deployment" in script
    assert "run \"$bundle_path\"" in script
    assert "--benchmark" in script
    assert "tokens_per_sec" in script
    assert "sampled_peak_gpu" in script
    assert "$log_path\" > \"$benchmark_path" in script
