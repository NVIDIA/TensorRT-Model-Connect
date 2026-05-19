"""Tests for GitHub Actions CI wiring."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_workflows_define_shared_hf_cache_env() -> None:
    for workflow in ("nightly.yml", "trtmc-ci.yml"):
        text = (REPO_ROOT / ".github" / "workflows" / workflow).read_text()
        assert "TRTMC_STORAGE_ROOT:" in text
        assert "HF_HOME:" in text
        assert "HF_HUB_CACHE:" in text
        assert "HUGGINGFACE_HUB_CACHE:" in text
        assert "HF_MODULES_CACHE:" in text
        assert "/workspace/users/yifeif/tensorrt-model-connect/hf-cache" in text


def test_github_stage_wrapper_mounts_and_exports_hf_cache_env() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-gha-stage.sh").read_text()
    for name in (
        "TRTMC_STORAGE_ROOT",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_MODULES_CACHE",
    ):
        assert f'mkdir_if_set "${{{name}:-}}"' in text
        assert f"-e {name}" in text
    assert "/workspace/users/yifeif:/workspace/users/yifeif" in text


def test_github_stage_wrapper_exports_e2e_gpu_controls() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-gha-stage.sh").read_text()
    assert "-e TRTMC_E2E_EXCLUDE_GPU0" in text
    assert "-e TRTMC_E2E_DEPRIORITIZE_GPU0" in text


def test_github_stage_wrapper_keeps_cpp_unit_stages_cpu_only() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-gha-stage.sh").read_text()
    gpu_case = text.split("case \"$stage\" in", maxsplit=1)[1].split(
        "*)",
        maxsplit=1,
    )[0]
    assert "graph-ops|selective-e2e|full-e2e)" in gpu_case
    assert "python-builder" not in gpu_case
    assert "cpp-unit|cpp-coverage)" in text
    assert "TRTMC_CPP_CPU_ONLY=1" in text
    assert "-e TRTMC_CPP_CPU_ONLY" in text


def test_github_cpp_unit_stages_exclude_gpu_labeled_ctests() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    assert "-LE requires_gpu" in text
    assert "Excluding C++ tests labeled requires_gpu in CPU-only CI container" in text
    assert "Excluding C++ tests labeled requires_gpu from CPU-only coverage container" in text


def test_cpp_coverage_ci_wrapper_forwards_ctest_filters() -> None:
    text = (REPO_ROOT / "tools" / "coverage_ci" / "run_cpp_coverage.sh").read_text()
    assert '"${REPO_ROOT}/tools/coverage/cpp_coverage.sh" "$@"' in text


def test_cmake_labels_cuda_device_tests() -> None:
    text = (REPO_ROOT / "CMakeLists.txt").read_text()
    for test_name in (
        "test_c_abi_runtime_regression",
        "test_trt_runtime_lifetime",
        "test_trt_module",
        "test_cuda_buffer",
        "test_cuda_stream",
        "test_cuda_graph",
        "test_device_kv_cache",
        "test_device_resources",
        "test_device_tensor",
        "test_kv_cache_new",
        "test_triattention_kv_cache",
        "test_recurrent_state",
        "test_encoder_pipeline",
        "test_vl_pipeline",
    ):
        line = next(line for line in text.splitlines() if f"trtmc_add_test({test_name}" in line)
        assert "REQUIRES_GPU" in line


def test_github_stage_wrapper_does_not_export_diffusion_vlm_waives_file() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-gha-stage.sh").read_text()
    assert "DIFFUSION_VLM_WAIVES_FILE" not in text


def test_diffusion_vlm_gate_failures_are_not_waived() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    assert "DIFFUSION_VLM_WAIVES_FILE" not in text
    assert "--waives" not in text


def test_diffusion_vlm_pair_count_uses_helper() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    vlm_block = text.split("run_diffusion_vlm_assessment() {", maxsplit=1)[1].split(
        "\ngenerate_coverage_map() {",
        maxsplit=1,
    )[0]
    assert "tools/count_diffusion_frame_pairs.py e2e_artifacts/artifacts" in vlm_block
    assert "python3 -c" not in vlm_block


def test_full_python_builder_runs_e2e_harness_unit_tests() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    assert "tests/e2e_harness/test_*.py" in text


def test_shared_setup_action_creates_hf_cache_dirs() -> None:
    text = (REPO_ROOT / ".github" / "actions" / "setup-trtmc" / "action.yml").read_text()
    assert '"${HF_HOME:-}"' in text
    assert '"${HF_HUB_CACHE:-}"' in text
    assert '"${HUGGINGFACE_HUB_CACHE:-}"' in text
    assert '"${HF_MODULES_CACHE:-}"' in text


def test_github_workflows_keep_e2e_artifact_retention_aligned_with_ci_mode() -> None:
    premerge = (REPO_ROOT / ".github" / "workflows" / "trtmc-ci.yml").read_text()
    nightly = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    assert "name: trtmc-ci-${{ github.run_id }}" in premerge
    assert "retention-days: 1" in premerge
    assert "name: trtmc-nightly-${{ github.run_id }}" in nightly
    assert "retention-days: 14" in nightly


def test_github_workflows_keep_html_report_in_full_artifacts() -> None:
    expectations = {
        "trtmc-ci.yml": ("trtmc-ci-html-report-${{ github.run_id }}", "retention-days: 1"),
        "nightly.yml": (
            "trtmc-nightly-html-report-${{ github.run_id }}",
            "retention-days: 14",
        ),
    }
    for workflow, (artifact_name, retention) in expectations.items():
        text = (REPO_ROOT / ".github" / "workflows" / workflow).read_text()
        assert "Upload E2E HTML report" in text
        assert artifact_name in text
        assert "path: e2e_artifacts/e2e_report.html" in text
        assert retention in text
        assert "e2e_artifacts/" in text
        assert "!e2e_artifacts/e2e_report.html" not in text


def test_github_workflows_write_e2e_markdown_summary() -> None:
    for workflow in ("trtmc-ci.yml", "nightly.yml"):
        text = (REPO_ROOT / ".github" / "workflows" / workflow).read_text()
        assert "Write CI summary" in text
        assert "scripts/generate_ci_summary.py" in text
        assert ">> \"$GITHUB_STEP_SUMMARY\"" in text


def test_selective_e2e_zero_model_path_still_generates_report_input_dir() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    zero_model_block = text.split(
        'echo "No E2E models affected by this change -- skipping E2E tests"',
        maxsplit=1,
    )[1].split("fi", maxsplit=1)[0]
    assert "mkdir -p e2e_artifacts/artifacts" in zero_model_block
