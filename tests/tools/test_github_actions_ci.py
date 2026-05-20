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


def test_github_stage_wrapper_exports_package_smoke_controls() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-gha-stage.sh").read_text()
    for name in (
        "TRTMC_PACKAGE_PYTHON_TAGS",
        "TRTMC_PACKAGE_WHEEL_ARCH",
        "TRTMC_PACKAGE_BUILD_ROOT",
        "TRTMC_PACKAGE_SMOKE_VENV",
        "TRTMC_WHEEL_QWEN_SMOKE_ROOT",
        "TRTMC_WHEEL_QWEN_MODEL_ID",
        "TRTMC_WHEEL_QWEN_MAX_CACHE",
        "TRTMC_WHEEL_QWEN_MAX_NEW_TOKENS",
        "TRTMC_WHEEL_QWEN_OPTIMIZATION_LEVEL",
        "TRTMC_WHEEL_QWEN_BUILD_TIMEOUT",
        "TRTMC_WHEEL_QWEN_RUN_TIMEOUT",
        "TRTMC_WHEEL_QWEN_PYTHON",
    ):
        assert f"-e {name}" in text


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


def test_nightly_runs_wheel_qwen_smoke_before_upload_and_release() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    package_index = text.index("Build trtmc pip package")
    smoke_index = text.index("Qwen smoke test from trtmc pip package")
    upload_index = text.index("Upload trtmc pip package artifact")
    publish_index = text.index("Publish trtmc pip package to GitHub Release")
    assert package_index < smoke_index < upload_index < publish_index
    assert "run-gha-stage.sh wheel-qwen-smoke" in text


def test_nightly_uses_manylinux_package_image_for_release_wheels() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    assert "TRTMC_PACKAGE_CI_IMAGE:" in text
    assert "trtmc-dev-gb300:manylinux_2_35" in text
    assert 'docker build -t "$TRTMC_PACKAGE_CI_IMAGE" -f Dockerfile .' in text
    assert "package image glibc=" in text
    assert "TRTMC_CI_IMAGE: ${{ env.TRTMC_PACKAGE_CI_IMAGE }}" in text


def test_package_stage_builds_py310_and_py312_wheels() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    assert "TRTMC_PACKAGE_PYTHON_TAGS:-py310 py312" in text
    assert 'WHEEL_PYVER="$tag"' in text
    assert "python -m build --wheel --outdir \"$PWD/dist\"" in text
    assert 'build-dir=$package_build_root/$tag' in text
    assert "manylinux_2_35_aarch64" in text
    assert "wheel-qwen-smoke)" in text
    assert "Qwen smoke test from trtmc pip package" in text


def test_package_stage_requires_manylinux_aarch64_wheels() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    assert 'TRTMC_PACKAGE_WHEEL_ARCH:-manylinux_2_35_aarch64' in text
    assert 'EXPECTED_PLATFORM = os.environ.get("TRTMC_PACKAGE_WHEEL_ARCH"' in text
    assert 'native wheel must not contain .data/purelib entries' in text
    assert ".data/scripts/trtmc" in text
    assert "native trtmc must be installed directly, not via console_scripts" in text
    assert '"auditwheel>=6.2"' in text
    assert 'sys.executable, "-m", "auditwheel", "show", wheel' in text
    assert "*-${py_tag}-none-manylinux_2_35_aarch64.whl" in text
    assert "validate_manylinux_build_environment" in text
    assert "build_glibc=" in text


def test_package_stage_uses_conan_py_build_inputs() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    assert "CONAN_PY_BUILD_PROFILE_AUTODETECT=1" in text
    assert 'TRTMC_TRT_INCLUDE_DIR="$trt_include"' in text
    assert 'TRTMC_TRT_LIBRARY="$trt_library"' in text
    assert 'TRTMC_CUDA_INCLUDE_DIR="$cuda_include"' in text
    assert 'TRTMC_CUDART_LIBRARY="$cudart_library"' in text


def test_release_wheel_build_disables_libtorch_linkage() -> None:
    text = (REPO_ROOT / "conanfile.py").read_text()
    assert 'toolchain.cache_variables["TRTMC_ENABLE_LIBTORCH_MULTINOMIAL"] = False' in text


def test_root_pyproject_configures_conan_py_build_wheel() -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text()
    assert 'requires = ["conan-py-build==0.4.3"]' in text
    assert 'build-backend = "conan_py_build.build"' in text
    assert 'packages = ["tensorrt_model_connect/tensorrt_model_connect"]' in text
    assert "[project.scripts]" not in text


def test_wheel_qwen_smoke_checks_py312_wheel_only() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    smoke_block = text.split("run_wheel_qwen_smoke() {", maxsplit=1)[1].split(
        "\n}",
        maxsplit=1,
    )[0]
    assert "select_wheel_by_tag py312 dist" in smoke_block
    assert "python312_bin" in smoke_block
    assert "select_compatible_wheel" not in smoke_block
    assert 'PATH="$smoke_venv/bin:$PATH"' not in smoke_block
    assert '"$smoke_venv/bin/trtmc" build "$model_id"' in smoke_block
    assert 'b"\\x7fELF"' in smoke_block


def test_selective_e2e_zero_model_path_still_generates_report_input_dir() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    zero_model_block = text.split(
        'echo "No E2E models affected by this change -- skipping E2E tests"',
        maxsplit=1,
    )[1].split("fi", maxsplit=1)[0]
    assert "mkdir -p e2e_artifacts/artifacts" in zero_model_block
