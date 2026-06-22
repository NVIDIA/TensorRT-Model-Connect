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
    stage_text = (REPO_ROOT / ".github" / "scripts" / "run-gha-stage.sh").read_text()
    start_text = (REPO_ROOT / ".github" / "scripts" / "start-gha-container.sh").read_text()
    for name in (
        "TRTMC_STORAGE_ROOT",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_MODULES_CACHE",
    ):
        assert f'mkdir_if_set "${{{name}:-}}"' in start_text
        assert name in start_text
        assert f"-e {name}" in stage_text
    assert "/workspace/users/yifeif:/workspace/users/yifeif" in start_text
    assert "docker exec" in stage_text


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
        "TRTMC_WHEEL_QWEN_MODEL_ID",
        "TRTMC_WHEEL_QWEN_MAX_CACHE",
        "TRTMC_WHEEL_QWEN_MAX_NEW_TOKENS",
        "TRTMC_WHEEL_QWEN_OPTIMIZATION_LEVEL",
        "TRTMC_WHEEL_QWEN_BUILD_TIMEOUT",
        "TRTMC_WHEEL_QWEN_RUN_TIMEOUT",
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


def test_premerge_ci_runs_from_manual_dispatch_or_trigger_labels() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "trtmc-ci.yml").read_text()
    trigger_block = text.split("permissions:", maxsplit=1)[0]
    assert "pull_request:" in trigger_block
    assert "types:" in trigger_block
    assert "- labeled" in trigger_block
    assert "workflow_dispatch:" in trigger_block
    assert "push:" not in trigger_block
    assert "issues: write" not in text
    assert "pull-requests: read" not in text
    assert "github.event_name == 'workflow_dispatch'" in text
    assert "github.event.label.name == 'run-ci'" in text
    assert "run-e2e" not in text
    assert "run-full-ci" not in text
    assert "Remove trigger label" not in text
    assert "actions/github-script" not in text
    assert "github.rest.issues.removeLabel" not in text


def test_label_triggered_premerge_ci_uses_pr_merge_ref_checkout() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "trtmc-ci.yml").read_text()
    checkout_block = text.split("- name: Check out source", maxsplit=1)[1].split(
        "\n\n",
        maxsplit=1,
    )[0]
    assert "uses: actions/checkout@v4" in checkout_block
    assert "ref:" not in checkout_block


def test_nightly_workflow_dispatch_can_validate_requested_ref() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    assert "workflow_dispatch:" in text
    assert "NIGHTLY_REF:" in text
    assert "ref: ${{ env.NIGHTLY_REF }}" in text
    assert 'base="origin/main"' in text


def test_workflow_dispatch_lint_uses_resolved_ci_base_ref() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    assert 'base_ref="${CI_BASE_REF:-origin/${GITHUB_REF_NAME:-main}}"' in text


def test_nightly_attempts_all_test_stages_after_failures() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    required_steps = (
        "Impact analysis",
        "Build C++ test executables",
        "Check family coverage",
        "Check cyclomatic complexity",
        "Lint changed files",
        "C++ unit tests",
        "Python builder and tools tests",
        "C++ coverage",
        "Graph-op GPU tests",
        "Full E2E tests",
        "Generate coverage map",
    )
    for step_name in required_steps:
        block = text.split(f"- name: {step_name}", maxsplit=1)[1].split("\n\n", maxsplit=1)[0]
        assert "if: ${{ always() }}" in block


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


def test_github_ci_uses_manylinux_image_and_builds_wheel_first() -> None:
    for workflow in ("trtmc-ci.yml", "nightly.yml"):
        text = (REPO_ROOT / ".github" / "workflows" / workflow).read_text()
        assert "TRTMC_CI_IMAGE:" in text
        assert "vars.TRTMC_MANYLINUX_CI_IMAGE" in text
        assert "vars.TRTMC_CI_IMAGE" not in text
        assert "trtmc-dev-gb300:manylinux_2_39" in text
        assert "TRTMC_PACKAGE_WHEEL_ARCH:" in text
        assert "manylinux_2_39_aarch64" in text
        assert "TRTMC_PACKAGE_CI_IMAGE" not in text
        assert text.index("Start CI container") < text.index("Build trtmc pip package")
        assert text.index("Build trtmc pip package") < text.index("Impact analysis")
        assert "Setup TensorRT-Model-Connect" not in text


def test_package_stage_builds_py310_and_py312_wheels() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    assert "TRTMC_PACKAGE_PYTHON_TAGS:-py310 py312" in text
    assert 'WHEEL_PYVER="$tag"' in text
    assert "python -m build --wheel --outdir \"$PWD/dist\"" in text
    assert 'build-dir=$package_build_root/$tag' in text
    assert "manylinux_2_39_aarch64" in text
    assert "wheel-qwen-smoke)" in text
    assert "Qwen smoke test from trtmc pip package" in text


def test_package_stage_requires_manylinux_aarch64_wheels() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    assert 'TRTMC_PACKAGE_WHEEL_ARCH:-manylinux_2_39_aarch64' in text
    assert 'EXPECTED_PLATFORM = os.environ.get("TRTMC_PACKAGE_WHEEL_ARCH"' in text
    assert 'native wheel must not contain .data/purelib entries' in text
    assert ".data/scripts/trtmc" in text
    assert "native trtmc must be installed directly, not via console_scripts" in text
    assert '"auditwheel>=6.2"' in text
    assert 'sys.executable, "-m", "auditwheel", "show", wheel' in text
    assert "*-${py_tag}-none-${wheel_arch}.whl" in text
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


def test_model_plugins_are_staged_for_installed_trtmc() -> None:
    cmake = (REPO_ROOT / "CMakeLists.txt").read_text()
    conanfile = (REPO_ROOT / "conanfile.py").read_text()
    loader = (
        REPO_ROOT / "src" / "runtime" / "registry" / "pipeline_plugin_loader.cpp"
    ).read_text()

    assert "install(TARGETS trtmc_model_${_trtmc_model}" in cmake
    assert "${CMAKE_INSTALL_LIBDIR}/trtmc/models/${_trtmc_model}" in cmake
    assert '"libtrtmc_model_*.so*"' in conanfile
    assert "model_plugins = sorted(package_bin.glob" in conanfile
    assert "TRTMC model plugin DSOs were not staged" in conanfile
    assert '"site-packages" / "tensorrt_model_connect" / "bin"' in loader
    assert '"trtmc" / "models"' in loader


def test_ci_source_build_defaults_to_packaged_libtorch_mode() -> None:
    conanfile = (REPO_ROOT / "conanfile.py").read_text()
    wrapper = (REPO_ROOT / ".github" / "scripts" / "run-gha-stage.sh").read_text()
    coverage = (REPO_ROOT / "tools" / "coverage" / "cpp_coverage.sh").read_text()
    assert 'toolchain.cache_variables["TRTMC_ENABLE_LIBTORCH_MULTINOMIAL"] = False' in conanfile
    assert 'TRTMC_ENABLE_LIBTORCH_MULTINOMIAL="${TRTMC_ENABLE_LIBTORCH_MULTINOMIAL:-OFF}"' in coverage
    assert '-DTRTMC_ENABLE_LIBTORCH_MULTINOMIAL="${TRTMC_ENABLE_LIBTORCH_MULTINOMIAL}"' in coverage
    assert "-e TRTMC_ENABLE_LIBTORCH_MULTINOMIAL" in wrapper


def test_ci_cpp_test_build_reuses_wheel_conan_tree() -> None:
    script = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    assert "TRTMC_CONAN_ENABLE_TEST_TARGETS=1" in script
    assert 'TRTMC_CONAN_BUILD_TARGETS="$targets"' in script
    assert 'conan build . -of "$TRTMC_REUSE_CONAN_OUT_DIR"' in script
    assert 'ctest --test-dir "$TRTMC_REUSE_CMAKE_BUILD_DIR"' in script
    assert "wheel_build_metadata_file" in script


def test_ci_prepares_isolated_model_plugins_for_e2e() -> None:
    script = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    assert "tools/model_plugin_isolation.py prepare" in script
    assert "trtmc_model_plugins" in script
    assert 'prepare_model_plugin_dir "$model_plugin_dir" --models-file e2e_models.txt' in script
    assert 'prepare_model_plugin_dir "$model_plugin_dir" --all' in script
    assert '--model-plugin-dir "$model_plugin_dir"' in script


def test_cpp_coverage_builds_excluded_test_target() -> None:
    coverage = (REPO_ROOT / "tools" / "coverage" / "cpp_coverage.sh").read_text()
    assert "--target trtmc_cpp_tests" in coverage


def test_root_pyproject_configures_conan_py_build_wheel() -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text()
    backend_text = (REPO_ROOT / "_pyproject_backend.py").read_text()
    assert 'build-backend = "_pyproject_backend"' in text
    assert 'return [_CONAN_PY_BUILD_REQUIREMENT]' in backend_text
    assert "conan_build.build_wheel" in backend_text
    assert "conan_build.build_sdist" in backend_text
    assert "_py_only_enabled" in backend_text
    assert 'packages = ["python/tensorrt_model_connect"]' in text
    assert "[project.scripts]" not in text


def test_wheel_qwen_smoke_checks_py312_wheel_only() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    smoke_block = text.split("run_wheel_qwen_smoke() {", maxsplit=1)[1].split(
        "\n}",
        maxsplit=1,
    )[0]
    assert "select_wheel_by_tag py312 dist" in smoke_block
    assert "sys.version_info[:2] != (3, 12)" in smoke_block
    assert "TRTMC_WHEEL_QWEN_PYTHON" not in smoke_block
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
