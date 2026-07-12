# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GitHub Actions CI wiring."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _single_default_model_config(filename: str) -> tuple[Path, dict]:
    configs = sorted((REPO_ROOT / "tests" / "e2e" / "models").glob(f"*/{filename}"))
    defaults = []
    for path in configs:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("default") is True:
            defaults.append((path, data))
    assert len(defaults) == 1
    return defaults[0]


def test_workflows_define_shared_hf_cache_env() -> None:
    nightly = (REPO_ROOT / ".github/workflows/nightly.yml").read_text()
    for name in (
        "TRTMC_STORAGE_ROOT",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_MODULES_CACHE",
    ):
        assert f"{name}:" in nightly

    proof = (REPO_ROOT / ".github/workflows/model-proof.yml").read_text()
    assert (
        "TRTMC_HF_CACHE: ${{ vars.TRTMC_HF_HOME || "
        "'/workspace/users/yifeif/tensorrt-model-connect/hf-cache' }}"
    ) in proof
    assert "TRTMC_HF_HUB_CACHE:" not in proof and "TRTMC_HF_MODULES_CACHE:" not in proof
    runner = (REPO_ROOT / ".github/scripts/run-model-proof.sh").read_text()
    assert "$HOME/.cache/huggingface" in runner
    assert "${TRTMC_HF_HUB_CACHE:-$hf_cache_root/hub}" in runner
    assert "-e HF_MODULES_CACHE=/work/hf-modules" in runner


def test_workflows_pull_tensorrt_sdk_from_ghcr_without_artifactory_secrets() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    assert "ghcr.io/nvidia/tensorrt-model-connect/tensorrt-sdk:11.2.0.113@sha256:" in dockerfile
    assert "ENV TRT_ROOT=" not in dockerfile
    assert "ENV PIP_FIND_LINKS=" not in dockerfile
    assert "ENV TRT_LIB_DIR=/opt/venv/lib/python3.12/site-packages/tensorrt_libs" in dockerfile
    assert "ENV TRT_INC_DIR=/usr/include/aarch64-linux-gnu" in dockerfile

    ci_script = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    install_call = 'install_tensorrt_sdk_wheel "$smoke_venv/bin/python"'
    assert ci_script.count(install_call) == 2

    for workflow in ("nightly.yml", "model-proof.yml"):
        text = (REPO_ROOT / ".github" / "workflows" / workflow).read_text()
        assert "packages: read" in text
        assert "GHCR_TOKEN: ${{ github.token }}" in text
        assert "DOCKER_CONFIG=$docker_config" in text
        assert "TRTMC_ARTIFACTORY_USERNAME" not in text
        assert "TRTMC_ARTIFACTORY_PASSWORD" not in text

    premerge = (REPO_ROOT / ".github" / "workflows" / "trtmc-ci.yml").read_text()
    assert "uses: ./.github/workflows/model-proof.yml" in premerge
    assert "packages: read" in premerge
    assert "TRTMC_ARTIFACTORY_USERNAME" not in premerge
    assert "TRTMC_ARTIFACTORY_PASSWORD" not in premerge


def test_tensorrt_sdk_publisher_is_temporary_and_self_contained() -> None:
    scripts = REPO_ROOT / "scripts"
    assert not (scripts / "load_artifactory_credentials.sh").exists()
    assert not (scripts / "fetch_tensorrt_sdk.sh").exists()

    publisher = (scripts / "publish_tensorrt_sdk.sh").read_text()
    assert "TEMPORARY:" in publisher
    assert "when TensorRT 11.2" in publisher
    assert "is publicly released" in publisher
    assert "load_artifactory_credentials()" in publisher
    assert "stage_tensorrt_sdk()" in publisher


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


def test_github_stage_wrapper_exports_premerge_unit_parallelism() -> None:
    stage = (REPO_ROOT / ".github" / "scripts" / "run-gha-stage.sh").read_text()
    start = (REPO_ROOT / ".github" / "scripts" / "start-gha-container.sh").read_text()
    for name in (
        "TRTMC_UNIT_BUILD_JOBS",
        "TRTMC_UNIT_TEST_JOBS",
        "TRTMC_PREMERGE_UNIT_SCOPE",
    ):
        assert f"-e {name}" in stage
        assert name in start


def test_github_stage_wrapper_exports_diffusion_vlm_config() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-gha-stage.sh").read_text()
    start_text = (REPO_ROOT / ".github" / "scripts" / "start-gha-container.sh").read_text()
    assert "-e DIFFUSION_VLM_CONFIG" in text
    assert "DIFFUSION_VLM_CONFIG" in start_text


def test_github_stage_wrapper_exports_package_smoke_controls() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-gha-stage.sh").read_text()
    for name in (
        "TRTMC_PACKAGE_PYTHON_TAGS",
        "TRTMC_PACKAGE_WHEEL_ARCH",
        "TRTMC_PACKAGE_BUILD_ROOT",
        "TRTMC_WHEEL_SMOKE_CONFIG",
        "TRTMC_WHEEL_SMOKE_MODEL_ID",
        "TRTMC_WHEEL_SMOKE_MAX_CACHE",
        "TRTMC_WHEEL_SMOKE_MAX_NEW_TOKENS",
        "TRTMC_WHEEL_SMOKE_OPTIMIZATION_LEVEL",
        "TRTMC_WHEEL_SMOKE_BUILD_TIMEOUT",
        "TRTMC_WHEEL_SMOKE_RUN_TIMEOUT",
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
    assert '--config "$vlm_config"' in vlm_block
    assert "python3 -c" not in vlm_block


def test_diffusion_vlm_assessment_default_is_model_owned() -> None:
    path, data = _single_default_model_config("diffusion_vlm_assessment.json")
    assert path.parent.parent == REPO_ROOT / "tests" / "e2e" / "models"
    for key in ("model_id", "max_side", "max_new_tokens", "timeout"):
        assert data.get(key)


def test_diffusion_vlm_shared_ci_has_no_model_owned_default() -> None:
    shared_paths = (
        REPO_ROOT / ".github" / "workflows" / "nightly.yml",
        REPO_ROOT / ".github" / "workflows" / "trtmc-ci.yml",
        REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh",
        REPO_ROOT / "tools" / "evaluate_diffusion_vlm_similarity.py",
    )
    _, data = _single_default_model_config("diffusion_vlm_assessment.json")
    forbidden = (str(data["model_id"]),)
    violations = [
        (path, needle)
        for path in shared_paths
        for needle in forbidden
        if needle in path.read_text(encoding="utf-8")
    ]
    assert not violations


def test_full_python_builder_runs_e2e_harness_unit_tests() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    assert "tests/e2e_harness/test_*.py" in text
    assert "--ignore=tests/builder/test_cli.py" not in text


def test_selective_python_always_runs_static_ci_smoke_tests() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    for test_path in (
        "tests/tools/test_github_actions_ci.py",
        "tests/tools/test_model_plugin_encapsulation_static.py",
        "tests/tools/test_schedule_e2e.py",
        "tests/tools/test_test_impact.py",
    ):
        assert test_path in text


def test_python_package_coverage_gate_excludes_family_owned_modules() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    assert "write_python_package_gate_coverage_config" in text
    assert "*/tensorrt_model_connect/families/*" in text
    assert 'python_cov_config="coverage/python-package-gate.coveragerc"' in text
    assert '--cov-config="$python_cov_config"' in text
    assert "PYTHON_COVERAGE_MIN_LINE" in text
    assert "PYTHON_COVERAGE_MIN_BRANCH" in text


def test_full_e2e_collection_uses_model_e2e_files_with_visible_errors() -> None:
    text = (REPO_ROOT / "scripts" / "run_e2e_parallel.sh").read_text()
    full_mode = text.split("mapfile -t E2E_COLLECT_FILES", maxsplit=1)[1].split(
        "\nTOTAL=", maxsplit=1
    )[0]
    assert "find tests/e2e/models -mindepth 2 -maxdepth 2 -type f" in full_mode
    assert "-name 'test_*_e2e.py'" in full_mode
    assert '"$HF_PYTHON" -m pytest "${E2E_COLLECT_FILES[@]}" --co -q' in full_mode
    assert 'grep "test_model_e2e\\[" | sort || true' in full_mode
    assert "2>/dev/null" not in full_mode


def test_qwen_flashinfer_scripts_skip_pytest_collection() -> None:
    for relpath in (
        "tests/e2e/models/qwen/test_flashinfer_plugin.py",
        "tests/e2e/models/qwen/test_flashinfer_trt_attention.py",
        "tests/e2e/models/qwen/test_qwen3_flashinfer.py",
    ):
        text = (REPO_ROOT / relpath).read_text(encoding="utf-8")
        assert 'if __name__ != "__main__":' in text
        assert "pytest.skip(" in text
        assert "allow_module_level=True" in text
        assert text.index("pytest.skip(") < text.index("import tvm_ffi")


def test_github_workflows_keep_e2e_artifact_retention_aligned_with_ci_mode() -> None:
    proof = (REPO_ROOT / ".github/workflows/model-proof.yml").read_text()
    nightly = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    assert "    name: ${{ inputs.model }}\n" in proof
    assert "Build + reference test" not in proof
    assert (
        "name: model-proof-${{ inputs.model }}-${{ inputs.revision }}-${{ github.run_attempt }}"
    ) in proof
    assert "retention-days: 7" in proof
    assert "${{ inputs.model }}/artifacts/" in proof
    assert "name: trtmc-nightly-${{ github.run_id }}" in nightly
    assert "retention-days: 14" in nightly


def test_github_workflows_publish_html_reports_for_nightly_and_model_proof() -> None:
    nightly = (REPO_ROOT / ".github/workflows/nightly.yml").read_text()
    assert "Upload E2E HTML report" in nightly
    assert "trtmc-nightly-html-report-${{ github.run_id }}" in nightly
    assert "path: e2e_artifacts/e2e_report.html" in nightly
    assert "retention-days: 14" in nightly
    assert "e2e_artifacts/" in nightly
    assert "!e2e_artifacts/e2e_report.html" not in nightly

    proof = (REPO_ROOT / ".github/workflows/model-proof.yml").read_text()
    assert "Upload isolated model proof artifact" in proof
    assert "model-proof-${{ inputs.model }}-${{ inputs.revision }}" in proof
    assert "model-proof-report.html" in proof
    assert "if-no-files-found: error" in proof
    assert "retention-days: 7" in proof

    premerge = (REPO_ROOT / ".github/workflows/trtmc-ci.yml").read_text()
    assert "model-proof.yml" in premerge
    assert "5 / Combined HTML report" in premerge
    assert "Download isolated model proof artifacts" in premerge
    assert "merge-multiple: false" in premerge
    assert (
        "pattern: model-proof-*-${{ needs.legal.outputs.tested_sha || github.sha }}-*" in premerge
    )
    assert "scripts/generate_model_proof_report.py" in premerge
    assert '--upstream-result "legal=$LEGAL_RESULT"' in premerge
    assert '--upstream-result "impact=$IMPACT_RESULT"' in premerge
    assert "Upload combined model proof HTML report" in premerge
    assert "model-proof-html-${{ needs.legal.outputs.tested_sha || github.sha }}" in premerge


def test_premerge_ci_is_triggered_only_by_one_shot_label_events() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "trtmc-ci.yml").read_text()
    trigger_block = text.split("permissions:", maxsplit=1)[0]
    assert "pull_request:" in trigger_block
    assert "- main" in trigger_block
    assert "CI-improvement" not in trigger_block
    types_block = trigger_block.split("types:", maxsplit=1)[1]
    assert "- labeled" in trigger_block
    for unwanted_type in (
        "opened",
        "reopened",
        "synchronize",
        "ready_for_review",
        "unlabeled",
    ):
        assert unwanted_type not in types_block
    assert "paths-ignore:" not in trigger_block
    assert "paths:" not in trigger_block
    assert "workflow_dispatch:" not in trigger_block
    assert "push:" not in trigger_block
    assert "inputs." not in text
    assert "contains(github.event.pull_request.labels.*.name" not in text
    assert "github.event.label.name == 'run-ci'" in text
    assert "permissions: {}" in text
    assert "pull-requests: read" not in text
    assert "run-e2e" not in text
    assert "run-full-ci" not in text
    assert "actions/github-script" not in text


def test_legal_job_pins_snapshot_rejects_forks_and_consumes_run_ci() -> None:
    text = (REPO_ROOT / ".github/workflows/trtmc-ci.yml").read_text()
    legal = text.split("\n  legal:", maxsplit=1)[1].split("\n  impact:", maxsplit=1)[0]

    assert "'Legal compliance' || 'Ignored label / Legal compliance'" in legal
    assert "pull-requests: write" in legal
    assert "contents: read" in legal
    assert text.count("pull-requests: write") == 1
    assert "issues: write" not in text
    for output in ("authorized", "tested_sha", "base_sha", "head_sha"):
        assert f"{output}: ${{{{ steps.authorize.outputs.{output} }}}}" in legal

    assert "TESTED_SHA: ${{ github.sha }}" in legal
    assert "BASE_SHA: ${{ github.event.pull_request.base.sha }}" in legal
    assert "HEAD_SHA: ${{ github.event.pull_request.head.sha }}" in legal
    assert "HEAD_REPOSITORY: ${{ github.event.pull_request.head.repo.full_name }}" in legal
    assert 'if [ "$TRIGGER_LABEL" != "run-ci" ]; then' in legal
    assert 'echo "authorized=false" >> "$GITHUB_OUTPUT"' in legal
    assert 'if [ "$HEAD_REPOSITORY" != "$GITHUB_REPOSITORY" ]; then' in legal

    capture_index = legal.index('echo "tested_sha=$TESTED_SHA"')
    delete_index = legal.index("gh api --silent")
    authorize_index = legal.index('echo "authorized=true"')
    assert capture_index < delete_index < authorize_index
    assert "--method DELETE" in legal
    assert 'issues/$PR_NUMBER/labels/run-ci"' in legal
    assert "|| true" not in legal

    assert "Check out pinned merge snapshot" in legal
    assert "ref: ${{ steps.authorize.outputs.tested_sha }}" in legal
    assert "persist-credentials: false" in legal
    assert "steps.authorize.outputs.authorized == 'true'" in legal


def test_unrelated_label_cannot_emit_or_satisfy_required_contexts() -> None:
    text = (REPO_ROOT / ".github/workflows/trtmc-ci.yml").read_text()
    legal = text.split("\n  legal:", maxsplit=1)[1].split("\n  impact:", maxsplit=1)[0]
    impact = text.split("\n  impact:", maxsplit=1)[1].split("\n  unit-tests:", maxsplit=1)[0]
    required = text.split("\n  required:", maxsplit=1)[1]

    # Both ruleset contexts exist only on the run-ci event. Unrelated labels
    # take distinct dynamic names and succeed cheaply instead of producing a
    # skipped required check, which GitHub would otherwise treat as success.
    assert "'Legal compliance' || 'Ignored label / Legal compliance'" in legal
    assert "'Premerge CI' || 'Ignored label / Premerge CI'" in required
    assert "if: ${{ always() }}" in required
    assert 'if [ "$TRIGGER_LABEL" != "run-ci" ]; then' in required
    ignored_index = required.index("Ignored unrelated label")
    validation_index = required.index('test "$LEGAL_RESULT" = "success"')
    assert ignored_index < validation_index
    assert "needs.legal.outputs.authorized == 'true'" in impact


def test_premerge_concurrency_separates_authorized_and_ignored_labels() -> None:
    text = (REPO_ROOT / ".github/workflows/trtmc-ci.yml").read_text()
    concurrency = text.split("concurrency:", maxsplit=1)[1].split("\njobs:", maxsplit=1)[0]
    assert "github.event.pull_request.number" in concurrency
    assert "github.event.label.name == 'run-ci' && 'authorized' || github.run_id" in concurrency
    assert "cancel-in-progress: ${{ github.event.label.name == 'run-ci' }}" in concurrency


def test_premerge_ci_exposes_the_model_owned_dependency_graph() -> None:
    text = (REPO_ROOT / ".github/workflows/trtmc-ci.yml").read_text()

    legal = text.split("\n  legal:", maxsplit=1)[1].split("\n  impact:", maxsplit=1)[0]
    impact = text.split("\n  impact:", maxsplit=1)[1].split("\n  unit-tests:", maxsplit=1)[0]
    unit_tests = text.split("\n  unit-tests:", maxsplit=1)[1].split(
        "\n  model-proof:", maxsplit=1
    )[0]
    model_proof = text.split("\n  model-proof:", maxsplit=1)[1].split("\n  no-model:", maxsplit=1)[
        0
    ]
    no_model = text.split("\n  no-model:", maxsplit=1)[1].split("\n  report:", maxsplit=1)[0]
    report = text.split("\n  report:", maxsplit=1)[1].split("\n  required:", maxsplit=1)[0]
    required = text.split("\n  required:", maxsplit=1)[1]

    assert "'Legal compliance' || 'Ignored label / Legal compliance'" in legal
    assert "python tools/legal_headers.py --check" in legal
    assert "needs: legal" in impact
    assert "needs.legal.outputs.authorized == 'true'" in impact
    assert "python3 tools/model_ci.py validate" in impact
    assert "python3 tools/model_ci.py impact" in impact
    assert "--platform-change-policy fallback" in impact
    assert "--platform-change-policy all" not in impact
    assert impact.count("--fallback-model ") == 5
    for model in ("deepseek_v2", "patchtsmixer", "segformer", "whisper", "ltx_video"):
        assert f"--fallback-model {model}" in impact
    assert "matrix: ${{ steps.impact.outputs.matrix }}" in impact
    assert "run_unit_tests: ${{ steps.impact.outputs.run_unit_tests }}" in impact
    assert "unit_scope: ${{ steps.impact.outputs.unit_scope }}" in impact

    assert "name: 3 / Unit / C++ and Python" in unit_tests
    assert "needs.impact.outputs.run_unit_tests == 'true'" in unit_tests
    assert "Start clean unit-test container" in unit_tests
    assert "run-gha-stage.sh premerge-unit" in unit_tests
    assert "TRTMC_PREMERGE_UNIT_SCOPE: ${{ needs.impact.outputs.unit_scope }}" in unit_tests
    assert "TRTMC_CI_WORKSPACE: ${{ github.workspace }}/premerge-unit-checkout" in unit_tests
    assert "path: premerge-unit-checkout" in unit_tests
    assert unit_tests.count("working-directory: ${{ env.TRTMC_CI_WORKSPACE }}") == 3
    assert 'TRTMC_CI_HARDENED: "true"' in unit_tests
    assert "TRTMC_CONTAINER_OPTIONS:" not in unit_tests
    assert "id: ci_image" in unit_tests
    assert "TRTMC_CI_IMAGE: ${{ steps.ci_image.outputs.image_ref }}" in unit_tests
    assert "timeout-minutes: 90" in unit_tests
    assert "timeout-minutes: 60" in unit_tests
    assert "Build trtmc pip package" not in unit_tests
    assert "chmod -R a+rwX" not in unit_tests
    assert ".ci/premerge-unit-build" not in unit_tests
    assert 'unit_scratch="${RUNNER_TEMP}/trtmc-premerge-unit-' in unit_tests
    assert 'rm -rf -- "$unit_scratch"' in unit_tests

    assert "- legal" in model_proof
    assert "- impact" in model_proof
    assert "- unit-tests" in model_proof
    assert "needs.legal.outputs.authorized == 'true'" in model_proof
    assert "needs.impact.outputs.has_models == 'true'" in model_proof
    assert "needs.impact.outputs.run_unit_tests == 'true'" in model_proof
    assert "needs.impact.outputs.run_unit_tests == 'false'" in model_proof
    assert "needs.unit-tests.result == 'success'" in model_proof
    assert "needs.unit-tests.result == 'skipped'" in model_proof
    assert "uses: ./.github/workflows/model-proof.yml" in model_proof
    assert "name: 4 / Model / ${{ matrix.model }}" in model_proof
    assert "fail-fast: true" in model_proof
    assert "continue-on-error" not in model_proof
    assert "max-parallel: 16" in model_proof
    assert "matrix: ${{ fromJSON(needs.impact.outputs.matrix) }}" in model_proof
    assert "model: ${{ matrix.model }}" in model_proof
    assert "models: ${{ needs.impact.outputs.affected_models }}" not in model_proof
    assert "expected_count:" not in model_proof

    assert "- legal" in no_model
    assert "- impact" in no_model
    assert "- unit-tests" in no_model
    assert "needs.legal.outputs.authorized == 'true'" in no_model
    assert "needs.impact.outputs.has_models == 'false'" in no_model
    assert 'none) echo "No model-owned, platform, or unit inputs changed."' in no_model
    assert 'unit) echo "Unit tests cover this change; no model proof is required."' in no_model

    for dependency in ("legal", "impact", "unit-tests", "model-proof", "no-model"):
        assert f"- {dependency}" in report
    assert "5 / Combined HTML report" in report
    assert "always()" in report
    assert "github.event.label.name == 'run-ci'" in report
    assert "needs.legal.outputs.authorized == 'true'" not in report
    assert '"upstream_results": upstream_results' in report
    assert "The combined report did not complete." in report
    assert "actions/download-artifact@v4" in report
    assert "merge-multiple: false" in report
    assert "needs.legal.outputs.tested_sha || github.sha" in report
    assert '--upstream-result "legal=$LEGAL_RESULT"' in report
    assert '--upstream-result "impact=$IMPACT_RESULT"' in report
    assert '--upstream-result "unit-tests=$UNIT_RESULT"' in report
    assert '--upstream-result "model-proof=$MODEL_RESULT"' in report
    assert '--upstream-result "no-model=$NO_MODEL_RESULT"' in report
    assert "scripts/generate_model_proof_report.py" in report
    assert "if python3 scripts/generate_model_proof_report.py" in report
    assert "compose_rc=$?" in report
    assert 'echo "exit_code=$compose_rc" >> "$GITHUB_OUTPUT"' in report
    assert "Upload combined model proof HTML report" in report
    assert "Enforce combined report certification" in report

    for dependency in ("legal", "impact", "unit-tests", "model-proof", "no-model", "report"):
        assert f"- {dependency}" in required
    assert "'Premerge CI' || 'Ignored label / Premerge CI'" in required
    assert "always()" in required
    assert 'test "$MODEL_RESULT" = "success"' in required
    assert 'test "$UNIT_RESULT" = "success"' in required
    assert 'test "$REPORT_RESULT" = "success"' in required


def test_premerge_ci_requires_gpu_free_source_quality() -> None:
    workflow = (REPO_ROOT / ".github/workflows/trtmc-ci.yml").read_text()
    source_quality = workflow.split("\n  source-quality:", maxsplit=1)[1].split(
        "\n  unit-tests:", maxsplit=1
    )[0]
    required = workflow.split("\n  required:", maxsplit=1)[1]
    stage = (REPO_ROOT / ".github/scripts/run-trtmc-ci.sh").read_text()
    source_quality_stage = stage.split("    source-quality)", maxsplit=1)[1].split(
        "      ;;", maxsplit=1
    )[0]

    assert "name: Source quality" in source_quality
    assert "needs: legal" in source_quality
    assert "needs.legal.outputs.authorized == 'true'" in source_quality
    assert "runs-on: ubuntu-latest" in source_quality
    assert "CI_BASE_REF: ${{ needs.legal.outputs.base_sha }}" in source_quality
    assert "ref: ${{ needs.legal.outputs.tested_sha }}" in source_quality
    assert "fetch-depth: 0" in source_quality
    assert "actions/setup-python@v5" in source_quality
    assert "pip install --disable-pip-version-check --quiet lizard ruff clang-format" in source_quality
    assert "bash .github/scripts/run-trtmc-ci.sh source-quality" in source_quality
    assert "self-hosted" not in source_quality
    assert "docker" not in source_quality.lower()
    assert "cuda" not in source_quality.lower()

    assert 'run_step "Check cyclomatic complexity" check_cyclomatic_complexity' in source_quality_stage
    assert 'run_step "Lint changed files" lint_changed_files' in source_quality_stage

    assert "- source-quality" in required
    assert "SOURCE_QUALITY_RESULT: ${{ needs.source-quality.result }}" in required
    assert 'test "$SOURCE_QUALITY_RESULT" = "success"' in required


def test_premerge_report_bootstrap_names_upstream_failure_before_checkout(
    tmp_path: Path,
) -> None:
    text = (REPO_ROOT / ".github/workflows/trtmc-ci.yml").read_text()
    block = text.split("- name: Bootstrap combined HTML before checkout", maxsplit=1)[1].split(
        "- name: Check out pinned merge snapshot for report tooling", maxsplit=1
    )[0]
    program = textwrap.dedent(
        block.split("<<'PY'\n", maxsplit=1)[1].split("\n          PY", maxsplit=1)[0]
    )
    output = tmp_path / "report"
    output.mkdir()

    subprocess.run(
        [
            sys.executable,
            "-",
            str(output),
            "",
            "a" * 40,
            "premerge",
            "",
            "failure",
            "skipped",
            "false",
            "skipped",
            "skipped",
            "skipped",
        ],
        input=program,
        text=True,
        check=True,
    )

    status = json.loads((output / "model-proof-report-status.json").read_text(encoding="utf-8"))
    assert status["outcome"] == "failed"
    assert status["upstream_results"]["legal"] == "failure"
    assert status["upstream_results"]["impact"] == "skipped"
    assert any("legal" in issue and "failure" in issue for issue in status["issues"])
    report = (output / "model-proof-report.html").read_text(encoding="utf-8")
    assert "legal" in report
    assert "failure" in report


def test_premerge_ci_preserves_the_main_ruleset_context_names() -> None:
    text = (REPO_ROOT / ".github/workflows/trtmc-ci.yml").read_text()
    legal = text.split("\n  legal:", maxsplit=1)[1].split("\n  impact:", maxsplit=1)[0]
    required = text.split("\n  required:", maxsplit=1)[1]

    assert "github.event.label.name == 'run-ci'" in legal
    assert "'Legal compliance' || 'Ignored label / Legal compliance'" in legal
    assert "uses: ./.github/workflows/legal.yml" not in legal
    assert "github.event.label.name == 'run-ci'" in required
    assert "'Premerge CI' || 'Ignored label / Premerge CI'" in required


def test_premerge_ci_compares_the_checked_out_merge_snapshot() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "trtmc-ci.yml").read_text()
    base_block = text.split("- name: Resolve comparison refs", maxsplit=1)[1].split(
        "- name: Validate ownership manifests", maxsplit=1
    )[0]

    assert "CAPTURED_BASE_SHA: ${{ needs.legal.outputs.base_sha }}" in base_block
    assert "CAPTURED_TESTED_SHA: ${{ needs.legal.outputs.tested_sha }}" in base_block
    assert "CAPTURED_HEAD_SHA: ${{ needs.legal.outputs.head_sha }}" in base_block
    assert 'base_tip_sha="$(git rev-parse "${CAPTURED_BASE_SHA}^{commit}")"' in base_block
    assert 'base_sha="$(git merge-base "$base_tip_sha" "$tested_sha")"' in base_block
    assert 'tested_sha="$(git rev-parse "${CAPTURED_TESTED_SHA}^{commit}")"' in base_block
    assert '--head "$TESTED_SHA"' in text


def test_label_triggered_premerge_ci_uses_pr_merge_ref_checkout() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "trtmc-ci.yml").read_text()
    legal_block = text.split("\n  legal:", maxsplit=1)[1].split("\n  impact:", maxsplit=1)[0]
    impact_block = text.split("\n  impact:", maxsplit=1)[1].split("\n  model-proof:", maxsplit=1)[0]
    model_block = text.split("\n  model-proof:", maxsplit=1)[1].split("\n  no-model:", maxsplit=1)[
        0
    ]

    assert "TESTED_SHA: ${{ github.sha }}" in legal_block
    assert "ref: ${{ steps.authorize.outputs.tested_sha }}" in legal_block
    assert "ref: ${{ needs.legal.outputs.tested_sha }}" in impact_block
    assert "revision: ${{ needs.legal.outputs.tested_sha }}" in model_block


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
    nightly = (REPO_ROOT / ".github/workflows/nightly.yml").read_text()
    assert "Write CI summary" in nightly
    assert "scripts/generate_ci_summary.py" in nightly
    assert '>> "$GITHUB_STEP_SUMMARY"' in nightly

    premerge = (REPO_ROOT / ".github/workflows/trtmc-ci.yml").read_text()
    assert "Summarize selection" in premerge
    assert "### Model impact" in premerge
    assert "All required premerge checks passed for" in premerge
    assert '>> "$GITHUB_STEP_SUMMARY"' in premerge


def test_nightly_runs_wheel_model_smoke_before_upload_and_release() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    package_index = text.index("Build trtmc pip package")
    smoke_index = text.index("Model smoke test from trtmc pip package")
    upload_index = text.index("Upload trtmc pip package artifact")
    publish_index = text.index("Publish trtmc pip package to GitHub Release")
    assert package_index < smoke_index < upload_index < publish_index
    assert "run-gha-stage.sh wheel-model-smoke" in text


def test_nightly_uses_manylinux_image_and_builds_wheel_first() -> None:
    text = (REPO_ROOT / ".github/workflows/nightly.yml").read_text()
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


def test_premerge_keeps_model_proofs_separate_from_source_only_units() -> None:
    premerge = (REPO_ROOT / ".github/workflows/trtmc-ci.yml").read_text()
    for obsolete_global_stage in (
        "Build trtmc pip package",
        "Build C++ test executables",
        "C++ unit tests",
        "Python builder and tools tests",
        "C++ coverage",
        "Graph-op GPU tests",
        "Selective E2E tests",
    ):
        assert obsolete_global_stage not in premerge
    assert premerge.count("run-gha-stage.sh") == 1
    assert "run-gha-stage.sh premerge-unit" in premerge
    assert "uses: ./.github/workflows/model-proof.yml" in premerge


def test_premerge_unit_stage_builds_no_model_plugins_or_native_wheel() -> None:
    script = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    stage = script.split("run_premerge_unit_tests() {", maxsplit=1)[1].split(
        "\nwrite_skipped_python_coverage() {", maxsplit=1
    )[0]
    cmake = (REPO_ROOT / "CMakeLists.txt").read_text()

    assert "pip install" not in stage
    assert 'PYTHONPATH="$source_root/python:$source_root' in stage
    assert 'TRTMC_CI_SCRATCH_DIR:-/tmp' in stage
    assert 'TRTMC_PREMERGE_UNIT_BUILD_DIR:-$scratch_dir/premerge-unit-build' in stage
    assert '-m "not gpu and not trt and not e2e and not model_proof_allocator"' in stage
    assert "tests/builder/" in stage
    assert "tests/tools/" in stage
    assert "tests/e2e_harness/test_*.py" in stage
    assert "-q -x" in stage
    assert "--dist=worksteal" in stage
    assert 'not model_proof_allocator"' in stage
    assert "-m model_proof_allocator" in stage
    assert "native_targets=(trtmc test_cli_args test_config_cli_support)" in stage
    assert "native_targets=(trtmc trtmc_platform_cpp_tests)" in stage
    assert 'TRTMC_PREMERGE_UNIT_SCOPE:-all' in stage
    assert "tests/builder/test_cli.py" in stage
    assert '"$build_dir/trtmc" version' in stage
    assert '"$build_dir/trtmc" --help' in stage
    assert "--stop-on-failure" in stage
    assert "libtrtmc_model_*.so*" in stage
    assert "-DTRTMC_ENABLE_TRT=OFF" not in stage
    assert "-DTRTMC_BUILD_BACKEND_TRT=OFF" not in stage
    assert "-DTRTMC_ENABLE_TVM_FFI=OFF" not in stage
    assert "conan " not in stage
    assert "build_pip_package" not in stage
    assert "trtmc_model_plugins" not in stage
    assert "add_custom_target(trtmc_platform_cpp_tests)" in cmake
    assert "trtmc_add_test(test_model_plugin_loader MODEL_OWNED)" in cmake
    assert "test_c_abi_runtime_regression   NO_SRC_INCLUDE MODEL_OWNED" in cmake
    assert "MODEL_OWNED\n        ${_trtmc_test_options}" in cmake
    for gpu_test in (
        "test_trt_runtime_lifetime REQUIRES_TRT REQUIRES_GPU",
        "test_trt_module REQUIRES_TRT REQUIRES_GPU",
        "test_cuda_buffer REQUIRES_TRT REQUIRES_GPU",
        "test_cuda_stream REQUIRES_TRT REQUIRES_GPU",
        "test_cuda_graph REQUIRES_TRT REQUIRES_GPU",
        "test_device_tensor REQUIRES_GPU",
        "test_tvm_ffi_plugin REQUIRES_TRT REQUIRES_GPU",
        "test_tvm_ffi_plugin_v2 NO_SRC_INCLUDE REQUIRES_TRT REQUIRES_GPU",
        "test_tvm_ffi_module_loader REQUIRES_TRT REQUIRES_GPU",
    ):
        assert f"trtmc_add_test({gpu_test})" in cmake


def test_premerge_unit_container_is_unprivileged_offline_and_cpu_only() -> None:
    start = (REPO_ROOT / ".github" / "scripts" / "start-gha-container.sh").read_text()
    workflow = (REPO_ROOT / ".github" / "workflows" / "trtmc-ci.yml").read_text()
    unit_tests = workflow.split("\n  unit-tests:", maxsplit=1)[1].split(
        "\n  model-proof:", maxsplit=1
    )[0]

    for option in (
        "--network none",
        "--read-only",
        "--tmpfs /tmp:rw,exec,nosuid,nodev,size=16g",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        '--user "$(id -u):$(id -g)"',
        "--ipc private",
        "-e HOME=/tmp",
        "-e TMPDIR=/work/tmp",
        "-e PIP_NO_INDEX=1",
        "-e TRTMC_CI_SCRATCH_DIR=/work",
        "-e NVIDIA_VISIBLE_DEVICES=void",
        "-e CUDA_VISIBLE_DEVICES=",
        "--runtime runc",
        'workspace_mount+=":ro"',
        'extra_mounts+=(-v "$scratch_host:/work")',
    ):
        assert option in start
    assert 'if [ "$hardened" = "true" ]' in start
    assert 'if [ "$hardened" != "true" ]' in start
    assert 'compgen -G "/dev/nvidia*"' in start
    assert "Hardened unit scratch must be inside RUNNER_TEMP" in start
    assert "Hardened unit scratch must not be a symlink" in start
    assert 'TRTMC_CI_HARDENED: "true"' in unit_tests
    assert "--gpus" not in unit_tests
    assert "/workspace/users/yifeif:/workspace/users/yifeif" not in unit_tests
    assert 'workspace="${TRTMC_CI_WORKSPACE:-${GITHUB_WORKSPACE:-}}"' in start
    assert 'workspace_mount="$workspace:$workspace"' in start
    hardened_allowlist = start.split('if [ "$hardened" = "true" ]; then', maxsplit=2)[2]
    hardened_allowlist = hardened_allowlist.split('env_args+=', maxsplit=1)[0]
    assert "HF_TOKEN" not in hardened_allowlist
    assert "HUGGING_FACE_HUB_TOKEN" not in hardened_allowlist

    stage = (REPO_ROOT / ".github" / "scripts" / "run-gha-stage.sh").read_text()
    hardened_exec = stage.split(
        'if [ "${TRTMC_CI_HARDENED:-false}" = "true" ]; then', maxsplit=1
    )[1].split("\nfi", maxsplit=1)[0]
    assert "TRTMC_PREMERGE_UNIT_SCOPE" in hardened_exec
    assert "HF_TOKEN" not in hardened_exec
    assert "HUGGING_FACE_HUB_TOKEN" not in hardened_exec


def test_unowned_gpu_only_builder_suites_are_excluded_from_cpu_units() -> None:
    stage = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    for relative in (
        "tests/builder/test_flashinfer_benchmark.py",
        "tests/builder/test_tvm_ffi_plugin.py",
    ):
        assert f"--ignore={relative}" in stage

    ffi_architecture = (REPO_ROOT / "tests/builder/test_ffi_architecture.py").read_text()
    flashinfer_section = ffi_architecture.split(
        "class TestFlashInferKernelSetup:", maxsplit=1
    )[1].split("class TestEngineBuilderKernelArtifacts:", maxsplit=1)[0]
    assert flashinfer_section.count("@pytest.mark.gpu") == 3
    assert flashinfer_section.count("@pytest.mark.trt") == 3


def test_model_proof_runs_one_isolated_model_with_unique_complete_evidence() -> None:
    proof = (REPO_ROOT / ".github/workflows/model-proof.yml").read_text()
    runner = (REPO_ROOT / ".github/scripts/run-model-proof.sh").read_text()

    assert "TRTMC_CI_IMAGE:" in proof
    assert "vars.TRTMC_MANYLINUX_CI_IMAGE" in proof
    assert "trtmc-dev-gb300:manylinux_2_39" in proof
    assert (
        "TRTMC_CI_IMAGE_LOCK_FILE: ${{ vars.TRTMC_CI_IMAGE_LOCK_FILE || "
        "'/tmp/trtmc-ci-docker-image.lock' }}"
    ) in proof
    assert "Ensure CI Docker image" in proof
    assert "bash .github/scripts/ensure-ci-docker-image.sh" in proof
    assert proof.count("actions/checkout@v4") == 1
    assert proof.count("bash .github/scripts/ensure-ci-docker-image.sh") == 1
    assert "TRTMC_HF_CACHE:" in proof
    assert "TRTMC_HF_HUB_CACHE:" not in proof
    assert "TRTMC_HF_MODULES_CACHE:" not in proof
    assert "TRTMC_MODEL_PROOF_BUILD_JOBS: ${{ vars.TRTMC_MODEL_PROOF_BUILD_JOBS || '2' }}" in proof
    assert "TRTMC_MODEL_PROOF_SLOTS_PER_GPU:" in proof
    assert "vars.TRTMC_MODEL_PROOF_SLOTS_PER_GPU || '4'" in proof
    assert "TRTMC_MODEL_PROOF_GPU_LOCK_DIR:" in proof
    assert "/tmp/trtmc-model-proof-gpu-locks" in proof
    assert "bash .github/scripts/run-model-proof.sh" in proof
    assert "run-model-proof-batch.sh" not in proof
    assert "env -u TRTMC_GPU_ID bash .github/scripts/run-model-proof.sh" in proof
    assert '--model "$MODEL"' in proof
    assert '--revision "$REVISION"' in proof
    assert '--suite "$SUITE"' in proof
    assert "Upload isolated model proof artifact" in proof
    assert "model-proof-${{ inputs.model }}-${{ inputs.revision }}" in proof
    assert "${{ github.run_attempt }}" in proof
    assert "model-proof-report.html" in proof
    assert "if-no-files-found: error" in proof
    assert "retention-days: 7" in proof
    assert "${{ inputs.model }}/artifacts/" in proof

    assert "dst=/src,readonly" in runner
    assert "dst=/hf-cache/hub,readonly" in runner
    assert "dst=/hf-cache/modules" not in runner
    assert "-e HF_MODULES_CACHE=/work/hf-modules" in runner
    assert "--network none" in runner
    assert "--read-only" in runner
    assert "proof.json" in runner


def test_package_stage_builds_py310_and_py312_wheels() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    assert "TRTMC_PACKAGE_PYTHON_TAGS:-py310 py312" in text
    assert 'WHEEL_PYVER="$tag"' in text
    assert 'python -m build --wheel --outdir "$PWD/dist"' in text
    assert "build-dir=$package_build_root/$tag" in text
    assert "manylinux_2_39_aarch64" in text
    assert "wheel-model-smoke)" in text
    assert "Model smoke test from trtmc pip package" in text


def test_package_smoke_default_is_model_owned() -> None:
    path, data = _single_default_model_config("package_smoke.json")
    assert path.parent.parent == REPO_ROOT / "tests" / "e2e" / "models"
    for key in (
        "name",
        "model_id",
        "bundle",
        "timing_cache",
        "max_cache",
        "max_new_tokens",
        "optimization_level",
        "build_timeout",
        "run_timeout",
        "precision",
        "prompt",
    ):
        assert data.get(key)
    assert isinstance(data.get("run_args", []), list)


def test_package_smoke_ci_surface_has_no_model_owned_names() -> None:
    shared_paths = (
        REPO_ROOT / ".github" / "workflows" / "nightly.yml",
        REPO_ROOT / ".github" / "scripts" / "run-gha-stage.sh",
        REPO_ROOT / ".github" / "scripts" / "start-gha-container.sh",
        REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh",
    )
    config_path, data = _single_default_model_config("package_smoke.json")
    family = config_path.parent.name
    model_name = str(data["name"])
    model_prefix = model_name.split("-", maxsplit=1)[0]
    family_tokens = {family, model_prefix}
    forbidden = {
        str(data[key]) for key in ("model_id", "name", "bundle", "timing_cache") if data.get(key)
    }
    for token in family_tokens:
        forbidden.update(
            {
                f"TRTMC_WHEEL_{token.upper()}",
                f"wheel-{token}-smoke",
                f"{token.title()} smoke test from trtmc pip package",
                f"trtmc-wheel-{token}-smoke",
            }
        )
    violations = [
        (path, needle)
        for path in shared_paths
        for needle in forbidden
        if needle in path.read_text(encoding="utf-8")
    ]
    assert not violations


def test_package_stage_requires_manylinux_aarch64_wheels() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    assert "TRTMC_PACKAGE_WHEEL_ARCH:-manylinux_2_39_aarch64" in text
    assert 'EXPECTED_PLATFORM = os.environ.get("TRTMC_PACKAGE_WHEEL_ARCH"' in text
    assert "native wheel must not contain .data/purelib entries" in text
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


def test_impact_stage_reuses_cached_json_for_summary() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    assert 'tools/test_impact.py "${impact_args[@]}" --json > impact.json' in text
    assert "ImpactResult(**json.load(f))" in text
    assert 'tools/test_impact.py "${impact_args[@]}" --verbose' not in text


def test_python_builder_fallback_is_per_tier() -> None:
    script = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    assert "builder_fallback = 'builder' in fallback" in script
    assert "tools_fallback = 'tools' in fallback" in script
    assert "if not (builder_fallback and tools_fallback):" in script
    assert "add(['tests/builder/'])" in script
    assert "add(['tests/tools/'])" in script
    assert "Path('tests/e2e_harness').glob('test_*.py')" in script
    assert "fallback.intersection({'builder', 'tools'})" not in script


def test_release_wheel_build_disables_libtorch_linkage() -> None:
    text = (REPO_ROOT / "conanfile.py").read_text()
    assert 'toolchain.cache_variables["TRTMC_ENABLE_LIBTORCH_MULTINOMIAL"] = False' in text


def test_model_plugins_are_staged_for_installed_trtmc() -> None:
    cmake = (REPO_ROOT / "CMakeLists.txt").read_text()
    conanfile = (REPO_ROOT / "conanfile.py").read_text()
    loader = (REPO_ROOT / "src" / "runtime" / "registry" / "pipeline_plugin_loader.cpp").read_text()

    assert "install(TARGETS trtmc_model_${_trtmc_model}" in cmake
    assert "${CMAKE_INSTALL_LIBDIR}/trtmc/models/${_trtmc_model}" in cmake
    assert 'cmake.build(target="trtmc_model_plugins")' in conanfile
    assert '"libtrtmc_model_*.so*"' in conanfile
    assert 'rglob("libtrtmc_model_*.so*")' in conanfile
    assert "src=str(model_plugin.parent)" in conanfile
    assert "model_plugins = sorted(package_bin.glob" in conanfile
    assert "TRTMC model plugin DSOs were not staged" in conanfile
    assert '"site-packages" / "tensorrt_model_connect" / "bin"' in loader
    assert '"trtmc" / "models"' in loader


def test_release_wheel_stages_core_runtime_and_uses_origin_rpath() -> None:
    cmake = (REPO_ROOT / "CMakeLists.txt").read_text()
    conanfile = (REPO_ROOT / "conanfile.py").read_text()
    script = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()

    assert 'set_target_properties(trtmc PROPERTIES' in cmake
    assert 'BUILD_RPATH_USE_ORIGIN TRUE' in cmake
    assert 'INSTALL_RPATH "\\$ORIGIN"' in cmake
    assert '"libtrtmc_core.so*"' in conanfile
    assert "for destination in (package_bin, wheel_data_scripts):" in conanfile
    assert "TRTMC core DSO was not staged beside the wheel script" in conanfile
    assert 'apache-tvm-ffi==0.1.12' in pyproject
    assert "script_core_entries" in script
    assert "grep -Fq '$ORIGIN'" in script
    assert "installed trtmc RUNPATH leaks the CI build directory" in script


def test_ci_source_build_defaults_to_packaged_libtorch_mode() -> None:
    conanfile = (REPO_ROOT / "conanfile.py").read_text()
    wrapper = (REPO_ROOT / ".github" / "scripts" / "run-gha-stage.sh").read_text()
    coverage = (REPO_ROOT / "tools" / "coverage" / "cpp_coverage.sh").read_text()
    assert 'toolchain.cache_variables["TRTMC_ENABLE_LIBTORCH_MULTINOMIAL"] = False' in conanfile
    assert (
        'TRTMC_ENABLE_LIBTORCH_MULTINOMIAL="${TRTMC_ENABLE_LIBTORCH_MULTINOMIAL:-OFF}"' in coverage
    )
    assert '-DTRTMC_ENABLE_LIBTORCH_MULTINOMIAL="${TRTMC_ENABLE_LIBTORCH_MULTINOMIAL}"' in coverage
    assert "-e TRTMC_ENABLE_LIBTORCH_MULTINOMIAL" in wrapper


def test_ci_cpp_test_build_reuses_wheel_conan_tree() -> None:
    script = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    assert "TRTMC_CONAN_ENABLE_TEST_TARGETS=1" in script
    assert 'TRTMC_CONAN_BUILD_TARGETS="$targets"' in script
    assert 'conan build . -of "$TRTMC_REUSE_CONAN_OUT_DIR"' in script
    assert 'ctest --test-dir "$TRTMC_REUSE_CMAKE_BUILD_DIR"' in script
    assert "wheel_build_metadata_file" in script


def test_selective_e2e_builds_and_runs_single_family_source_projections() -> None:
    script = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    selective = script.split("run_selective_e2e() {", maxsplit=1)[1].split(
        "\nrun_full_e2e() {", maxsplit=1
    )[0]
    group_runner = script.split("run_isolated_e2e_group() {", maxsplit=1)[1].split(
        "\nrun_selective_e2e() {", maxsplit=1
    )[0]

    assert "tools/model_plugin_isolation.py plan" in group_runner
    assert "tools/model_plugin_isolation.py" in selective
    assert "run_model_owned_isolation_e2e" in selective
    assert "impact-models" in selective
    assert "e2e_isolation_models.txt" in selective
    assert './scripts/run_e2e_parallel.sh "${standard_args[@]}"' in selective
    assert "--exclude-ci-tier nightly_only" in selective
    assert "--exclude-ci-tier multi_device" in selective
    assert 'if [ "$standard_rc" -ne 0 ]; then' in selective
    assert "Skipping strict model-owned isolation" in selective
    assert 'prepare_model_plugin_dir "$full_model_plugin_dir"' in selective
    assert "schedule_args=(" in group_runner
    assert "run_isolated_gpu_queue" in group_runner
    assert "queue_pids" in group_runner
    assert "tools/model_plugin_isolation.py stage-source" in group_runner
    assert "configure_isolated_model_build" in group_runner
    assert "CMAKE_TOOLCHAIN_FILE" in script
    assert "FETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON" in script
    assert '--target trtmc trtmc_backend_trt "$model_target"' in group_runner
    assert 'PYTHONPATH="$source_dir/python:$source_dir' in group_runner
    assert 'LD_LIBRARY_PATH="$isolated_library_path"' in group_runner
    assert '--trtmc-binary "$build_dir/trtmc"' in group_runner
    assert '--engine-dir "$engine_dir"' in group_runner
    assert '--model-plugin-dir "$model_plugin_dir"' in group_runner
    assert 'CUDA_VISIBLE_DEVICES="$gpu_id"' in group_runner
    assert '--rootdir "$source_dir"' in group_runner
    assert '--e2e-models-file "$models_file"' in group_runner
    assert "--e2e-exclude-ci-tier nightly_only" in group_runner
    assert "${SELECTIVE_E2E_GROUP_TIMEOUT:-90m}" in group_runner
    assert '"${model_filter_args[@]}"' in group_runner
    assert "scripts/run_e2e_parallel.sh" not in group_runner
    assert "verify-results" in group_runner
    assert "expected exactly 1" in group_runner
    assert "prepare_model_plugin_dir" not in group_runner


def test_full_e2e_stages_all_runtime_plugins_from_reusable_build() -> None:
    script = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    assert 'prepare_model_plugin_dir "$model_plugin_dir" --all' in script
    assert '--model-plugin-dir "$model_plugin_dir"' in script


def test_cpp_coverage_builds_excluded_test_target() -> None:
    coverage = (REPO_ROOT / "tools" / "coverage" / "cpp_coverage.sh").read_text()
    assert "--target trtmc_cpp_tests" in coverage


def test_cpp_coverage_gate_excludes_model_owned_runtime_plugins() -> None:
    coverage = (REPO_ROOT / "tools" / "coverage" / "cpp_coverage.sh").read_text()
    assert "GCOVR_EXCLUDES" in coverage
    assert '"${REPO_ROOT}/src/runtime/models"' in coverage
    assert 'gcovr_base+=(--exclude "${exclude}")' in coverage


def test_root_pyproject_configures_conan_py_build_wheel() -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text()
    backend_text = (REPO_ROOT / "_pyproject_backend.py").read_text()
    assert 'build-backend = "_pyproject_backend"' in text
    assert "return [_CONAN_PY_BUILD_REQUIREMENT]" in backend_text
    assert "conan_build.build_wheel" in backend_text
    assert "conan_build.build_sdist" in backend_text
    assert "_py_only_enabled" in backend_text
    assert 'packages = ["python/tensorrt_model_connect"]' in text
    assert "[project.scripts]" not in text


def test_wheel_model_smoke_checks_py312_wheel_only() -> None:
    text = (REPO_ROOT / ".github" / "scripts" / "run-trtmc-ci.sh").read_text()
    smoke_block = text.split("run_wheel_model_smoke() {", maxsplit=1)[1].split(
        "\n}",
        maxsplit=1,
    )[0]
    assert "select_wheel_by_tag py312 dist" in smoke_block
    assert "sys.version_info[:2] != (3, 12)" in smoke_block
    assert "TRTMC_WHEEL_SMOKE_PYTHON" not in smoke_block
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
