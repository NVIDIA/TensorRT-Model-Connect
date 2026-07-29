# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for GitHub Actions CI wiring."""

from __future__ import annotations

import ast
import json
import os
import re
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from tools.ci.quality import UnitTestRunner


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _ci_source(*filenames: str) -> str:
    """Return the review surface for one or more class-based CI modules."""
    return "\n".join(
        (REPO_ROOT / "tools" / "ci" / filename).read_text(encoding="utf-8")
        for filename in filenames
    )


def _single_default_model_config(filename: str) -> tuple[Path, dict]:
    configs = sorted((REPO_ROOT / "tests" / "e2e" / "models").glob(f"*/{filename}"))
    defaults = []
    for path in configs:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("default") is True:
            defaults.append((path, data))
    assert len(defaults) == 1
    return defaults[0]


def test_internal_ci_bridge_only_dispatches_an_exact_trusted_snapshot() -> None:
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "internal-ci-bridge.yml"
    ).read_text(encoding="utf-8")
    legacy_premerge = (
        REPO_ROOT / ".github" / "workflows" / "trtmc-ci.yml"
    ).read_text(encoding="utf-8")
    authorize = workflow.split("\n  authorize:", maxsplit=1)[1].split(
        "\n  dispatch:", maxsplit=1
    )[0]
    dispatch = workflow.split("\n  dispatch:", maxsplit=1)[1]
    authorize_permissions = authorize.split(
        "    permissions:", maxsplit=1
    )[1].split("\n    outputs:", maxsplit=1)[0]
    dispatch_permissions = dispatch.split("    permissions:", maxsplit=1)[1].split(
        "\n\n", maxsplit=1
    )[0]

    assert "pull_request_target:" in workflow
    assert "name: TensorRT-Model-Connect Internal CI Bridge" in workflow
    assert "types: [labeled]" in workflow
    assert "github.event.label.name == 'run-internal-ci'" in workflow
    assert "run-internal-ci" not in legacy_premerge
    assert "/labels/run-ci" not in workflow
    assert "workflow_dispatch:" in workflow
    assert "github.repository == 'NVIDIA/TensorRT-Model-Connect'" in workflow
    assert "github.event_name == 'workflow_dispatch'" in workflow
    assert "github.event_name == 'pull_request_target'" in workflow
    assert "github.ref == 'refs/heads/main'" in workflow
    assert "permissions: {}" in workflow
    assert authorize_permissions.strip() == "pull-requests: write"
    assert dispatch_permissions.strip() == "statuses: write"

    assert "collaborators/$ACTOR/permission" in authorize
    assert "write|maintain|admin)" in authorize
    assert "Only actors with write, maintain, or admin access" in authorize
    assert "environment:" not in authorize
    assert "secrets." not in authorize
    assert "statuses: write" not in authorize
    assert 'pull="$(gh api --method GET' in authorize
    assert 'test "$state" = "open"' in authorize
    assert 'test "$base_repo" = "$GITHUB_REPOSITORY"' in authorize
    assert 'test "$base_ref" = "main"' in authorize
    assert 'if [ "$EVENT_NAME" = "pull_request_target" ]; then' in authorize
    assert 'test "$base_sha" = "$EVENT_BASE_SHA"' in authorize
    assert 'test "$head_sha" = "$EVENT_HEAD_SHA"' in authorize
    for name in ("base_sha", "head_sha", "merge_sha"):
        assert f'[[ "${name}" =~ ^[0-9a-f]{{40}}$ ]]' in authorize
        assert f'echo "{name}=${name}"' in authorize
    assert "pr_number=$PR_NUMBER" in authorize
    assert (
        "/repos/$GITHUB_REPOSITORY/issues/$PR_NUMBER/labels/run-internal-ci"
    ) in authorize
    assert "gh api --silent --method DELETE" in authorize

    assert "needs: authorize" in dispatch
    assert "environment: ci-dispatch" in dispatch
    secret_references = set(re.findall(r"secrets\.([A-Z][A-Z0-9_]*)", workflow))
    assert secret_references == {
        "TRTMC_CI_DISPATCH_TOKEN",
        "TRTMC_PRIVATE_CI_OWNER",
        "TRTMC_PRIVATE_CI_REPOSITORY",
    }
    for secret in secret_references:
        assert secret not in authorize
        assert secret in dispatch
    assert workflow.count("${{ secrets.TRTMC_CI_DISPATCH_TOKEN }}") == 1

    assert "actions/create-github-app-token@" not in workflow
    assert "permission-checks:" not in workflow
    assert "actions/checkout@" not in workflow
    assert "private_ci_bridge.py" not in workflow
    assert "self-hosted" not in workflow
    assert "secrets: inherit" not in workflow
    assert "NVIDIA-dev/TensorRT-Model-Connect-CI" not in workflow
    assert (
        "/repos/$PRIVATE_CI_OWNER/$PRIVATE_CI_REPOSITORY/"
        "actions/workflows/premerge.yml/dispatches"
    ) in workflow
    assert re.search(r'"/repos/[A-Za-z0-9]', workflow) is None
    assert '[[ "$PRIVATE_CI_OWNER" =~ ^[A-Za-z0-9]' in workflow
    assert '[[ "$PRIVATE_CI_REPOSITORY" =~ ^[A-Za-z0-9]' in workflow
    assert 'echo "$PRIVATE_CI_' not in dispatch
    assert "GITHUB_STEP_SUMMARY" not in dispatch

    assert "/repos/$GITHUB_REPOSITORY/statuses/$MERGE_SHA" in dispatch
    assert "-f state=pending" in dispatch
    assert "-f state=failure" in dispatch
    assert dispatch.count("-f context=trtmc/premerge/required") == 2
    assert '-f description="PENDING: TensorRT-Model-Connect premerge"' in dispatch
    assert (
        '-f description="FAIL: TensorRT-Model-Connect premerge dispatch"'
    ) in dispatch
    assert dispatch.count(
        '-f target_url="https://github.com/$GITHUB_REPOSITORY/tree/$MERGE_SHA/tests"'
    ) == 2
    assert dispatch.index("/statuses/$MERGE_SHA") < dispatch.index(
        "actions/workflows/premerge.yml/dispatches"
    )

    assert 'ref: "main"' in dispatch
    for name in ("pr_number", "base_sha", "head_sha", "merge_sha"):
        assert f"{name}: ${name}" in dispatch
    assert "umask 077" in dispatch
    assert 'trap \'rm -f "$payload"\' EXIT' in dispatch
    assert 'if: ${{ failure() }}' in dispatch


def test_ci_orchestration_uses_the_class_based_python_entrypoint() -> None:
    legacy_scripts = (
        ".github/scripts/ensure-ci-docker-image.sh",
        ".github/scripts/start-gha-container.sh",
        ".github/scripts/run-gha-stage.sh",
        ".github/scripts/run-trtmc-ci.sh",
        ".github/scripts/run-model-proof.sh",
        "scripts/run_e2e_parallel.sh",
        "tools/coverage_ci/run_cpp_coverage.sh",
        "tools/coverage_ci/run_python_coverage.sh",
        "tools/coverage/cpp_coverage.sh",
        "tools/coverage/python_coverage.sh",
        "tools/coverage/run_coverage_all.sh",
    )
    assert not [path for path in legacy_scripts if (REPO_ROOT / path).exists()]

    source = _ci_source(
        "container.py",
        "coverage.py",
        "docker_image.py",
        "e2e_scheduler.py",
        "model_proof.py",
        "pipeline.py",
        "stage.py",
    )
    for class_name in (
        "CiContainer",
        "CoverageRunner",
        "DockerImageManager",
        "E2EParallelRunner",
        "ModelProofRunner",
        "CiPipeline",
        "ContainerStageRunner",
    ):
        assert f"class {class_name}" in source

    workflows = "\n".join(
        (REPO_ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
        for name in ("trtmc-ci.yml", "nightly.yml", "model-proof.yml")
    )
    assert "python3 -m tools.ci" in workflows
    assert "bash .github/scripts/" not in workflows


def test_ci_modules_have_minimal_role_comments_and_a_complete_tutorial() -> None:
    ci_directory = REPO_ROOT / "tools" / "ci"
    modules = sorted(ci_directory.glob("*.py"))
    for module in modules:
        docstring = ast.get_docstring(ast.parse(module.read_text(encoding="utf-8")))
        assert docstring, f"{module.name} has no module documentation"
        assert "Boundary:" in docstring, f"{module.name} does not state its responsibility boundary"

    readme = (ci_directory / "README.md").read_text(encoding="utf-8")
    missing_modules = [module.name for module in modules if f"`{module.name}`" not in readme]
    assert missing_modules == []
    contracts = readme.split("## Component contracts", 1)[1].split(
        "## Data passed between stages", 1
    )[0]
    for module in modules:
        match = re.search(
            rf"^### `{re.escape(module.name)}`$(.*?)(?=^### `|^## |\Z)",
            contracts,
            flags=re.MULTILINE | re.DOTALL,
        )
        assert match, f"{module.name} has no component contract"
        for field in ("Functionality / units", "Inputs", "Outputs", "Boundary"):
            assert f"**{field}:**" in match.group(1), f"{module.name} has no {field} contract"
    for section in (
        "## The system at a glance",
        "## Pre-merge, step by step",
        "## What nightly adds",
        "## Module map",
        "## Component contracts",
        "## Making a CI change",
        "## Reading a failure",
    ):
        assert section in readme


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
    assert (
        "HF_HUB_CACHE: ${{ vars.TRTMC_HF_HUB_CACHE || "
        "format('{0}/hub', vars.TRTMC_HF_HOME || "
        "'/workspace/users/yifeif/tensorrt-model-connect/hf-cache') }}"
    ) in nightly
    assert (
        "HUGGINGFACE_HUB_CACHE: ${{ vars.TRTMC_HF_HUB_CACHE || "
        "format('{0}/hub', vars.TRTMC_HF_HOME || "
        "'/workspace/users/yifeif/tensorrt-model-connect/hf-cache') }}"
    ) in nightly
    assert (
        "HF_MODULES_CACHE: ${{ vars.TRTMC_HF_MODULES_CACHE || "
        "format('{0}/modules', vars.TRTMC_HF_HOME || "
        "'/workspace/users/yifeif/tensorrt-model-connect/hf-cache') }}"
    ) in nightly

    proof = (REPO_ROOT / ".github/workflows/model-proof.yml").read_text()
    assert (
        "TRTMC_HF_CACHE: ${{ vars.TRTMC_HF_HOME || "
        "'/workspace/users/yifeif/tensorrt-model-connect/hf-cache' }}"
    ) in proof
    assert (
        "TRTMC_HF_HUB_CACHE: ${{ vars.TRTMC_HF_HUB_CACHE || "
        "format('{0}/hub', vars.TRTMC_HF_HOME || "
        "'/workspace/users/yifeif/tensorrt-model-connect/hf-cache') }}"
    ) in proof
    assert "TRTMC_HF_MODULES_CACHE:" not in proof
    runner = _ci_source("model_proof.py", "model_proof_inner.py")
    assert ".cache/huggingface" in runner
    assert 'self.context.env.get("TRTMC_HF_HUB_CACHE"' in runner
    assert '"HF_MODULES_CACHE": "/work/hf-modules"' in runner


def test_workflows_pull_tensorrt_sdk_from_ghcr_without_artifactory_secrets() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    assert "ghcr.io/nvidia/tensorrt-model-connect/tensorrt-sdk:11.2.0.113@sha256:" in dockerfile
    assert "ENV TRT_ROOT=" not in dockerfile
    assert "ENV PIP_FIND_LINKS=" not in dockerfile
    assert "ENV TRT_LIB_DIR=/opt/venv/lib/python3.12/site-packages/tensorrt_libs" in dockerfile
    assert "ENV TRT_INC_DIR=/usr/include/aarch64-linux-gnu" in dockerfile

    package = _ci_source("package.py")
    assert package.count("_install_tensorrt_sdk") >= 2

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
    stage_text = _ci_source("stage.py", "environment.py")
    start_text = _ci_source("container.py", "environment.py")
    for name in (
        "TRTMC_STORAGE_ROOT",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_MODULES_CACHE",
    ):
        assert name in start_text
        assert name in stage_text
    assert 'Path("/workspace/users/yifeif")' in start_text
    assert 'f"{shared_users}:{shared_users}"' in start_text
    assert '"docker"' in stage_text
    assert '"exec"' in stage_text


def test_github_stage_wrapper_removes_exact_container_on_cancellation(
    tmp_path: Path,
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    exec_started = tmp_path / "exec-started"
    container_removed = tmp_path / "container-removed"
    docker = fake_bin / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$*" >> "$DOCKER_LOG"\n'
        'case "${1:-}" in\n'
        "  inspect) printf 'true\\n' ;;\n"
        "  exec) touch \"$DOCKER_EXEC_STARTED\"; trap '' INT TERM; "
        'while [ ! -f "$DOCKER_REMOVED" ]; do sleep 0.1; done ;;\n'
        '  rm) touch "$DOCKER_REMOVED"; exit 0 ;;\n'
        "  *) exit 2 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    container_name = "trtmc-nightly-package-4242-1"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "DOCKER_LOG": str(docker_log),
            "DOCKER_EXEC_STARTED": str(exec_started),
            "DOCKER_REMOVED": str(container_removed),
            "TRTMC_CI_CONTAINER_NAME": container_name,
            "TRTMC_CI_WORKSPACE": str(workspace),
        }
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "tools.ci", "stage", "python-builder"],
        cwd=REPO_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not exec_started.is_file():
            assert process.poll() is None
            time.sleep(0.05)
        assert exec_started.is_file()
        started = time.monotonic()
        os.killpg(process.pid, signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=10)
        elapsed = time.monotonic() - started
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=10)

    assert process.returncode == 143, stdout + stderr
    assert elapsed < 2
    assert container_removed.is_file()
    assert f"rm -f {container_name}" in docker_log.read_text(encoding="utf-8")


def test_github_container_only_exports_nonempty_hf_transport_controls() -> None:
    stage_text = _ci_source("stage.py")
    start_text = _ci_source("container.py", "environment.py")

    assert "OPTIONAL_HUGGING_FACE_ENVIRONMENT" in start_text
    assert 'if self.env.get(name, "")' in start_text
    for name in ("HF_HUB_DISABLE_XET", "HF_HUB_DOWNLOAD_TIMEOUT", "HF_HUB_ETAG_TIMEOUT"):
        assert name not in stage_text


def test_github_stage_wrapper_exports_e2e_gpu_controls() -> None:
    text = _ci_source("environment.py")
    assert "TRTMC_E2E_EXCLUDE_GPU0" in text
    assert "TRTMC_E2E_DEPRIORITIZE_GPU0" in text


def test_github_stage_wrapper_exports_premerge_unit_parallelism() -> None:
    stage = _ci_source("stage.py", "environment.py")
    start = _ci_source("container.py", "environment.py")
    for name in (
        "TRTMC_UNIT_BUILD_JOBS",
        "TRTMC_UNIT_TEST_JOBS",
        "TRTMC_PREMERGE_UNIT_SCOPE",
    ):
        assert name in stage
        assert name in start


def test_github_stage_wrapper_exports_cpp_coverage_scope() -> None:
    stage = _ci_source("stage.py", "environment.py")
    start = _ci_source("container.py", "environment.py")
    assert "CPP_COVERAGE_SCOPE" in stage
    assert "CPP_COVERAGE_SCOPE" in start


def test_github_stage_wrapper_exports_diffusion_vlm_config() -> None:
    text = _ci_source("stage.py", "environment.py")
    start_text = _ci_source("container.py", "environment.py")
    assert "DIFFUSION_VLM_CONFIG" in text
    assert "DIFFUSION_VLM_CONFIG" in start_text


def test_github_stage_wrapper_exports_package_smoke_controls() -> None:
    text = _ci_source("environment.py")
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
        assert name in text


def test_github_stage_wrapper_does_not_export_diffusion_vlm_waives_file() -> None:
    text = _ci_source("stage.py", "environment.py")
    assert "DIFFUSION_VLM_WAIVES_FILE" not in text


def test_diffusion_vlm_gate_failures_are_not_waived() -> None:
    text = _ci_source("e2e.py")
    assert "DIFFUSION_VLM_WAIVES_FILE" not in text
    assert "--waives" not in text


def test_diffusion_vlm_pair_count_uses_helper() -> None:
    text = _ci_source("e2e.py")
    vlm_block = text.split("def diffusion_vlm_assessment", maxsplit=1)[1].split(
        "def _prepare_plugins", maxsplit=1
    )[0]
    assert "tools/count_diffusion_frame_pairs.py" in vlm_block
    assert '"--config"' in vlm_block
    assert "config_path" in vlm_block


def test_diffusion_vlm_assessment_default_is_model_owned() -> None:
    path, data = _single_default_model_config("diffusion_vlm_assessment.json")
    assert path.parent.parent == REPO_ROOT / "tests" / "e2e" / "models"
    for key in ("model_id", "max_side", "max_new_tokens", "timeout"):
        assert data.get(key)


def test_diffusion_vlm_shared_ci_has_no_model_owned_default() -> None:
    shared_paths = (
        REPO_ROOT / ".github" / "workflows" / "nightly.yml",
        REPO_ROOT / ".github" / "workflows" / "trtmc-ci.yml",
        REPO_ROOT / "tools" / "ci" / "e2e.py",
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
    text = _ci_source("coverage.py")
    builder = text.split("def python_builder_tests", maxsplit=1)[1].split("def cpp", maxsplit=1)[0]
    assert 'glob("test_*.py")' in builder
    assert "--ignore=tests/builder/test_cli.py" not in builder


def test_selective_python_always_runs_static_ci_smoke_tests() -> None:
    text = _ci_source("coverage.py")
    for test_path in (
        "tests/tools/test_github_actions_ci.py",
        "tests/tools/test_model_plugin_encapsulation_static.py",
        "tests/tools/test_schedule_e2e.py",
        "tests/tools/test_test_impact.py",
    ):
        assert test_path in text


def test_python_package_coverage_gate_excludes_family_owned_modules() -> None:
    text = _ci_source("coverage.py")
    assert "_write_python_config" in text
    assert "*/tensorrt_model_connect/families/*" in text
    assert 'self.directory / "python-package-gate.coveragerc"' in text
    assert 'f"--cov-config={config}"' in text
    assert "PYTHON_COVERAGE_MIN_LINE" in text
    assert "PYTHON_COVERAGE_MIN_BRANCH" in text


def test_full_e2e_collection_uses_model_e2e_files_with_visible_errors() -> None:
    text = _ci_source("e2e_scheduler.py")
    full_mode = text.split("def _collect_tests", maxsplit=1)[1].split(
        "def _model_name", maxsplit=1
    )[0]
    assert 'glob("*/test_*_e2e.py")' in full_mode
    assert '"--co"' in full_mode
    assert '"-q"' in full_mode
    assert '"test_model_e2e[" in line' in full_mode


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
    assert (
        "name: trtmc-nightly-html-report-${{ github.run_id }}-${{ github.run_attempt }}" in nightly
    )
    assert "retention-days: 14" in nightly


def test_optimized_runtime_proof_is_manual_and_not_a_premerge_dependency() -> None:
    workflows = REPO_ROOT / ".github" / "workflows"
    proof_path = workflows / "optimized-runtime-proof.yml"
    assert proof_path.is_file()
    assert not (workflows / "qwen-edgellm-a100.yml").exists()

    proof = proof_path.read_text(encoding="utf-8")
    assert "python3 tools/ci/optimized_runtime_qualifications.py" in proof
    assert "matrix: ${{ fromJSON(needs.select.outputs.producer_matrix) }}" in proof
    assert "runs-on: ${{ matrix.runner_labels }}" in proof
    assert "TRTMC_QUALIFICATION_PROFILE_FILES: ${{ matrix.profile_files }}" in proof
    assert '"$TRTMC_QUALIFICATION_ENTRYPOINT" --cleanup' in proof
    assert "actions/upload-artifact@v4" in proof
    assert "if-no-files-found: error" in proof
    assert "retention-days: 7" in proof
    assert "consumer_matrix" not in proof
    assert "TRTMC_QUALIFICATION_EXPORT_" not in proof
    assert "TRTMC_QUALIFICATION_INPUT_DIR" not in proof
    assert "actions/download-artifact@v4" not in proof
    assert "optimized-runtime-transfer-" not in proof
    assert "Qwen" not in proof
    assert "EdgeLLM" not in proof

    premerge = (workflows / "trtmc-ci.yml").read_text(encoding="utf-8")
    required = premerge.split("\n  required:", maxsplit=1)[1]
    assert "workflow_dispatch:" in proof
    assert "uses: ./.github/workflows/optimized-runtime-proof.yml" not in premerge
    assert "optimized-runtime-proof:" not in premerge
    assert "needs.optimized-runtime-proof" not in required
    assert "OPTIMIZED_RUNTIME_" not in required


def test_github_workflows_publish_html_reports_for_nightly_and_model_proof() -> None:
    nightly = (REPO_ROOT / ".github/workflows/nightly.yml").read_text()
    assert "Upload combined nightly HTML report" in nightly
    assert "trtmc-nightly-html-report-${{ github.run_id }}-${{ github.run_attempt }}" in nightly
    assert "model-proof-report.html" in nightly
    assert "model-proof-report-status.json" in nightly
    assert "retention-days: 14" in nightly
    assert "scripts/generate_model_proof_report.py" in nightly
    assert "--suite nightly" in nightly

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
    assert "authorized: ${{ steps.authorize.outputs.authorized }}" in legal
    for output in ("tested_sha", "base_sha", "head_sha"):
        assert f"{output}: ${{{{ steps.snapshot.outputs.{output} }}}}" in legal

    assert "TESTED_SHA: ${{ github.sha }}" in legal
    assert "HEAD_SHA: ${{ github.event.pull_request.head.sha }}" in legal
    assert "github.event.pull_request.base.sha" not in text
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
    unit_tests = text.split("\n  unit-tests:", maxsplit=1)[1].split("\n  model-proof:", maxsplit=1)[
        0
    ]
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
    assert "direct_models: ${{ steps.impact.outputs.direct_models }}" in impact
    assert "fallback_models: ${{ steps.impact.outputs.fallback_models }}" in impact
    assert "run_unit_tests: ${{ steps.impact.outputs.run_unit_tests }}" in impact
    assert "unit_scope: ${{ steps.impact.outputs.unit_scope }}" in impact
    assert "Directly affected models" in impact
    assert "Representative fallback models" in impact

    assert "name: 3 / Unit / C++ and Python" in unit_tests
    assert "needs.impact.outputs.run_unit_tests == 'true'" in unit_tests
    assert "Start clean unit-test container" in unit_tests
    assert "python3 -m tools.ci stage premerge-unit" in unit_tests
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
    assert "name: 4 / Model / ${{ matrix.model }} [${{ matrix.selection_kind }}]" in model_proof
    assert "fail-fast: true" in model_proof
    assert "continue-on-error" not in model_proof
    assert "max-parallel:" not in model_proof
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
    stage = _ci_source("pipeline.py", "quality.py")
    source_quality_stage = stage.split('"source-quality":', maxsplit=1)[1].split(
        '"cpp-unit":', maxsplit=1
    )[0]

    assert "name: Source quality" in source_quality
    assert "needs: legal" in source_quality
    assert "needs.legal.outputs.authorized == 'true'" in source_quality
    assert "runs-on: ubuntu-latest" in source_quality
    assert "CI_BASE_REF: ${{ needs.legal.outputs.base_sha }}" in source_quality
    assert "ref: ${{ needs.legal.outputs.tested_sha }}" in source_quality
    assert "fetch-depth: 0" in source_quality
    assert "actions/setup-python@v5" in source_quality
    assert (
        "pip install --disable-pip-version-check --quiet lizard ruff clang-format pytest"
        in source_quality
    )
    assert "python3 -m tools.ci pipeline source-quality" in source_quality
    assert "self-hosted" not in source_quality
    assert "docker" not in source_quality.lower()
    assert "cuda" not in source_quality.lower()

    assert '"Check cyclomatic complexity"' in source_quality_stage
    assert "self.quality.complexity" in source_quality_stage
    assert '"Lint changed files"' in source_quality_stage
    assert "self.quality.lint_changed_files" in source_quality_stage
    assert '"Check model architecture contracts"' in source_quality_stage
    assert "self.quality.architecture_contracts" in source_quality_stage

    architecture_contract = stage.split("def architecture_contracts", maxsplit=1)[1].split(
        "def _changed_files", maxsplit=1
    )[0]
    assert '"pytest"' in architecture_contract
    assert "tests/tools/test_model_plugin_encapsulation_static.py" in architecture_contract
    assert '"-q"' in architecture_contract
    assert '"no:cacheprovider"' in architecture_contract

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
    snapshot_block = text.split("- name: Resolve pinned PR merge snapshot", maxsplit=1)[1].split(
        "- name: Set up Python", maxsplit=1
    )[0]
    impact_block = text.split("\n  impact:", maxsplit=1)[1].split(
        "\n  source-quality:", maxsplit=1
    )[0]

    assert "CAPTURED_TESTED_SHA: ${{ steps.authorize.outputs.tested_sha }}" in snapshot_block
    assert "CAPTURED_HEAD_SHA: ${{ steps.authorize.outputs.head_sha }}" in snapshot_block
    assert 'tested_sha="$(git rev-parse "${CAPTURED_TESTED_SHA}^{commit}")"' in snapshot_block
    assert 'git rev-list --parents -n 1 "$tested_sha"' in snapshot_block
    assert '"${#commits[@]}" -ne 3' in snapshot_block
    assert 'base_sha="$(git rev-parse "${commits[1]}^{commit}")"' in snapshot_block
    assert 'head_sha="$(git rev-parse "${commits[2]}^{commit}")"' in snapshot_block
    assert 'if [ "$head_sha" != "$captured_head_sha" ]; then' in snapshot_block
    assert "github.event.pull_request.base.sha" not in text
    assert "git merge-base" not in snapshot_block
    assert "BASE_SHA: ${{ needs.legal.outputs.base_sha }}" in impact_block
    assert "TESTED_SHA: ${{ needs.legal.outputs.tested_sha }}" in impact_block
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


def test_nightly_pins_the_event_snapshot_before_any_job_is_queued() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    assert "workflow_dispatch:" in text
    assert "inputs:" not in text.split("permissions:", maxsplit=1)[0]
    assert "Nightly · ${{ github.ref_name }} · ${{ github.sha }}" in text
    assert "group: trtmc-nightly-${{ github.ref }}" in text
    legal = text.split("\n  legal:", maxsplit=1)[1].split("\n  inventory:", maxsplit=1)[0]
    assert "uses: ./.github/workflows/legal.yml" in legal
    assert "revision: ${{ github.sha }}" in legal


def test_reusable_legal_outputs_the_immutable_checked_out_sha() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "legal.yml").read_text()
    assert "tested_sha:" in text
    assert "value: ${{ jobs.legal-compliance.outputs.tested_sha }}" in text
    assert "tested_sha: ${{ steps.resolve.outputs.tested_sha }}" in text
    assert "id: resolve" in text
    assert "persist-credentials: false" in text
    assert "git rev-parse 'HEAD^{commit}'" in text
    assert 'git cat-file -e "${tested_sha}^{commit}"' in text
    assert 'echo "tested_sha=$tested_sha" >> "$GITHUB_OUTPUT"' in text


def test_nightly_reuses_the_resolved_sha_with_a_trigger_sha_fallback() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    validation, promotion = text.split("\n  release:", maxsplit=1)
    release = promotion.split("\n  nightly-issue:", maxsplit=1)[0]
    revision = "${{ needs.legal.outputs.tested_sha || github.sha }}"

    sha_consumers = [
        line
        for line in validation.splitlines()
        if "needs.legal.outputs.tested_sha" in line
    ]
    assert sha_consumers
    assert all("|| github.sha" in line for line in sha_consumers)
    assert f"revision: {revision}" in validation
    assert f"pattern: model-proof-*-{revision}-*" in validation
    assert f"TESTED_SHA: {revision}" in validation
    assert "needs.required.result == 'success'" in release
    assert "ref: ${{ needs.legal.outputs.tested_sha }}" in release
    assert "|| github.sha" not in release


def test_workflow_dispatch_lint_uses_resolved_ci_base_ref() -> None:
    text = _ci_source("quality.py")
    assert "f\"origin/{self.context.env.get('GITHUB_REF_NAME', 'main')}\"" in text


def test_manual_branch_nightly_lints_the_complete_main_diff() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    source_quality = text.split("\n  source-quality:", maxsplit=1)[1].split(
        "\n  unit-tests:", maxsplit=1
    )[0]
    comparison = source_quality.split("- name: Resolve source-quality comparison base", maxsplit=1)[
        1
    ].split("- name: Set up Python", maxsplit=1)[0]

    assert '"${GITHUB_EVENT_NAME:-}" = "workflow_dispatch"' in comparison
    assert '"${GITHUB_REF_NAME:-}" != "main"' in comparison
    assert 'base_sha="$(git merge-base "$tested_sha" origin/main)"' in comparison
    assert 'base_sha="$(git rev-parse "${tested_sha}^"' in comparison


def test_nightly_runs_every_runnable_stage_after_an_upstream_failure() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    inventory = text.split("\n  inventory:", maxsplit=1)[1].split(
        "\n  source-quality:", maxsplit=1
    )[0]
    source_quality = text.split("\n  source-quality:", maxsplit=1)[1].split(
        "\n  unit-tests:", maxsplit=1
    )[0]
    unit_tests = text.split("\n  unit-tests:", maxsplit=1)[1].split("\n  cache-warm:", maxsplit=1)[
        0
    ]
    cache_warm = text.split("\n  cache-warm:", maxsplit=1)[1].split("\n  package:", maxsplit=1)[0]
    package = text.split("\n  package:", maxsplit=1)[1].split("\n  model-proof:", maxsplit=1)[0]
    model_proof = text.split("\n  model-proof:", maxsplit=1)[1].split(
        "\n  diffusion-vlm:", maxsplit=1
    )[0]
    diffusion_vlm = text.split("\n  diffusion-vlm:", maxsplit=1)[1].split(
        "\n  report:", maxsplit=1
    )[0]
    report = text.split("\n  report:", maxsplit=1)[1].split("\n  required:", maxsplit=1)[0]
    required = text.split("\n  required:", maxsplit=1)[1].split("\n  release:", maxsplit=1)[0]

    assert "needs: legal" in inventory
    assert "if: ${{ !cancelled() }}" in inventory
    assert "ref: ${{ needs.legal.outputs.tested_sha || github.sha }}" in inventory
    assert 'python3 tools/model_ci.py validate --revision "$TESTED_SHA"' in inventory
    assert "python3 tools/model_ci.py all" in inventory
    assert '--revision "$TESTED_SHA"' in inventory
    assert '--github-output "$GITHUB_OUTPUT"' in inventory
    assert "!cancelled() && steps.checkout.outcome == 'success'" in inventory
    assert "!cancelled() && steps.inventory.outcome == 'success'" in inventory
    for output in (
        "has_models",
        "matrix",
        "affected_models",
        "expected_count",
        "expected_cases_by_model",
        "expected_result_count",
        "mode",
    ):
        assert f"{output}: ${{{{ steps.inventory.outputs.{output} }}}}" in inventory
    assert 'test "$EXPECTED_RESULT_COUNT" -ge "$EXPECTED_COUNT"' in inventory
    assert "sorted(cases_by_model) != models" in inventory
    assert "nightly case inventory count is inconsistent" in inventory

    assert "- legal" in source_quality and "- inventory" in source_quality
    assert "Check family coverage" in source_quality
    assert (
        "pip install --disable-pip-version-check --quiet lizard ruff clang-format pytest"
        in source_quality
    )
    assert "python3 -m tools.ci pipeline source-quality" in source_quality
    assert "- legal" in unit_tests and "- inventory" in unit_tests
    assert "python3 -m tools.ci stage premerge-unit" in unit_tests
    assert "TRTMC_PREMERGE_UNIT_SCOPE: all" in unit_tests
    assert 'TRTMC_CI_HARDENED: "true"' in unit_tests
    for independent_stage in (source_quality, unit_tests, cache_warm, package):
        assert "!cancelled()" in independent_stage
        assert "needs.legal.result == 'success'" not in independent_stage
        assert "needs.inventory.result == 'success'" not in independent_stage
        assert "needs.legal.outputs.tested_sha || github.sha" in independent_stage

    for dependency in (
        "legal",
        "inventory",
        "source-quality",
        "unit-tests",
        "cache-warm",
        "package",
    ):
        assert f"- {dependency}" in model_proof
    assert "!cancelled()" in model_proof
    assert "needs.legal.result == 'success'" not in model_proof
    assert "needs.inventory.outputs.has_models == 'true'" in model_proof
    for upstream in (
        "inventory",
        "source-quality",
        "unit-tests",
        "cache-warm",
        "package",
    ):
        assert f"needs.{upstream}.result == 'success'" not in model_proof
    assert "fail-fast: false" in model_proof
    assert "max-parallel:" not in model_proof
    assert "matrix: ${{ fromJSON(needs.inventory.outputs.matrix) }}" in model_proof
    assert "uses: ./.github/workflows/model-proof.yml" in model_proof
    assert "model: ${{ matrix.model }}" in model_proof
    assert "revision: ${{ needs.legal.outputs.tested_sha || github.sha }}" in model_proof
    assert "suite: nightly" in model_proof

    assert "- legal" in diffusion_vlm
    assert "- inventory" in diffusion_vlm
    assert "- model-proof" in diffusion_vlm
    assert "!cancelled()" in diffusion_vlm
    assert "needs.legal.result == 'success'" not in diffusion_vlm
    assert "needs.inventory.outputs.has_models == 'true'" in diffusion_vlm
    assert "needs.inventory.result == 'success'" not in diffusion_vlm
    assert "needs.model-proof.result == 'success'" not in diffusion_vlm
    assert "needs.legal.outputs.tested_sha || github.sha" in diffusion_vlm

    assert "if: ${{ always() }}" in report
    for dependency in (
        "legal",
        "inventory",
        "source-quality",
        "unit-tests",
        "cache-warm",
        "package",
        "model-proof",
        "diffusion-vlm",
    ):
        assert f"- {dependency}" in report
    assert "if: ${{ always() }}" in required
    for dependency in (
        "legal",
        "inventory",
        "source-quality",
        "unit-tests",
        "cache-warm",
        "package",
        "model-proof",
        "diffusion-vlm",
        "report",
    ):
        assert f"- {dependency}" in required


def test_nightly_source_quality_does_not_use_self_hosted_shared_storage() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    source_quality = text.split("\n  source-quality:", maxsplit=1)[1].split(
        "\n  unit-tests:", maxsplit=1
    )[0]

    assert "runs-on: ubuntu-latest" in source_quality
    for variable in (
        "ENGINE_DIR",
        "TRTMC_STORAGE_ROOT",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "HF_MODULES_CACHE",
    ):
        assert f'{variable}: ""' in source_quality
    assert "/workspace/" not in source_quality


def test_nightly_has_no_fail_fast_validation_matrix_or_cache_warm() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()

    assert "fail-fast: true" not in text
    assert "--fail-fast" not in text


def test_nightly_self_hosted_stages_use_the_configured_proof_runner_pool() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    selector = (
        "runs-on: ${{ fromJSON(vars.TRTMC_MODEL_RUNNER_LABELS || "
        "vars.TRTMC_RUNNER_LABELS || '[\"self-hosted\"]') }}"
    )

    for start, end in (
        ("unit-tests", "cache-warm"),
        ("package", "model-proof"),
        ("diffusion-vlm", "report"),
    ):
        block = text.split(f"\n  {start}:", maxsplit=1)[1].split(f"\n  {end}:", maxsplit=1)[0]
        assert selector in block


def test_nightly_warms_the_shared_cache_once_on_each_gpu_node() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    cache = text.split("\n  cache-warm:", maxsplit=1)[1].split("\n  package:", maxsplit=1)[0]

    assert "name: 3 / Model caches / ${{ matrix.node_label }}" in cache
    assert "fail-fast: false" in cache
    assert "runs-on: ${{ matrix.node_label }}" in cache
    assert "timeout-minutes: 270" in cache
    assert cache.count("- trtmc-node-") == 2
    for node in ("gb300-nvl-019-compute01", "gb300-nvl-019-compute02"):
        assert f"trtmc-node-{node}" in cache


def test_nightly_strictly_warms_all_active_non_multi_device_cases() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    cache = text.split("\n  cache-warm:", maxsplit=1)[1].split("\n  package:", maxsplit=1)[0]
    model_warm = cache.split(
        "- name: Strictly warm every active single-GPU nightly model", maxsplit=1
    )[1].split("- name: Strictly warm declared Nightly model references", maxsplit=1)[0]
    assert "Strictly warm every active single-GPU nightly model" in cache
    assert '--user "$(id -u):$(id -g)"' in model_warm
    assert model_warm.index("--user") < model_warm.index('"$TRTMC_CI_CONTAINER_NAME"')
    for argument in (
        "python -u scripts/warm_hf_cache.py",
        "--exclude-ci-tier multi_device",
        "--strict",
        "--attempt-timeout-seconds 600",
    ):
        assert argument in cache
    assert "--fail-fast" not in cache
    assert 'HF_HUB_DOWNLOAD_TIMEOUT: "60"' in cache
    assert 'HF_HUB_ETAG_TIMEOUT: "30"' in cache
    assert "--exclude-ci-tier l0_only" not in cache
    assert "--exclude-ci-tier nightly_only" not in cache


def test_nightly_strictly_warms_declared_model_references_on_every_node() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    cache = text.split("\n  cache-warm:", maxsplit=1)[1].split("\n  package:", maxsplit=1)[0]
    reference_warm = cache.split(
        "- name: Strictly warm declared Nightly model references", maxsplit=1
    )[1].split("- name: Clean cache-warm container", maxsplit=1)[0]

    assert (
        "TRTMC_MODEL_REFERENCE_CACHE_ROOT: "
        "${{ vars.TRTMC_MODEL_REFERENCE_CACHE_ROOT || "
        "'/workspace/users/yifeif/tensorrt-model-connect' }}" in cache
    )
    assert "if: ${{ !cancelled() }}" in reference_warm
    assert "working-directory: ${{ env.TRTMC_CI_WORKSPACE }}" in reference_warm
    assert (
        "python3 -m tools.ci model-reference-cache warm --suite nightly"
        in reference_warm
    )


def test_nightly_report_requires_exact_all_model_results_and_evidence() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    report = text.split("\n  report:", maxsplit=1)[1].split("\n  required:", maxsplit=1)[0]
    assert "EXPECTED_MODELS: ${{ needs.inventory.outputs.affected_models }}" in report
    assert "EXPECTED_RESULT_COUNT: ${{ needs.inventory.outputs.expected_result_count }}" in report
    assert (
        "EXPECTED_CASES_BY_MODEL: ${{ needs.inventory.outputs.expected_cases_by_model }}" in report
    )
    assert "REVISION: ${{ needs.legal.outputs.tested_sha || github.sha }}" in report
    assert "pattern: model-proof-*-${{ needs.legal.outputs.tested_sha || github.sha }}-*" in report
    assert "merge-multiple: false" in report
    assert "scripts/generate_model_proof_report.py" in report
    assert '--expected-cases-by-model "$EXPECTED_CASES_BY_MODEL"' in report
    assert '--expected-result-count "$EXPECTED_RESULT_COUNT"' in report
    assert "--suite nightly" in report
    for upstream in (
        "legal",
        "inventory",
        "source-quality",
        "unit-tests",
        "cache-warm",
        "package",
        "model-proof",
        "diffusion-vlm",
    ):
        assert f'--upstream-result "{upstream}=$' in report
    assert '"outcome": "passed"' in report
    assert '"discovered_models": expected' in report
    assert '"selected_artifact_count": len(expected)' in report
    assert 'entry.get("status") != "passed"' in report
    assert '"expected_result_count": expected_result_count' in report
    assert '"expected_cases_by_model": expected_cases_by_model' in report
    assert '"result_count": expected_result_count' in report
    assert "selected_cases != inventory_cases" in report
    assert "result_cases != inventory_cases" in report
    assert "reported_result_count != expected_result_count" in report
    assert "expected exactly {expected_result_count}" in report
    assert "at least one E2E result per model" not in report
    assert "issue_count" in report
    assert "Enforce complete nightly report certification" in report


def test_nightly_preserves_diffusion_vlm_gate_and_injects_it_into_the_html() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    vlm = text.split("\n  diffusion-vlm:", maxsplit=1)[1].split("\n  report:", maxsplit=1)[0]
    report = text.split("\n  report:", maxsplit=1)[1].split("\n  required:", maxsplit=1)[0]
    assert "needs:" in vlm and "- model-proof" in vlm
    assert "!cancelled()" in vlm
    assert "tools/count_diffusion_frame_pairs.py" in vlm
    assert "--require-complete" in vlm
    assert "No complete TRT/reference diffusion frame pairs were produced" in vlm
    assert "tools/evaluate_diffusion_vlm_similarity.py" in vlm
    assert "diffusion_vlm_assessment.json" in vlm
    assert "validate_complete_diffusion_frame_pairs" in vlm
    assert 'assessment["coverage_complete"] = True' in vlm
    assert 'assessment["source_revision"] = source_revision' in vlm
    assert 'assessment["workflow_run_id"] = workflow_run_id' in vlm
    assert 'assessment["workflow_run_attempt"] = workflow_run_attempt' in vlm
    assert "needs.model-proof.result == 'success'" not in vlm
    assert "does not exactly cover current frame pairs" in vlm
    assert "Upload diffusion semantic assessment" in vlm
    assert (
        "trtmc-nightly-vlm-${{ github.run_id }}-"
        "${{ needs.legal.outputs.tested_sha || github.sha }}-${{ github.run_attempt }}" in vlm
    )
    assert "Download diffusion semantic assessment" in report
    assert (
        "pattern: trtmc-nightly-vlm-${{ github.run_id }}-"
        "${{ needs.legal.outputs.tested_sha || github.sha }}-*" in report
    )
    assert "merge-multiple: false" in report
    assert "Select latest diffusion semantic assessment attempt" in report
    assert "tools/select_latest_attempt_artifact.py" in report
    assert 'cp "$VLM_ASSESSMENT"' in report
    assert 'diffusion_vlm_assessment.json"' in report
    assert '--upstream-result "diffusion-vlm=$VLM_RESULT"' in report
    assert '--upstream-result "model-artifact-download=$DOWNLOAD_OUTCOME"' in report
    assert '--upstream-result "vlm-artifact-download=$VLM_DOWNLOAD_OUTCOME"' in report
    assert '--upstream-result "vlm-artifact-selection=$VLM_SELECTION_OUTCOME"' in report


def test_nightly_all_gpu_gate_uses_the_model_proof_machine_lock() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    vlm = text.split("\n  diffusion-vlm:", maxsplit=1)[1].split("\n  report:", maxsplit=1)[0]
    proof_runner = _ci_source("gpu_lease.py")

    assert "TRTMC_MODEL_PROOF_GPU_LOCK_DIR:" in text
    assert "whole-machine.lock" in vlm
    assert 'flock -w "$TRTMC_WHOLE_MACHINE_GPU_LOCK_TIMEOUT_SECONDS" -x 9' in vlm
    assert 'FileLock(self.lock_dir / "whole-machine.lock")' in proof_runner
    assert "self._wait_lock(self.machine, deadline, shared=True)" in proof_runner
    assert text.index("- model-proof", text.index("\n  diffusion-vlm:")) < text.index(
        "Run diffusion VLM semantic gate"
    )


def test_nightly_grants_mutation_permissions_only_after_the_final_gate() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    before_release, after_release = text.split("\n  release:", maxsplit=1)
    release, issue_tracker = after_release.split("\n  nightly-issue:", maxsplit=1)
    assert "permissions: {}" in before_release
    assert "contents: write" not in before_release
    assert "issues: write" not in before_release
    assert text.count("contents: write") == 1
    assert text.count("issues: write") == 1
    assert "contents: write" in release
    assert "issues: write" not in release
    assert "needs.required.result == 'success'" in release
    assert "github.event_name == 'schedule' || github.ref == 'refs/heads/main'" in release
    assert "target_commitish" in release
    assert 'os.environ["TESTED_SHA"]' in release
    assert "issues: write" in issue_tracker
    assert "actions: read" in issue_tracker
    assert "contents: read" in issue_tracker
    assert "- required" in issue_tracker
    assert "- release" in issue_tracker
    assert "always()" in issue_tracker
    assert "github.event_name == 'schedule'" in issue_tracker
    assert "github.ref == 'refs/heads/main'" in issue_tracker
    assert "github.repository == 'NVIDIA/TensorRT-Model-Connect'" in issue_tracker
    assert "NIGHTLY_REQUIRED_RESULT: ${{ needs.required.result }}" in issue_tracker
    assert "NIGHTLY_RELEASE_RESULT: ${{ needs.release.result }}" in issue_tracker
    assert "python3 tools/nightly_issue_tracker.py --apply" in issue_tracker
    assert text.count("python3 tools/nightly_issue_tracker.py --apply") == 1
    assert "persist-credentials: false" in issue_tracker
    assert "secrets: inherit" not in text


def test_nightly_removes_the_legacy_monolithic_gpu_and_coverage_paths() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    for obsolete in (
        "python3 -m tools.ci stage impact",
        "python3 -m tools.ci stage graph-ops",
        "python3 -m tools.ci stage full-e2e",
        "python3 -m tools.ci stage coverage-map",
        "Impact analysis",
        "Graph-op GPU tests",
        "Full E2E tests",
        "ETTh1 task-eval",
        "Generate coverage map",
        "RUN_COVERAGE_MAP:",
        "tools/test_impact.py",
    ):
        assert obsolete not in text


def test_nightly_preserves_python_and_cpp_coverage_without_rebuilding_the_wheel() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    package = text.split("\n  package:", maxsplit=1)[1].split("\n  model-proof:", maxsplit=1)[0]
    assert package.count("python3 -m tools.ci stage package") == 1
    assert "Python package coverage" in package
    assert "python3 -m tools.ci stage python-builder" in package
    assert "C++ platform coverage" in package
    assert "python3 -m tools.ci stage cpp-coverage" in package
    assert "CPP_COVERAGE_SCOPE: platform" in package
    assert 'FULL_E2E: "true"' in package
    for threshold in (
        'PYTHON_COVERAGE_MIN_LINE: "38"',
        'PYTHON_COVERAGE_MIN_BRANCH: "25"',
        'CPP_COVERAGE_MIN_LINE: "36"',
        'CPP_COVERAGE_MIN_FUNCTION: "46"',
        'CPP_COVERAGE_MIN_BRANCH: "21"',
    ):
        assert threshold in package
    assert "Upload nightly coverage artifacts" in package
    assert "trtmc-nightly-coverage-${{ github.run_id }}" in package

    coverage = _ci_source("coverage.py")
    assert 'build_target, ctest_args = "trtmc_platform_cpp_tests", ["-L", "platform"]' in coverage
    assert "class CppCoverageEngine" in coverage
    assert '["cmake", "--build", build_dir, "--target", build_target, *parallel]' in coverage
    assert 'if build_target == "trtmc_cpp_tests":' in coverage


def test_nightly_long_jobs_reserve_finalization_time() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    package = text.split("\n  package:", maxsplit=1)[1].split("\n  model-proof:", maxsplit=1)[0]
    vlm = text.split("\n  diffusion-vlm:", maxsplit=1)[1].split("\n  report:", maxsplit=1)[0]

    package_job_minutes = 330
    package_step_minutes = 90 + 5 + 60 + 60 + 60
    vlm_job_minutes = 420
    vlm_step_minutes = 90 + 5 + 270
    assert "timeout-minutes: 330" in package
    assert package_job_minutes >= package_step_minutes + 30
    assert "timeout-minutes: 420" in vlm
    assert vlm_job_minutes >= vlm_step_minutes + 30


def test_nightly_python_coverage_runs_allocator_contract_serially() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    package = workflow.split("\n  package:", maxsplit=1)[1].split("\n  model-proof:", maxsplit=1)[0]
    script = _ci_source("coverage.py")
    builder_conftest = (REPO_ROOT / "tests" / "builder" / "conftest.py").read_text()
    coverage = script.split("def python_builder_tests", maxsplit=1)[1].split("def cpp", maxsplit=1)[
        0
    ]

    assert "python3 -m tools.ci stage python-builder" in package
    assert '"-n", "auto"' in coverage
    assert '"not model_proof_allocator and not gpu and not trt"' in coverage
    assert '"tests/tools/test_model_proof_runner.py"' in coverage
    assert '"model_proof_allocator"' in coverage
    assert '"--cov-append"' in coverage
    assert '"TRTMC_TEST_INSTALLED_WHEEL": "1"' in coverage
    assert "source_pkgs =" in script and "tensorrt_model_connect" in script
    assert 'os.environ.get("TRTMC_TEST_INSTALLED_WHEEL") == "1"' in builder_conftest
    assert "imported tensorrt_model_connect" in builder_conftest
    assert coverage.index('"-n", "auto"') < coverage.index(
        '"tests/tools/test_model_proof_runner.py"'
    )
    assert "TRTMC_CPU_CONTAINER_OPTIONS" in package
    assert "--gpus all" not in package


def test_nightly_graph_stage_numbers_are_unambiguous() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    assert "name: 5 / Diffusion semantic assessment" in text
    assert "name: 6 / Combined HTML report" in text
    assert "name: 7 / Nightly CI" in text
    assert "name: 8 / Publish nightly wheels" in text
    assert "name: 9 / Nightly failure issue" in text


def test_github_workflows_write_e2e_markdown_summary() -> None:
    nightly = (REPO_ROOT / ".github/workflows/nightly.yml").read_text()
    assert "Write nightly report summary" in nightly
    assert "### Nightly isolated model report" in nightly
    assert "trtmc-nightly-html-report-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}" in nightly
    assert '>> "$GITHUB_STEP_SUMMARY"' in nightly

    premerge = (REPO_ROOT / ".github/workflows/trtmc-ci.yml").read_text()
    assert "Summarize selection" in premerge
    assert "### Model impact" in premerge
    assert "All required premerge checks passed for" in premerge
    assert '>> "$GITHUB_STEP_SUMMARY"' in premerge


def test_etth1_model_proofs_use_the_single_task_eval_entry_point() -> None:
    stage = _ci_source("task_eval.py", "model_proof.py", "model_proof_inner.py")
    task_eval = (REPO_ROOT / "tools" / "task_eval.py").read_text()

    assert '"/src/tools/task_eval.py"' in stage
    assert '"prepare-ci-dataset"' in stage
    assert '"eval"' in stage
    assert "task_eval_ci.py" not in stage
    for argument in (
        "--suite",
        "etth1_time_series_parity",
        "--ci-lane",
        "nightly",
        "--engine-dir",
        "/work/engines",
        "--model-plugin-dir",
        "/work/model-plugins",
        "--require-prebuilt-bundles",
    ):
        assert f'"{argument}"' in stage
    assert "ETTh1 task-eval requires a GB300 GPU" in stage
    assert '"--network"' in stage and '"none"' in stage
    assert "validate_eval_summary" in task_eval
    assert 'result.get("status") == "passed"' in task_eval
    assert "return complete and all" in task_eval
    assert (
        '"work_dir"'
        not in task_eval.split("def _public_ci_result", maxsplit=1)[1].split(")", maxsplit=1)[0]
    )


def test_etth1_dataset_preparation_imports_the_projected_python_package(tmp_path: Path) -> None:
    from tools.ci.task_eval import TaskEvalDatasetPreparer

    projection = tmp_path / "projection"
    work = tmp_path / "work"
    artifacts = tmp_path / "artifacts"
    destination = work / "task-eval-data" / TaskEvalDatasetPreparer.DATASET
    projection.mkdir()
    artifacts.mkdir()
    docker_commands: list[list[str]] = []

    class Commands:
        @staticmethod
        def run(command, **_kwargs):
            rendered = [str(item) for item in command]
            docker_commands.append(rendered)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text("date,HUFL\n", encoding="utf-8")
            return subprocess.CompletedProcess(rendered, 0, "", "")

    class Context:
        commands = Commands()

        @staticmethod
        def run(command, **_kwargs):
            return subprocess.CompletedProcess(command, 0, "", "")

    result = TaskEvalDatasetPreparer(
        Context(),
        "nightly",
        "timesfm",
        projection,
        work,
        artifacts,
        "ci-image",
        "dataset-container",
        [],
    ).prepare()

    assert result == destination.parents[1]
    assert len(docker_commands) == 1
    command = docker_commands[0]
    image_index = command.index("ci-image")
    assert command.index("PYTHONPATH=/src/python:/src") < image_index
    assert command.index("PYTHONNOUSERSITE=1") < image_index
    assert command[image_index + 1 : image_index + 4] == [
        "/opt/venv/bin/python",
        "/src/tools/task_eval.py",
        "prepare-ci-dataset",
    ]


def test_nightly_validates_the_installed_wheel_without_a_second_model_build() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "nightly.yml").read_text()
    package = text.split("\n  package:", maxsplit=1)[1].split("\n  model-proof:", maxsplit=1)[0]
    release = text.split("\n  release:", maxsplit=1)[1]
    package_index = package.index("Build trtmc pip package")
    installed_wheel_index = package.index("Python package coverage")
    upload_index = package.index("Upload trtmc pip package artifact")
    assert package_index < installed_wheel_index < upload_index
    assert "python3 -m tools.ci stage python-builder" in package
    assert "python3 -m tools.ci stage wheel-model-smoke" not in package
    assert "Model smoke test from trtmc pip package" not in package
    ci_script = _ci_source("pipeline.py", "package.py")
    assert "_verify_wheel_runtime" in ci_script
    assert "verify_installed" in ci_script
    assert "needs.required.result == 'success'" in release
    assert "github.event_name == 'schedule' || github.ref == 'refs/heads/main'" in release
    assert "Download certified trtmc pip package" in release
    assert "pattern: trtmc-pip-package-${{ github.run_id }}-*" in release
    assert "merge-multiple: false" in release
    assert "Select latest certified package attempt" in release
    assert "tools/select_latest_attempt_artifact.py" in release
    assert 'for asset in "$WHEEL_DIR"/*.whl' in release
    assert "Publish trtmc pip package to GitHub Release" in release
    assert "target_commitish" in release
    assert 'os.environ["TESTED_SHA"]' in release


def test_nightly_uses_manylinux_image_and_builds_wheel_first() -> None:
    text = (REPO_ROOT / ".github/workflows/nightly.yml").read_text()
    assert "TRTMC_CI_IMAGE:" in text
    assert "vars.TRTMC_MANYLINUX_CI_IMAGE" in text
    assert "vars.TRTMC_CI_IMAGE" not in text
    assert "trtmc-dev-gb300:manylinux_2_39" in text
    assert "TRTMC_PACKAGE_WHEEL_ARCH:" in text
    assert "manylinux_2_39_aarch64" in text
    assert "TRTMC_PACKAGE_CI_IMAGE" not in text
    package = text.split("\n  package:", maxsplit=1)[1].split("\n  model-proof:", maxsplit=1)[0]
    assert package.index("Start package container") < package.index("Build trtmc pip package")
    assert package.index("Build trtmc pip package") < package.index("Python package coverage")
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
    assert premerge.count("python3 -m tools.ci stage") == 1
    assert "python3 -m tools.ci stage premerge-unit" in premerge
    assert "uses: ./.github/workflows/model-proof.yml" in premerge


def test_premerge_unit_stage_builds_no_model_plugins_or_native_wheel() -> None:
    script = _ci_source("quality.py")
    stage = script.split("def premerge", maxsplit=1)[1].split("def _premerge_scope", maxsplit=1)[0]
    cmake = (REPO_ROOT / "CMakeLists.txt").read_text()

    assert "pip install" not in stage
    assert "source / 'python'" in stage
    assert '"TRTMC_CI_SCRATCH_DIR", "/tmp"' in stage
    assert "TRTMC_PREMERGE_UNIT_BUILD_DIR" in stage
    assert '"not gpu and not trt and not e2e and not model_proof_allocator"' in stage
    assert "tests/builder/" in stage
    assert "tests/tools/" in stage
    assert 'glob("test_*.py")' in script
    assert '"-q"' in stage and '"-x"' in stage
    assert '"--dist=worksteal"' in stage
    assert 'not model_proof_allocator"' in stage
    assert '"-m"' in stage and '"model_proof_allocator"' in stage
    assert '["trtmc", "test_cli_args", "test_config_cli_support"]' in script
    assert '["trtmc", "trtmc_platform_cpp_tests"]' in script
    assert '"TRTMC_PREMERGE_UNIT_SCOPE", "all"' in stage
    assert "tests/builder/test_cli.py" in script
    assert 'if scope == "builder"' in script
    assert '["tests/builder/"]' in script
    assert "if native_targets:" in stage
    assert '[build / "trtmc", "version"]' in stage
    assert '[build / "trtmc", "--help"]' in stage
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


def test_builder_unit_scope_runs_python_without_native_build(tmp_path: Path) -> None:
    class RecordingContext:
        repository = tmp_path
        env = {
            "GITHUB_WORKSPACE": str(tmp_path),
            "TRTMC_PREMERGE_UNIT_SCOPE": "builder",
            "TRTMC_UNIT_BUILD_JOBS": "8",
            "TRTMC_UNIT_TEST_JOBS": "8",
        }

        def __init__(self) -> None:
            self.commands: list[list[object]] = []

        def positive_integer(self, value: str, _name: str) -> int:
            return int(value)

        def run(self, command: list[object], **_kwargs: object) -> subprocess.CompletedProcess:
            self.commands.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    context = RecordingContext()
    UnitTestRunner(context).premerge()

    pytest_commands = [
        command
        for command in context.commands
        if command[:3] == ["python", "-m", "pytest"]
    ]
    assert len(pytest_commands) == 1
    assert "tests/builder/" in pytest_commands[0]
    assert not [
        command for command in context.commands if command[0] in {"cmake", "ctest"}
    ]


def test_premerge_unit_container_is_unprivileged_offline_and_cpu_only() -> None:
    start = _ci_source("container.py", "environment.py")
    workflow = (REPO_ROOT / ".github" / "workflows" / "trtmc-ci.yml").read_text()
    unit_tests = workflow.split("\n  unit-tests:", maxsplit=1)[1].split(
        "\n  model-proof:", maxsplit=1
    )[0]

    for option in (
        '"--network"',
        '"none"',
        "--read-only",
        "/tmp:rw,exec,nosuid,nodev,size=16g",
        "--cap-drop",
        "--security-opt",
        "no-new-privileges",
        'f"{os.getuid()}:{os.getgid()}"',
        "--ipc",
        "HOME=/tmp",
        "USER=trtmc-ci",
        "LOGNAME=trtmc-ci",
        "TMPDIR=/work/tmp",
        "TEMP=/work/tmp",
        "TMP=/work/tmp",
        "XDG_CACHE_HOME=/work/cache",
        "TORCHINDUCTOR_CACHE_DIR=/work/torch-cache",
        "PIP_NO_INDEX=1",
        "TRTMC_CI_SCRATCH_DIR=/work",
        "NVIDIA_VISIBLE_DEVICES=void",
        "CUDA_VISIBLE_DEVICES=",
        "--runtime",
        'f"{mount}:ro"',
        'f"{scratch}:/work"',
    ):
        assert option in start
    assert "if self.config.hardened" in start
    assert "if not self.config.hardened" in start
    assert "Path('/dev').glob('nvidia*')" in start
    assert "Hardened unit scratch must be inside RUNNER_TEMP" in start
    assert "Hardened unit scratch must not be a symlink" in start
    assert 'TRTMC_CI_HARDENED: "true"' in unit_tests
    assert "--gpus" not in unit_tests
    assert "/workspace/users/yifeif:/workspace/users/yifeif" not in unit_tests
    assert 'env.get("TRTMC_CI_WORKSPACE")' in start
    assert 'env.get("GITHUB_WORKSPACE", "")' in start
    assert 'mount = f"{self.config.workspace}:{self.config.workspace}"' in start
    common = start.split("COMMON_ENVIRONMENT =", maxsplit=1)[1].split(
        "TRUSTED_ENVIRONMENT =", maxsplit=1
    )[0]
    assert "TRTMC_PREMERGE_UNIT_SCOPE" in common
    assert "HF_TOKEN" not in common
    assert "HUGGING_FACE_HUB_TOKEN" not in common

    stage = _ci_source("stage.py")
    assert "COMMON_ENVIRONMENT if self.config.hardened else TRUSTED_ENVIRONMENT" in stage


def test_unowned_gpu_only_builder_suites_are_excluded_from_cpu_units() -> None:
    stage = _ci_source("quality.py")
    for relative in (
        "tests/builder/test_flashinfer_benchmark.py",
        "tests/builder/test_tvm_ffi_plugin.py",
    ):
        assert f"--ignore={relative}" in stage

    ffi_architecture = (REPO_ROOT / "tests/builder/test_ffi_architecture.py").read_text()
    flashinfer_section = ffi_architecture.split("class TestFlashInferKernelSetup:", maxsplit=1)[
        1
    ].split("class TestEngineBuilderKernelArtifacts:", maxsplit=1)[0]
    assert flashinfer_section.count("@pytest.mark.gpu") == 3
    assert flashinfer_section.count("@pytest.mark.trt") == 3


def test_model_proof_runs_one_isolated_model_with_unique_complete_evidence() -> None:
    proof = (REPO_ROOT / ".github/workflows/model-proof.yml").read_text()
    runner = _ci_source("model_proof.py", "model_proof_inner.py")

    assert "TRTMC_CI_IMAGE:" in proof
    assert "vars.TRTMC_MANYLINUX_CI_IMAGE" in proof
    assert "trtmc-dev-gb300:manylinux_2_39" in proof
    assert "TRTMC_CI_IMAGE_LOCK_FILE:" not in proof
    image_manager = _ci_source("docker_image.py")
    assert 'env.get("TRTMC_CI_IMAGE_LOCK_FILE", "/tmp/trtmc-ci-docker-image.lock")' in image_manager
    assert "Ensure CI Docker image" in proof
    assert "python3 -m tools.ci image ensure" in proof
    assert proof.count("actions/checkout@v4") == 1
    assert proof.count("python3 -m tools.ci image ensure") == 1
    assert "TRTMC_HF_CACHE:" in proof
    assert "TRTMC_HF_HUB_CACHE:" in proof
    assert "TRTMC_HF_MODULES_CACHE:" not in proof
    assert "TRTMC_MODEL_PROOF_BUILD_JOBS: ${{ vars.TRTMC_MODEL_PROOF_BUILD_JOBS || '2' }}" in proof
    assert "TRTMC_MODEL_PROOF_SLOTS_PER_GPU:" in proof
    assert "vars.TRTMC_MODEL_PROOF_SLOTS_PER_GPU || '4'" in proof
    assert "TRTMC_MODEL_PROOF_GPU_LOCK_DIR:" in proof
    assert "/tmp/trtmc-model-proof-gpu-locks" in proof
    assert "python3 -m tools.ci model-proof" in proof
    assert "run-model-proof-batch.sh" not in proof
    assert "env -u TRTMC_GPU_ID python3 -m tools.ci model-proof" in proof
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
    assert '"HF_MODULES_CACHE": "/work/hf-modules"' in runner
    assert '"--network"' in runner and '"none"' in runner
    assert "--read-only" in runner
    assert "proof.json" in runner


def test_model_proof_keeps_long_lived_scratch_out_of_runner_temp() -> None:
    text = (REPO_ROOT / ".github/workflows/model-proof.yml").read_text()
    bootstrap = text.split("      - name: Bootstrap model HTML before checkout", maxsplit=1)[
        1
    ].split("      - name: Check model proof disk headroom", maxsplit=1)[0]
    disk = text.split("      - name: Check model proof disk headroom", maxsplit=1)[1].split(
        "      - name: Check out exact source revision", maxsplit=1
    )[0]
    checkout = text.split("      - name: Check out exact source revision", maxsplit=1)[1].split(
        "      - name: Log in to GitHub Container Registry", maxsplit=1
    )[0]
    run_proof = text.split("      - name: Run isolated model proof", maxsplit=1)[1].split(
        "      - name: Reconcile model proof containers", maxsplit=1
    )[0]
    finalize = text.split("      - name: Finalize model proof fallback", maxsplit=1)[1].split(
        "      - name: Upload isolated model proof artifact", maxsplit=1
    )[0]
    upload = text.split("      - name: Upload isolated model proof artifact", maxsplit=1)[1].split(
        "      - name: Enforce isolated model certification", maxsplit=1
    )[0]
    cleanup = text.split("      - name: Clean model proof scratch space", maxsplit=1)[1]
    durable = (
        "${{ github.workspace }}/model-proof-output-${{ github.run_id }}-"
        "${{ github.run_attempt }}-${{ inputs.model }}"
    )

    assert f"MODEL_PROOF_OUTPUT_DIR: {durable}" in bootstrap
    assert bootstrap.index('[[ "$MODEL" =~ ^[a-z0-9][a-z0-9._-]*$ ]]') < bootstrap.index(
        'mkdir -p "$MODEL_PROOF_OUTPUT_DIR/artifacts"'
    )
    assert "path: model-proof-source" in checkout
    assert "model-proof-source" not in durable
    assert "${{ runner.temp }}/model-proof-" not in text
    assert '"$RUNNER_TEMP"/model-proof-' not in text
    assert "-name 'model-proof-output-*'" in disk
    assert 'df -Pk "$GITHUB_WORKSPACE"' in disk
    assert "GITHUB_OUTPUT" not in run_proof
    assert "steps.proof.outputs.exit_code" not in finalize
    assert "proof-exit-code.txt" in run_proof
    assert "proof-exit-code.txt" in finalize
    assert f"path: {durable}/artifacts/" in upload
    assert '"$GITHUB_WORKSPACE"/model-proof-output-*' in cleanup
    assert 'df -h "$GITHUB_WORKSPACE"' in cleanup


def test_package_stage_builds_py310_and_py312_wheels() -> None:
    text = _ci_source("package.py", "pipeline.py")
    assert '"TRTMC_PACKAGE_PYTHON_TAGS", "py310 py312"' in text
    assert '"WHEEL_PYVER": tag' in text
    assert '"build"' in text and '"--wheel"' in text and '"--outdir"' in text
    assert 'f"build-dir={tag_root}"' in text
    assert "manylinux_2_39_aarch64" in text
    assert '"wheel-model-smoke":' in text
    assert "Model smoke test from trtmc pip package" in text


def test_package_reuses_conan_cmake_build_directory(tmp_path: Path) -> None:
    from tools.ci.package import WheelPackageManager

    release = tmp_path / "conan_out" / "build" / "Release"
    release.mkdir(parents=True)
    (release / "CMakeCache.txt").touch()

    assert WheelPackageManager(object())._conan_cmake_build_dir(
        tmp_path / "conan_out"
    ) == release


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
        REPO_ROOT / "tools" / "ci" / "stage.py",
        REPO_ROOT / "tools" / "ci" / "container.py",
        REPO_ROOT / "tools" / "ci" / "package.py",
        REPO_ROOT / "tools" / "ci" / "pipeline.py",
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
    text = _ci_source("package.py")
    assert '"TRTMC_PACKAGE_WHEEL_ARCH", "manylinux_2_39_aarch64"' in text
    assert "self.platform = platform" in text
    assert "native wheel must not contain .data/purelib entries" in text
    assert ".data/scripts/trtmc" in text
    assert "native trtmc must be installed directly, not via console_scripts" in text
    assert '"auditwheel>=6.2"' in text
    assert 'sys.executable, "-m", "auditwheel", "show", wheel' in text
    assert 'f"*-{tag}-none-{platform}.whl"' in text
    assert "_validate_build_platform" in text
    assert "build_glibc" in text


def test_package_stage_uses_conan_py_build_inputs() -> None:
    text = _ci_source("package.py")
    assert '"CONAN_PY_BUILD_PROFILE_AUTODETECT": "1"' in text
    assert '"TRTMC_TRT_INCLUDE_DIR": trt_include' in text
    assert '"TRTMC_TRT_LIBRARY": trt_library' in text
    assert '"TRTMC_CUDA_INCLUDE_DIR": cuda_include' in text
    assert '"TRTMC_CUDART_LIBRARY": cudart' in text


def test_impact_stage_reuses_cached_json_for_summary() -> None:
    text = _ci_source("quality.py")
    assert 'arguments.append("--json")' in text
    assert "write_text(result.stdout" in text
    assert "ImpactResult(**impact)" in text
    assert '"--verbose"' not in text


def test_python_builder_fallback_is_per_tier() -> None:
    script = _ci_source("coverage.py")
    assert 'if {"builder", "tools"}.issubset(fallback)' in script
    assert '["tests/builder/"] if "builder" in fallback' in script
    assert 'if "tools" in fallback' in script
    assert 'add(["tests/tools/"])' in script
    assert 'glob("test_*.py")' in script


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
    script = _ci_source("package.py")
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()

    assert "set_target_properties(trtmc PROPERTIES" in cmake
    assert "BUILD_RPATH_USE_ORIGIN TRUE" in cmake
    assert 'INSTALL_RPATH "\\$ORIGIN"' in cmake
    assert '"libtrtmc_core.so*"' in conanfile
    assert "for destination in (package_bin, wheel_data_scripts):" in conanfile
    assert "TRTMC core DSO was not staged beside the wheel script" in conanfile
    assert "apache-tvm-ffi==0.1.12" in pyproject
    assert "script_cores" in script
    assert 'if "$ORIGIN" not in dynamic' in script
    assert "installed trtmc RUNPATH leaks the CI build directory" in script


def test_ci_source_build_defaults_to_packaged_libtorch_mode() -> None:
    conanfile = (REPO_ROOT / "conanfile.py").read_text()
    wrapper = _ci_source("environment.py")
    coverage = _ci_source("coverage.py")
    assert 'toolchain.cache_variables["TRTMC_ENABLE_LIBTORCH_MULTINOMIAL"] = False' in conanfile
    assert "self._value('TRTMC_ENABLE_LIBTORCH_MULTINOMIAL', 'OFF')" in coverage
    assert '"-DTRTMC_ENABLE_LIBTORCH_MULTINOMIAL="' in coverage
    assert "TRTMC_ENABLE_LIBTORCH_MULTINOMIAL" in wrapper


def test_ci_cpp_test_build_reuses_wheel_conan_tree() -> None:
    script = _ci_source("quality.py", "package.py")
    assert '"TRTMC_CONAN_ENABLE_TEST_TARGETS": "1"' in script
    assert '"TRTMC_CONAN_BUILD_TARGETS": "\\n".join(targets)' in script
    assert '"conan", "build", ".", "-of", metadata["conan_out_dir"]' in script
    assert 'arguments = ["ctest", "--test-dir", build_dir]' in script
    assert "build_metadata" in script


def test_selective_e2e_builds_and_runs_single_family_source_projections() -> None:
    selective = _ci_source("e2e.py")
    group_runner = _ci_source("isolation.py")
    script = selective + group_runner

    assert '"tools/model_plugin_isolation.py"' in group_runner
    assert '"plan"' in group_runner
    assert "tools/model_plugin_isolation.py" in selective
    assert "IsolatedModelRunner" in selective
    assert "impact-models" in selective
    assert "e2e_isolation_models.txt" in selective
    assert "E2EParallelRunner" in selective
    assert '"--exclude-ci-tier"' in selective
    assert '"nightly_only"' in selective
    assert '"multi_device"' in selective
    assert "if standard_rc" in selective
    assert "strict model-owned isolation E2E" in selective
    assert "_prepare_plugins" in selective
    assert '"tools/model_plugin_isolation.py"' in group_runner
    assert '"schedule"' in group_runner
    assert "_run_queue" in group_runner
    assert "concurrent.futures" in group_runner
    assert '"stage-source"' in group_runner
    assert "def _configure" in group_runner
    assert "CMAKE_TOOLCHAIN_FILE" in script
    assert "FETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON" in script
    assert 'str(group["runtime_plugin"]["target"])' in group_runner
    assert '"PYTHONPATH": f"{source / \'python\'}:{source}"' in group_runner
    assert '"LD_LIBRARY_PATH": ":".join(library_path)' in group_runner
    assert '"--trtmc-binary"' in group_runner and 'build / "trtmc"' in group_runner
    assert '"--engine-dir"' in group_runner and "engines" in group_runner
    assert '"--model-plugin-dir"' in group_runner and "plugins" in group_runner
    assert '"CUDA_VISIBLE_DEVICES": str(gpu)' in group_runner
    assert '"--rootdir"' in group_runner and "source" in group_runner
    assert '"--e2e-models-file"' in group_runner and "models" in group_runner
    assert '"--e2e-exclude-ci-tier"' in group_runner and '"nightly_only"' in group_runner
    assert '"SELECTIVE_E2E_GROUP_TIMEOUT", "90m"' in group_runner
    assert "for model in selected" in group_runner
    assert "verify-results" in group_runner
    assert "expected exactly 1" in group_runner
    assert "prepare_model_plugin_dir" not in group_runner


def test_full_e2e_stages_all_runtime_plugins_from_reusable_build() -> None:
    script = _ci_source("e2e.py")
    assert 'self._prepare_plugins(plugins, ["--all"])' in script
    assert '"--model-plugin-dir"' in script


def test_cpp_coverage_builds_excluded_test_target() -> None:
    coverage = _ci_source("coverage.py")
    assert '"CPP_COVERAGE_BUILD_TARGET", "trtmc_cpp_tests"' in coverage
    assert '["cmake", "--build", build_dir, "--target", build_target, *parallel]' in coverage
    assert "CommandRunner(cwd=build_dir" in coverage


def test_cpp_coverage_gate_excludes_model_owned_runtime_plugins() -> None:
    coverage = _ci_source("coverage.py")
    assert 'self._words("GCOVR_EXCLUDES")' in coverage
    assert 'str(self.repository / "src/runtime/models")' in coverage
    assert 'gcovr_base.extend(("--exclude", value))' in coverage


def test_cpp_coverage_engine_runs_tools_directly_without_shell(tmp_path: Path, monkeypatch) -> None:
    from tools.ci.coverage import CppCoverageEngine

    commands: list[list[str]] = []
    gcovr_commands: list[list[str]] = []

    class Context:
        repository = tmp_path
        env = {"PATH": os.environ["PATH"], "BUILD_DIR": "relative-build"}

        @staticmethod
        def executable(name: str) -> str:
            return name

        @staticmethod
        def output(command: list[str]) -> str:
            assert command == ["gcovr", "--help"]
            return "--fail-under-function"

        @staticmethod
        def run(command, **_kwargs):
            commands.append([str(item) for item in command])
            return subprocess.CompletedProcess(command, 0, "", "")

    def fake_gcovr(_runner, command, **_kwargs):
        rendered = [str(item) for item in command]
        gcovr_commands.append(rendered)
        stdout = "lines: 100.0%\nfunctions: 100.0%\nbranches: 100.0%\n"
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr("tools.ci.coverage.CommandRunner.run", fake_gcovr)
    report = tmp_path / "reports"
    result = CppCoverageEngine(Context(), report).run(
        ["-L", "platform"],
        build_target="trtmc_platform_cpp_tests",
        limit=None,
    )

    assert result == 0
    assert [
        "cmake",
        "--build",
        str(tmp_path / "relative-build"),
        "--target",
        "trtmc_platform_cpp_tests",
        "--parallel",
    ] in commands
    assert [
        "ctest",
        "--test-dir",
        str(tmp_path / "relative-build"),
        "--output-on-failure",
        "-L",
        "platform",
    ] in commands
    assert len(gcovr_commands) == 3
    assert all(
        "--gcov-ignore-parse-errors" in command
        and command[command.index("--gcov-ignore-parse-errors") + 1]
        == "negative_hits.warn_once_per_file"
        for command in gcovr_commands
    )
    gate = gcovr_commands[-1]
    assert gate[gate.index("--fail-under-line") + 1] == "100"
    assert gate[gate.index("--fail-under-function") + 1] == "100"
    assert gate[gate.index("--fail-under-branch") + 1] == "100"
    assert all(command[0] != "bash" for command in commands + gcovr_commands)
    assert (report / "cpp-coverage-summary.txt").is_file()


def test_python_coverage_engine_runs_coverage_directly(tmp_path: Path) -> None:
    from tools.ci.coverage import PythonCoverageEngine

    commands: list[list[str]] = []

    class Context:
        repository = tmp_path
        env = {"PATH": os.environ["PATH"]}

        @staticmethod
        def executable(name: str) -> str:
            return name

        @staticmethod
        def run(command, **_kwargs):
            rendered = [str(item) for item in command]
            commands.append(rendered)
            stdout = ""
            if rendered[2:4] == ["coverage", "report"]:
                stdout = "TOTAL 10 0 100%\n"
            if rendered[2:4] == ["coverage", "xml"]:
                destination = Path(rendered[rendered.index("-o") + 1])
                destination.write_text(
                    '<coverage line-rate="1.0" branch-rate="1.0"/>\n',
                    encoding="utf-8",
                )
            return subprocess.CompletedProcess(command, 0, stdout, "")

    report = tmp_path / "reports"
    PythonCoverageEngine(Context(), report).run(["-q"])

    coverage_run = next(command for command in commands if command[2:4] == ["coverage", "run"])
    assert coverage_run[-3:] == ["tests/builder", "tests/tools", "-q"]
    assert all(command[0] != "bash" for command in commands)
    assert (report / "python-cobertura.xml").is_file()
    assert (report / "python-coverage.txt").read_text() == "TOTAL 10 0 100%\n"


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
    text = _ci_source("package.py")
    smoke_block = text.split("def model_smoke", maxsplit=1)[1].split(
        "def _clean_venv_smoke", maxsplit=1
    )[0]
    assert 'self.select_wheel("py312")' in smoke_block
    assert "sys.version_info[:2] != (3, 12)" in smoke_block
    assert "TRTMC_WHEEL_SMOKE_PYTHON" not in smoke_block
    assert "select_compatible_wheel" not in smoke_block
    assert '"PATH"' not in smoke_block
    assert 'trtmc,\n                "build"' in smoke_block
    assert "InstalledWheelValidator.require_elf(trtmc)" in smoke_block


def test_selective_e2e_zero_model_path_still_generates_report_input_dir() -> None:
    text = _ci_source("e2e.py")
    zero_model_block = text.split(
        'print("No E2E models affected by this change -- skipping E2E tests")',
        maxsplit=1,
    )[1].split("return", maxsplit=1)[0]
    assert '"e2e_artifacts/artifacts"' in zero_model_block
    assert "mkdir(parents=True, exist_ok=True)" in zero_model_block
