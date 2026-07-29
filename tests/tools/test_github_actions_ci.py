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


def test_source_workflow_inventory_does_not_repeat_premerge_after_merge() -> None:
    workflows = REPO_ROOT / ".github" / "workflows"
    workflow_files = {
        *workflows.glob("*.yml"),
        *workflows.glob("*.yaml"),
    }
    assert sorted(path.name for path in workflow_files) == [
        "internal-ci-bridge.yml",
        "legal.yml",
        "pages.yml",
    ]

    bridge = (workflows / "internal-ci-bridge.yml").read_text(encoding="utf-8")
    legal = (workflows / "legal.yml").read_text(encoding="utf-8")
    pages = (workflows / "pages.yml").read_text(encoding="utf-8")

    for forbidden_trigger in (
        "push",
        "schedule",
        "workflow_run",
        "repository_dispatch",
    ):
        assert f"\n  {forbidden_trigger}:" not in bridge
    assert "\n  push:" not in legal
    assert "workflow_call:" in legal
    assert "workflow_dispatch:" in legal

    assert "\n  push:" in pages
    assert '      - "website/**"' in pages
    assert "python3 -m tools.ci" not in legal + pages
    assert "actions/workflows/premerge.yml/dispatches" not in legal + pages


def test_internal_ci_bridge_only_dispatches_an_exact_trusted_head() -> None:
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "internal-ci-bridge.yml"
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
    assert 'test "$head_sha" = "$EVENT_HEAD_SHA"' in authorize
    assert '[[ "$head_sha" =~ ^[0-9a-f]{40}$ ]]' in authorize
    assert 'echo "head_sha=$head_sha"' in authorize
    assert "pr_number=$PR_NUMBER" in authorize
    for legacy in (
        "base_sha",
        "merge_sha",
        "BASE_SHA",
        "MERGE_SHA",
        "EVENT_BASE_SHA",
    ):
        assert legacy not in workflow
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
    assert (
        "/repos/$PRIVATE_CI_OWNER/$PRIVATE_CI_REPOSITORY/"
        "actions/workflows/premerge.yml/dispatches"
    ) in workflow
    assert re.search(r'"/repos/[A-Za-z0-9]', workflow) is None
    assert '[[ "$PRIVATE_CI_OWNER" =~ ^[A-Za-z0-9]' in workflow
    assert '[[ "$PRIVATE_CI_REPOSITORY" =~ ^[A-Za-z0-9]' in workflow
    assert 'echo "$PRIVATE_CI_' not in dispatch
    assert "GITHUB_STEP_SUMMARY" not in dispatch

    assert (
        dispatch.count("HEAD_SHA: ${{ needs.authorize.outputs.head_sha }}") == 3
    )
    assert "/repos/$GITHUB_REPOSITORY/statuses/$HEAD_SHA" in dispatch
    assert "/repos/$GITHUB_REPOSITORY/statuses/$MERGE_SHA" not in dispatch
    assert "-f state=pending" in dispatch
    assert "-f state=failure" in dispatch
    assert dispatch.count("-f context=trtmc/premerge/required") == 2
    assert '-f description="PENDING: TensorRT-Model-Connect premerge"' in dispatch
    assert (
        '-f description="FAIL: TensorRT-Model-Connect premerge dispatch"'
    ) in dispatch
    assert dispatch.count(
        '-f target_url="https://github.com/$GITHUB_REPOSITORY/tree/$HEAD_SHA/tests"'
    ) == 2
    assert dispatch.index("/statuses/$HEAD_SHA") < dispatch.index(
        "actions/workflows/premerge.yml/dispatches"
    )

    assert 'ref: "main"' in dispatch
    for name in ("pr_number", "head_sha"):
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


def test_source_quality_pipeline_keeps_the_full_static_gate() -> None:
    source = _ci_source("pipeline.py", "quality.py")
    source_quality = source.split('"source-quality":', maxsplit=1)[1].split(
        '"cpp-unit":', maxsplit=1
    )[0]

    assert '"Check cyclomatic complexity"' in source_quality
    assert "self.quality.complexity" in source_quality
    assert '"Lint changed files"' in source_quality
    assert "self.quality.lint_changed_files" in source_quality
    assert '"Check model architecture contracts"' in source_quality
    assert "self.quality.architecture_contracts" in source_quality

    architecture_contract = source.split(
        "def architecture_contracts", maxsplit=1
    )[1].split("def _changed_files", maxsplit=1)[0]
    assert '"pytest"' in architecture_contract
    assert (
        "tests/tools/test_model_plugin_encapsulation_static.py"
        in architecture_contract
    )
    assert '"-q"' in architecture_contract
    assert '"no:cacheprovider"' in architecture_contract


def test_source_tensorrt_install_contract_remains_self_contained() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "ENV TRT_ROOT=" not in dockerfile
    assert "ENV PIP_FIND_LINKS=" not in dockerfile
    assert (
        "ENV TRT_LIB_DIR=/opt/venv/lib/python3.12/site-packages/tensorrt_libs"
        in dockerfile
    )
    assert "ENV TRT_INC_DIR=/usr/include/aarch64-linux-gnu" in dockerfile

    package = _ci_source("package.py")
    assert package.count("_install_tensorrt_sdk") >= 2


def test_hardened_unit_container_is_unprivileged_offline_and_cpu_only() -> None:
    source = _ci_source("container.py", "environment.py")
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
        assert option in source
    assert "if self.config.hardened" in source
    assert "if not self.config.hardened" in source
    assert "Path('/dev').glob('nvidia*')" in source
    assert "Hardened unit scratch must be inside RUNNER_TEMP" in source
    assert "Hardened unit scratch must not be a symlink" in source
    assert 'env.get("TRTMC_CI_WORKSPACE")' in source
    assert 'env.get("GITHUB_WORKSPACE", "")' in source
    assert 'mount = f"{self.config.workspace}:{self.config.workspace}"' in source

    common = source.split("COMMON_ENVIRONMENT =", maxsplit=1)[1].split(
        "TRUSTED_ENVIRONMENT =", maxsplit=1
    )[0]
    assert "TRTMC_PREMERGE_UNIT_SCOPE" in common
    assert "HF_TOKEN" not in common
    assert "HUGGING_FACE_HUB_TOKEN" not in common

    stage = _ci_source("stage.py")
    assert (
        "COMMON_ENVIRONMENT if self.config.hardened else TRUSTED_ENVIRONMENT"
        in stage
    )




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


def test_full_python_builder_preserves_parallel_and_allocator_coverage() -> None:
    text = _ci_source("coverage.py")
    builder = text.split("def python_builder_tests", maxsplit=1)[1].split(
        "def cpp", maxsplit=1
    )[0]
    builder_conftest = (
        REPO_ROOT / "tests" / "builder" / "conftest.py"
    ).read_text(encoding="utf-8")

    assert 'glob("test_*.py")' in builder
    assert "--ignore=tests/builder/test_cli.py" not in builder
    assert '"-n", "auto"' in builder
    assert '"not model_proof_allocator and not gpu and not trt"' in builder
    assert '"tests/tools/test_model_proof_runner.py"' in builder
    assert '"model_proof_allocator"' in builder
    assert '"--cov-append"' in builder
    assert '"TRTMC_TEST_INSTALLED_WHEEL": "1"' in builder
    assert "source_pkgs =" in text
    assert "tensorrt_model_connect" in text
    assert (
        'os.environ.get("TRTMC_TEST_INSTALLED_WHEEL") == "1"'
        in builder_conftest
    )
    assert "imported tensorrt_model_connect" in builder_conftest
    assert builder.index('"-n", "auto"') < builder.index(
        '"tests/tools/test_model_proof_runner.py"'
    )


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




def test_source_quality_lint_uses_resolved_ci_base_ref() -> None:
    text = _ci_source("quality.py")
    assert "f\"origin/{self.context.env.get('GITHUB_REF_NAME', 'main')}\"" in text




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


def test_etth1_model_proofs_use_the_single_validation_engine_entry_point() -> None:
    stage = _ci_source("validation.py", "model_proof.py", "model_proof_inner.py")
    validation_engine = (
        REPO_ROOT / "tools" / "validation" / "engine.py"
    ).read_text()

    assert '"/src/tools/validation/engine.py"' in stage
    assert '"prepare-ci-dataset"' in stage
    assert '"eval"' in stage
    assert "validation_engine_ci.py" not in stage
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
    assert "ETTh1 validation requires a GB300 GPU" in stage
    assert '"--network"' in stage and '"none"' in stage
    assert "validate_eval_summary" in validation_engine
    assert 'result.get("status") == "passed"' in validation_engine
    assert "return complete and all" in validation_engine
    assert (
        '"work_dir"'
        not in validation_engine.split("def _public_ci_result", maxsplit=1)[1].split(
            ")", maxsplit=1
        )[0]
    )


def test_etth1_dataset_preparation_imports_the_projected_python_package(
    tmp_path: Path,
) -> None:
    from tools.ci.validation import ValidationDatasetPreparer

    projection = tmp_path / "projection"
    work = tmp_path / "work"
    artifacts = tmp_path / "artifacts"
    destination = work / "validation-data" / ValidationDatasetPreparer.DATASET
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

    result = ValidationDatasetPreparer(
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
        "/src/tools/validation/engine.py",
        "prepare-ci-dataset",
    ]
