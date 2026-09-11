# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contributor-visible Community CI entrypoint."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from tools import community_ci, legal_headers


REPO_ROOT = Path(__file__).resolve().parents[2]


def _workflow_step_script(workflow_name: str, job_name: str, step_name: str) -> str:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / workflow_name).read_text(encoding="utf-8")
    )
    return next(
        step["run"] for step in workflow["jobs"][job_name]["steps"] if step["name"] == step_name
    )


def test_pre_commit_config_installs_only_lightweight_commit_hooks() -> None:
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    assert "default_install_hook_types" not in config

    repositories = {repository["repo"]: repository for repository in config["repos"]}
    assert repositories["https://github.com/astral-sh/ruff-pre-commit"]["rev"] == "v0.16.4"
    assert repositories["https://github.com/pre-commit/mirrors-clang-format"]["rev"] == "v22.1.8"

    hooks = {hook["id"]: hook for repository in config["repos"] for hook in repository["hooks"]}
    for hook_id in ("trailing-whitespace", "end-of-file-fixer", "check-yaml"):
        assert hooks[hook_id]["stages"] == ["pre-commit"]
    assert hooks["ruff-check"]["stages"] == ["pre-commit"]
    assert hooks["clang-format"]["stages"] == ["pre-commit"]
    assert hooks["clang-format"]["entry"] == "clang-format --dry-run --Werror"
    assert all(hook["stages"] == ["pre-commit"] for hook in hooks.values())

    source = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "python3 -m tools.community_ci format-" not in source
    assert "pre-push" not in source


def test_contributor_guide_matches_the_live_ci_flow() -> None:
    path = REPO_ROOT / "CONTRIBUTING.md"
    source = path.read_text(encoding="utf-8")
    ordered_markers = [
        "pre-commit install --install-hooks",
        "git commit --signoff",
        "git push --set-upstream origin",
        "Community CPU / Required",
        "Community GPU",
        "run-internal-ci",
        "TRTMC Internal CI / Automated premerge gate",
    ]

    positions = [source.index(marker) for marker in ordered_markers]
    assert positions == sorted(positions)
    for marker in (
        "automatically",
        "GitHub-hosted",
        "ubuntu-24.04",
        "read-only repository permission",
        "no access to private",
        "runners, secrets, or",
        "GPUs",
        "Only after `Community CPU / Required` passes",
        "execution is temporarily disabled by repository policy",
        "Community GPU is not a merge",
        "pull-request code executes only on the",
        "isolated GPU instance",
        "pull-request checks",
        "public Actions logs",
        "py -3 -m pip",
    ):
        assert marker in source
    assert "/run-ci" not in source
    assert "status comment" not in source


def test_website_contributing_page_points_to_the_canonical_guide() -> None:
    source = (REPO_ROOT / "website/docs/extend/contributing.md").read_text(encoding="utf-8")
    assert "CONTRIBUTING.md" in source


def test_impact_publishes_only_the_public_cpu_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    github_output = tmp_path / "github-output"
    github_summary = tmp_path / "github-summary"
    runner = community_ci.CommunityCI(
        REPO_ROOT,
        {
            **os.environ,
            "GITHUB_OUTPUT": str(github_output),
            "GITHUB_STEP_SUMMARY": str(github_summary),
        },
    )
    monkeypatch.setattr(runner, "resolve_base", lambda _base: "base-sha")
    monkeypatch.setattr(community_ci.test_impact, "validate", lambda _repo: None)
    monkeypatch.setattr(
        community_ci.test_impact,
        "changed_files",
        lambda *_args: ["families/qwen/model.py"],
    )
    monkeypatch.setattr(
        community_ci.test_impact,
        "classify",
        lambda *_args: community_ci.test_impact.Impact(
            scope="families",
            families=("qwen",),
            changed_files=("families/qwen/model.py",),
            run_core_tests=True,
            run_docs=False,
        ),
    )

    result = runner.impact(None)

    assert result["families"] == ["qwen"]
    assert github_output.read_text(encoding="utf-8") == 'families=["qwen"]\n'
    summary = github_summary.read_text(encoding="utf-8")
    assert "families/qwen/model.py" in summary


@pytest.mark.parametrize("change", ["valid", "missing-header", "LICENSE", "NOTICE"])
def test_public_source_quality_enforces_legal_compliance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    change: str,
) -> None:
    header = legal_headers.HASH_STYLE.render(b"\n").decode() + "\n\n"
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    shutil.copyfile(REPO_ROOT / "tools/legal_headers.py", tools_dir / "legal_headers.py")
    (tools_dir / "legal_header_exceptions.toml").write_text(
        header + "schema_version = 1\n", encoding="utf-8"
    )
    for name in ("LICENSE", "NOTICE"):
        (tmp_path / name).write_text("Original legal document.\n", encoding="utf-8")

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", *arguments],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "--quiet")
    git("add", ".")
    git("commit", "--quiet", "-m", "Base fixture")
    base = git("rev-parse", "HEAD")
    source = tmp_path / "families/example/support.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        ("" if change == "missing-header" else header) + '"""Example family support."""\n',
        encoding="utf-8",
    )
    if change in ("LICENSE", "NOTICE"):
        (tmp_path / change).write_text("Changed legal document.\n", encoding="utf-8")
    git("add", ".")
    git("commit", "--quiet", "-m", "Contribution fixture")

    # Keep the real public entrypoint, header audit, and Git comparison; unrelated
    # architecture and formatter checks need the full project and toolchain.
    for name in ("family_coverage", "complexity", "lint_changed_files", "architecture_contracts"):
        monkeypatch.setattr(community_ci.SourceQualityChecks, name, lambda _self: None)
    runner = community_ci.CommunityCI(tmp_path, dict(os.environ))
    if change == "valid":
        runner.source_quality(base)
        assert "findings=0" in capfd.readouterr().out
    else:
        with pytest.raises(community_ci.CiError):
            runner.source_quality(base)
        captured = capfd.readouterr()
        output = captured.out + captured.err
        if change == "missing-header":
            assert (
                "[missing] families/example/support.py: missing required hash SPDX header" in output
            )
        else:
            assert change in output


def test_public_workflow_is_one_exact_merge_cpu_then_gpu_authorization() -> None:
    path = REPO_ROOT / ".github" / "workflows" / "community-ci.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    source = path.read_text(encoding="utf-8")

    assert workflow["name"] == "Community CI"
    assert "Manual PR #{0} · community CI" in workflow["run-name"]
    assert "PR #{0} · community CI · head {1} · base {2}" in workflow["run-name"]
    assert "github.sha" not in workflow["run-name"]
    assert workflow["permissions"] == {}
    assert workflow["env"] == {"COMMUNITY_GPU_EXECUTION_ENABLED": "false"}
    assert "pull_request_target:" in source
    assert "branches: [main]" in source
    assert "types: [opened, synchronize, reopened, ready_for_review]" in source
    assert "workflow_dispatch:" in source
    assert "\n  pull_request:\n" not in source
    assert "workflow_run:" not in source
    assert "issue_comment:" not in source
    assert "/run-ci" not in source
    assert "checks: write" not in source
    assert "pull-requests: write" not in source
    assert "self-hosted" not in source
    assert source.count("secrets.BREV_API_KEY") == 2
    assert source.count("CI_BASE_REF: ${{ needs.authorize.outputs.merge_sha }}^1") == 2
    assert "persist-credentials: false" in source
    assert "cancel-in-progress: true" in source
    assert "check-runs" not in source
    assert "issues/comments" not in source
    workflows = REPO_ROOT / ".github" / "workflows"
    assert not (workflows / "community-cpu.yml").exists()
    assert not (workflows / "community-gpu-ci.yml").exists()

    jobs = workflow["jobs"]
    assert all(job["runs-on"] == "ubuntu-24.04" for job in jobs.values())
    assert jobs["authorize"]["permissions"] == {
        "contents": "read",
        "pull-requests": "read",
    }
    assert set(jobs["authorize"]["outputs"]) == {
        "pr_number",
        "head_sha",
        "base_sha",
        "merge_sha",
    }
    for job_name in ("source-quality", "docs", "ownership-impact", "unit"):
        assert jobs[job_name]["permissions"] == {"contents": "read"}
        assert jobs[job_name]["needs"] == "authorize"
        checkout = jobs[job_name]["steps"][0]
        assert checkout["name"] == "Check out the exact PR merge"
        assert checkout["with"] == {
            "ref": "${{ needs.authorize.outputs.merge_sha }}",
            "fetch-depth": 0,
            "persist-credentials": False,
        }
    assert "if" not in jobs["unit"]
    unit_steps = {step["name"]: step for step in jobs["unit"]["steps"]}
    assert unit_steps["Run hardened source-only units"]["run"] == (
        "python3 -m tools.community_ci unit"
    )
    assert jobs["required"]["needs"] == [
        "source-quality",
        "docs",
        "ownership-impact",
        "unit",
    ]
    assert jobs["required"]["permissions"] == {}
    assert jobs["required"]["if"] == "${{ !cancelled() }}"

    gpu_authorize = jobs["gpu-authorize"]
    assert gpu_authorize["needs"] == ["authorize", "required"]
    assert gpu_authorize["if"] == (
        "${{ always() && needs.authorize.result == 'success' && "
        "needs.required.result == 'success' }}"
    )
    assert gpu_authorize["permissions"] == {"contents": "read"}
    assert gpu_authorize["outputs"]["merge_sha"] == ("${{ needs.authorize.outputs.merge_sha }}")
    assert gpu_authorize["outputs"]["added_families"] == (
        "${{ steps.impact.outputs.added_families }}"
    )
    assert gpu_authorize["outputs"]["gpu_enabled"] == ("${{ steps.impact.outputs.gpu_enabled }}")
    assert gpu_authorize["outputs"]["run_gpu"] == "${{ steps.impact.outputs.run_gpu }}"
    gpu_authorize_steps = {step["name"]: step for step in gpu_authorize["steps"]}
    impact_step = gpu_authorize_steps["Resolve the changed model families"]
    assert impact_step["env"]["GPU_EXECUTION_ENABLED"] == (
        "${{ env.COMMUNITY_GPU_EXECUTION_ENABLED }}"
    )
    policy_step = gpu_authorize_steps["Report the GPU execution policy"]
    assert "temporarily disabled and is not a merge gate" in policy_step["run"]
    assert jobs["announce"]["needs"] == "gpu-authorize"
    assert jobs["announce"]["if"] == "${{ needs.gpu-authorize.outputs.run_gpu == 'true' }}"
    assert jobs["provision-and-test"]["needs"] == ["gpu-authorize", "announce"]
    assert jobs["provision-and-test"]["environment"] == {
        "name": "gpu-ci-dispatch",
        "deployment": False,
    }
    assert jobs["provision-and-test"]["concurrency"] == {
        "group": "trtmc-community-gpu",
        "cancel-in-progress": False,
    }
    assert jobs["publish"]["needs"] == [
        "authorize",
        "gpu-authorize",
        "announce",
        "provision-and-test",
        "cleanup",
    ]
    assert jobs["publish"]["name"] == "Community GPU / Result"
    assert "needs.gpu-authorize.outputs.gpu_enabled == 'true'" in jobs["publish"]["if"]
    assert jobs["cleanup"]["needs"] == ["gpu-authorize", "provision-and-test"]
    assert jobs["cleanup"]["environment"] == {
        "name": "gpu-ci-dispatch",
        "deployment": False,
    }
    gpu_test = {step["name"]: step for step in jobs["provision-and-test"]["steps"]}[
        "Build the GPU image, check out the exact PR merge, and run the smoke test"
    ]
    assert gpu_test["env"]["MERGE_SHA"] == "${{ needs.gpu-authorize.outputs.merge_sha }}"
    assert "refs/pull/$PR_NUMBER/merge" in gpu_test["run"]
    assert r"\$(git rev-parse FETCH_HEAD)" in gpu_test["run"]
    assert '= $MERGE_SHA && git checkout --detach $MERGE_SHA"' in gpu_test["run"]
    assert "python3.12 -m tools.community_gpu_ci" in gpu_test["run"]
    assert "tests/e2e/models" not in gpu_test["run"]
    assert "py-only" not in gpu_test["run"]
    assert "python3.12 -m pytest" not in gpu_test["run"]

    internal_bridge = (REPO_ROOT / ".github" / "workflows" / "internal-ci-bridge.yml").read_text(
        encoding="utf-8"
    )
    assert "community-ci.yml/runs?event=pull_request_target&head_sha=$head_sha" in (internal_bridge)
    assert "community-cpu.yml" not in internal_bridge
    assert "/actions/runs/$community_ci_run/jobs?filter=latest&per_page=100" in internal_bridge
    assert 'name == "Community CPU / Required" and .conclusion == "success"' in internal_bridge
    assert ".display_title == $title" in internal_bridge
    assert "· base $base_sha" in internal_bridge

    docs = jobs["docs"]
    assert "if" not in docs
    docs_steps = {step["name"]: step for step in docs["steps"]}
    assert list(docs_steps) == [
        "Check out the exact PR merge",
        "Set up Node",
        "Install website dependencies",
        "Test generated model support inventory",
        "Build production documentation",
    ]
    assert all("if" not in step for step in docs_steps.values())
    assert docs_steps["Set up Node"] == {
        "name": "Set up Node",
        "uses": "actions/setup-node@820762786026740c76f36085b0efc47a31fe5020",
        "with": {"node-version": "20"},
    }
    assert docs_steps["Install website dependencies"] == {
        "name": "Install website dependencies",
        "working-directory": "website",
        "run": "npm ci",
    }
    assert docs_steps["Test generated model support inventory"] == {
        "name": "Test generated model support inventory",
        "working-directory": "website",
        "run": "npm run test:model-support",
    }
    assert docs_steps["Build production documentation"] == {
        "name": "Build production documentation",
        "working-directory": "website",
        "env": {
            "SITE_URL": "https://nvidia.github.io",
            "BASE_URL": "/TensorRT-Model-Connect/",
        },
        "run": "npm run build",
    }


@pytest.mark.parametrize(
    ("source_quality", "docs", "ownership_impact", "unit", "expected_returncode"),
    [
        ("success", "success", "success", "success", 0),
        ("failure", "success", "success", "success", 1),
        ("success", "failure", "success", "success", 1),
        ("success", "skipped", "success", "success", 1),
        ("success", "success", "failure", "failure", 1),
    ],
)
def test_public_required_job_fails_closed(
    source_quality: str,
    docs: str,
    ownership_impact: str,
    unit: str,
    expected_returncode: int,
) -> None:
    environment = {
        **os.environ,
        "SOURCE_QUALITY_RESULT": source_quality,
        "DOCS_RESULT": docs,
        "OWNERSHIP_IMPACT_RESULT": ownership_impact,
        "UNIT_RESULT": unit,
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-ci.yml",
                "required",
                "Require every public CPU stage",
            ),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == expected_returncode, result.stdout + result.stderr
    assert f"Source quality: {source_quality}" in result.stdout
    assert f"Docs: {docs}" in result.stdout
    assert f"Ownership and impact: {ownership_impact}" in result.stdout
    assert f"Unit / C++ and Python: {unit}" in result.stdout


@pytest.mark.parametrize(
    ("combined_cpu_job_id", "combined_title_matches", "expected_returncode"),
    [
        ("33", True, 0),
        ("", True, 1),
        ("33", False, 1),
    ],
)
def test_internal_label_bridge_accepts_only_the_exact_combined_cpu_gate(
    tmp_path: Path,
    combined_cpu_job_id: str,
    combined_title_matches: bool,
    expected_returncode: int,
) -> None:
    head_sha = "a" * 40
    base_sha = "b" * 40
    expected_title = f"PR #17 · community CI · head {head_sha} · base {base_sha}"
    run_title = expected_title if combined_title_matches else "PR #17 · stale Community CI"
    github_output = tmp_path / "github-output"
    gh = tmp_path / "gh"
    gh.write_text(
        """#!/bin/bash
set -euo pipefail
arguments="$*"
case "$arguments" in
  *collaborators/tester/permission*) printf '%s\n' maintain ;;
  *pulls/17*)
    printf '{"state":"open","base":{"repo":{"full_name":"example/repo"},"ref":"main","sha":"%s"},"head":{"sha":"%s"}}\n' "$BASE_SHA" "$HEAD_SHA"
    ;;
  *community-ci.yml*)
    printf '{"workflow_runs":[{"id":22,"display_title":"%s","updated_at":"2026-01-01T00:00:00Z"}]}\n' "$RUN_TITLE"
    ;;
  *actions/runs/22/jobs*) printf '%s\n' "$COMBINED_CPU_JOB_ID" ;;
  *) printf 'unexpected gh call: %s\n' "$arguments" >&2; exit 99 ;;
esac
""",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "internal-ci-bridge.yml",
                "authorize",
                "Capture the exact pull-request snapshot",
            ),
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "ACTOR": "tester",
            "PR_NUMBER": "17",
            "EVENT_NAME": "pull_request_target",
            "EVENT_HEAD_SHA": head_sha,
            "GITHUB_REPOSITORY": "example/repo",
            "GITHUB_OUTPUT": str(github_output),
            "HEAD_SHA": head_sha,
            "BASE_SHA": base_sha,
            "COMBINED_CPU_JOB_ID": combined_cpu_job_id,
            "RUN_TITLE": run_title,
        },
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == expected_returncode, result.stdout + result.stderr
    if expected_returncode == 0:
        assert github_output.read_text(encoding="utf-8") == (
            f"pr_number=17\nhead_sha={head_sha}\nbase_sha={base_sha}\n"
        )
    else:
        assert "Community CPU / Required must pass" in result.stdout + result.stderr


def test_cpu_image_installs_the_same_pinned_community_requirements() -> None:
    dockerfile = (REPO_ROOT / "Dockerfile.community-cpu").read_text(encoding="utf-8")
    dockerignore = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert "-base-ubuntu24.04@sha256:" in dockerfile
    assert "COPY community-ci.txt" in dockerfile
    assert "pip install --requirement /tmp/trtmc-community-ci.txt" in dockerfile
    assert '"libnvinfer11=${TENSORRT_APT_VERSION}"' in dockerfile
    assert '"libnvinfer-safe-headers-dev=${TENSORRT_APT_VERSION}"' in dockerfile
    assert "libcurand-dev-13-3" in dockerfile
    assert "cuda-nvrtc-dev-13-3" in dockerfile
    assert "      jq \\\n" in dockerfile
    assert "2.12.0+cu130" in dockerfile
    assert "torch.version.cuda == '13.0'" in dockerfile
    assert "ENV TORCH_CUDA_ARCH_LIST=10.0" in dockerfile
    assert "pip install --no-deps" in dockerfile
    assert '"tensorrt_cu13_bindings==${TENSORRT_VERSION}"' in dockerfile
    assert '"tensorrt==${TENSORRT_VERSION}"' not in dockerfile
    assert 'multiarch="$(gcc -dumpmachine)"' in dockerfile
    assert "ENV TRT_LIB_DIR=/opt/trtmc-tensorrt-lib" in dockerfile
    assert "ENV TRT_INC_DIR=/opt/trtmc-tensorrt-include" in dockerfile
    assert "/usr/lib/x86_64-linux-gnu" not in dockerfile
    assert "/usr/include/x86_64-linux-gnu" not in dockerfile
    assert "NVIDIA_VISIBLE_DEVICES" not in dockerfile
    assert "!requirements/" in dockerignore
    assert "requirements/*" in dockerignore
    assert "!requirements/base.txt" in dockerignore
    assert "!requirements/community-ci.txt" not in dockerignore


def test_gpu_image_verifies_the_native_byok_dependency() -> None:
    """The GPU image fails during construction if the native TVM-FFI input is absent."""
    dockerfile = (REPO_ROOT / "Dockerfile.dev.x86-gpu").read_text(encoding="utf-8")

    assert "import onnx, tensorrt, torch, tvm_ffi" in dockerfile
    assert "metadata.version('apache-tvm-ffi') == '0.1.12'" in dockerfile


def test_cpu_image_builds_from_the_minimal_requirements_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = community_ci.CommunityCI(REPO_ROOT, dict(os.environ))
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(runner.commands, "run", run)

    runner._ensure_cpu_image()

    assert calls[0][:3] == ["docker", "build", "--file"]
    assert calls[0][-1] == "requirements"


@pytest.mark.parametrize(
    ("job_status", "test_outcome", "test_conclusion", "expected"),
    [
        ("success", "success", "success", "success"),
        ("failure", "skipped", "", "failure"),
        ("failure", "failure", "", "failure"),
        ("failure", "success", "success", "failure"),
        ("success", "skipped", "", "failure"),
        ("success", "success", "", "failure"),
        ("success", "failure", "success", "failure"),
        ("cancelled", "cancelled", "", "cancelled"),
        ("cancelled", "success", "success", "cancelled"),
        ("", "", "", "failure"),
    ],
)
def test_gpu_step_conclusion_requires_completed_success(
    tmp_path: Path,
    job_status: str,
    test_outcome: str,
    test_conclusion: str,
    expected: str,
) -> None:
    output = tmp_path / "output"
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-ci.yml", "provision-and-test", "Record the step conclusion"
            ),
        ],
        env={
            **os.environ,
            "JOB_STATUS": job_status,
            "TEST_OUTCOME": test_outcome,
            "TEST_CONCLUSION": test_conclusion,
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.read_text(encoding="utf-8") == f"conclusion={expected}\n"


@pytest.mark.parametrize(
    ("run_gpu", "job_result", "conclusion", "cleanup_result", "expected_state"),
    [
        ("false", "skipped", "", "skipped", "success"),
        ("false", "success", "success", "success", "failure"),
        ("true", "success", "success", "success", "success"),
        ("true", "success", "success", "failure", "failure"),
        ("true", "failure", "success", "success", "failure"),
        ("true", "cancelled", "success", "success", "failure"),
        ("true", "skipped", "", "skipped", "failure"),
        ("true", "success", "failure", "success", "failure"),
        ("true", "success", "cancelled", "success", "failure"),
        ("true", "success", "", "success", "failure"),
    ],
)
def test_gpu_published_status_requires_job_and_test_success(
    tmp_path: Path,
    run_gpu: str,
    job_result: str,
    conclusion: str,
    cleanup_result: str,
    expected_state: str,
) -> None:
    output = tmp_path / "status-args"
    gh = tmp_path / "gh"
    gh.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$STATUS_ARGS"\n', encoding="utf-8")
    gh.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script("community-ci.yml", "publish", "Publish the terminal status"),
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "STATUS_ARGS": str(output),
            "RUN_GPU": run_gpu,
            "JOB_RESULT": job_result,
            "CONCLUSION": conclusion,
            "CLEANUP_RESULT": cleanup_result,
            "GITHUB_REPOSITORY": "example/model-connect",
            "GITHUB_SERVER_URL": "https://github.com",
            "GITHUB_RUN_ID": "123",
            "HEAD_SHA": "a" * 40,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"state={expected_state}" in output.read_text(encoding="utf-8").splitlines()


def test_gpu_status_and_cleanup_fail_closed() -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/community-ci.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["provision-and-test"]
    steps = {step["name"]: step for step in job["steps"]}
    assert steps["Reserve a GPU instance"]["id"] == "reserve"
    test_step = steps["Build the GPU image, check out the exact PR merge, and run the smoke test"]
    assert "sudo docker build -f Dockerfile.dev.x86-gpu" in test_step["run"]
    assert "sudo docker run --rm --gpus all" in test_step["run"]
    result = steps["Record the step conclusion"]
    assert result["id"] == "result"
    assert result["if"] == "always()"
    assert result["env"] == {
        "JOB_STATUS": "${{ job.status }}",
        "TEST_OUTCOME": "${{ steps.test.outcome }}",
        "TEST_CONCLUSION": "${{ steps.test.outputs.conclusion }}",
    }
    assert "${{" not in result["run"]
    cleanup = steps["Always tear down the GPU instance"]
    assert cleanup["if"] == "${{ always() && steps.reserve.outputs.instance_name != '' }}"
    assert cleanup["env"] == {"INSTANCE_NAME": "${{ steps.reserve.outputs.instance_name }}"}
    assert cleanup["run"] == 'brev delete "$INSTANCE_NAME" || true'
    assert job["outputs"] == {"conclusion": "${{ steps.result.outputs.conclusion }}"}
    cleanup_job = workflow["jobs"]["cleanup"]
    assert "always()" in cleanup_job["if"]
    assert "needs.gpu-authorize.outputs.run_gpu == 'true'" in cleanup_job["if"]
    cleanup_steps = {step["name"]: step for step in cleanup_job["steps"]}
    assert cleanup_steps["Delete the deterministic GPU instance"]["run"] == (
        'brev delete "trtmc-gpu-ci-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}" || true'
    )
    publish = workflow["jobs"]["publish"]["steps"][0]
    assert publish["env"]["RUN_GPU"] == "${{ needs.gpu-authorize.outputs.run_gpu }}"
    assert publish["env"]["JOB_RESULT"] == "${{ needs.provision-and-test.result }}"
    assert publish["env"]["CLEANUP_RESULT"] == "${{ needs.cleanup.result }}"

    for install_step in (
        steps["Install the Brev CLI"],
        cleanup_steps["Install the pinned Brev CLI"],
    ):
        assert install_step["env"] == {
            "BREV_VERSION": "0.6.335",
            "BREV_ARCHIVE_SHA256": (
                "89d778e6f1e5e52495f3e0f10393f1666a1b16d180001b13faec6f906955e6f8"
            ),
        }
        assert "raw.githubusercontent.com" not in install_step["run"]
        assert "sha256sum --check --strict" in install_step["run"]


@pytest.mark.parametrize(
    ("changed_path", "expected_scope", "expected_families", "expected_added_families"),
    [
        ("families/bert/model.py", "families", ["bert"], []),
        ("families/new_family/model.py", "all", ["bert", "gpt2"], ["new_family"]),
        ("README.md", "docs", [], []),
    ],
)
def test_gpu_impact_executes_only_trusted_base_code(
    tmp_path: Path,
    changed_path: str,
    expected_scope: str,
    expected_families: list[str],
    expected_added_families: list[str],
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    for family in ("bert", "gpt2"):
        root = repository / "families" / family
        root.mkdir(parents=True)
        (root / "model.py").write_text("# trusted base\n", encoding="utf-8")
    tools = repository / "tools"
    tools.mkdir()
    (tools / "__init__.py").write_text("", encoding="utf-8")
    (tools / "test_impact.py").write_text(
        (REPO_ROOT / "tools/test_impact.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (repository / "README.md").write_text("Trusted documentation\n", encoding="utf-8")

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=repository,
            env={
                **os.environ,
                "GIT_AUTHOR_NAME": "CI Test",
                "GIT_AUTHOR_EMAIL": "test@example.invalid",
                "GIT_COMMITTER_NAME": "CI Test",
                "GIT_COMMITTER_EMAIL": "test@example.invalid",
            },
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    git("init")
    git("add", ".")
    git("-c", "core.hooksPath=/dev/null", "commit", "-m", "trusted fixture")
    base = git("rev-parse", "HEAD")
    changed = repository / changed_path
    changed.parent.mkdir(parents=True, exist_ok=True)
    changed.write_text("# pull-request content\n", encoding="utf-8")
    git("add", changed_path)
    git("-c", "core.hooksPath=/dev/null", "commit", "-m", "untrusted fixture")
    head = git("rev-parse", "HEAD")

    sentinel = tmp_path / "untrusted-code-executed"
    (tools / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\n"
        "raise RuntimeError('untrusted')\n",
        encoding="utf-8",
    )
    git("add", "tools/__init__.py")
    git("-c", "core.hooksPath=/dev/null", "commit", "-m", "poison fixture")
    poisoned_head = git("rev-parse", "HEAD")
    git("checkout", "--detach", base)

    output = tmp_path / "output"
    script = _workflow_step_script(
        "community-ci.yml", "gpu-authorize", "Resolve the changed model families"
    )
    for revision in (head, poisoned_head):
        output.write_text("", encoding="utf-8")
        result = subprocess.run(
            ["bash", "-c", script],
            cwd=repository,
            env={
                **os.environ,
                "PYTHONPATH": "",
                "BASE_SHA": base,
                "HEAD_SHA": revision,
                "GPU_EXECUTION_ENABLED": "false",
                "RUNNER_TEMP": str(tmp_path),
                "GITHUB_OUTPUT": str(output),
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert not sentinel.exists()
        assert git("rev-parse", "HEAD") == base
        summary = json.loads(result.stdout)
        values = dict(line.split("=", 1) for line in output.read_text().splitlines())
        if revision == poisoned_head:
            assert summary["scope"] == "all"
            assert summary["families"] == ["bert", "gpt2"]
        else:
            assert summary["scope"] == expected_scope
            assert summary["families"] == expected_families
        assert json.loads(values["families"]) == summary["families"]
        assert json.loads(values["added_families"]) == expected_added_families
        assert values["scope"] == summary["scope"]
        assert values["gpu_enabled"] == "false"
        assert values["run_gpu"] == "false"


@pytest.mark.parametrize("create_exitcode", [0, 1])
def test_gpu_cleanup_can_delete_instance_after_reservation_failure(
    tmp_path: Path,
    create_exitcode: int,
) -> None:
    output = tmp_path / "output"
    calls = tmp_path / "brev-calls"
    brev = tmp_path / "brev"
    brev.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$BREV_CALLS"\n'
        'if [ "$1" = "create" ]; then exit "$CREATE_EXITCODE"; fi\n',
        encoding="utf-8",
    )
    brev.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
        "BREV_CALLS": str(calls),
        "CREATE_EXITCODE": str(create_exitcode),
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "2",
        "GITHUB_OUTPUT": str(output),
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-ci.yml", "provision-and-test", "Reserve a GPU instance"
            ),
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == create_exitcode, result.stdout + result.stderr
    instance_name = "trtmc-gpu-ci-123-2"
    assert output.read_text(encoding="utf-8") == f"instance_name={instance_name}\n"
    cleanup = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-ci.yml",
                "provision-and-test",
                "Always tear down the GPU instance",
            ),
        ],
        env={**environment, "INSTANCE_NAME": instance_name},
        capture_output=True,
        text=True,
        check=False,
    )
    assert cleanup.returncode == 0, cleanup.stdout + cleanup.stderr
    assert calls.read_text(encoding="utf-8").splitlines() == [
        f"create {instance_name} -g L40 --timeout 600",
        f"delete {instance_name}",
    ]
