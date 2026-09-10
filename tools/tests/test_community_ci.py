# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contributor-visible Community CI entrypoints."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from tools import community_ci


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


def test_public_workflow_is_an_automatic_read_only_exact_merge_gate() -> None:
    path = REPO_ROOT / ".github" / "workflows" / "community-cpu.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    source = path.read_text(encoding="utf-8")

    assert workflow["run-name"] == (
        "PR #${{ github.event.pull_request.number }} · public CPU · merge ${{ github.sha }}"
    )
    assert workflow["permissions"] == {}
    assert "pull_request:" in source
    assert "branches: [main]" in source
    assert "types: [opened, synchronize, reopened, ready_for_review]" in source
    assert "issue_comment:" not in source
    assert "pull_request_target" not in source
    assert "workflow_dispatch:" not in source
    assert "/run-ci" not in source
    assert "checks: write" not in source
    assert "pull-requests: write" not in source
    assert "secrets." not in source
    assert "self-hosted" not in source
    assert "github.event.pull_request.base.sha" not in source
    assert source.count("CI_BASE_REF: ${{ github.sha }}^1") == 2
    assert "ref: ${{ github.sha }}" in source
    assert "persist-credentials: false" in source
    assert "cancel-in-progress: true" in source
    assert "--gpus" not in source
    assert "check-runs" not in source
    assert "issues/comments" not in source

    jobs = workflow["jobs"]
    assert [job["name"] for job in jobs.values()] == [
        "Community CPU / Source quality",
        "Community CPU / Docs",
        "Community CPU / Ownership and impact",
        "Community CPU / Unit / C++ and Python",
        "Community CPU / Required",
    ]
    assert all(job["runs-on"] == "ubuntu-24.04" for job in jobs.values())
    for job_name in ("source-quality", "docs", "ownership-impact", "unit"):
        assert jobs[job_name]["permissions"] == {"contents": "read"}
    assert "if" not in jobs["unit"]
    assert "needs" not in jobs["unit"]
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

    docs = jobs["docs"]
    assert "if" not in docs
    assert "needs" not in docs
    docs_steps = {step["name"]: step for step in docs["steps"]}
    assert list(docs_steps) == [
        "Check out the exact PR merge",
        "Set up Node",
        "Install website dependencies",
        "Test generated model support inventory",
        "Build production documentation",
    ]
    assert all("if" not in step for step in docs_steps.values())
    assert docs_steps["Check out the exact PR merge"]["with"] == {
        "ref": "${{ github.sha }}",
        "fetch-depth": 0,
        "persist-credentials": False,
    }
    assert docs_steps["Set up Node"] == {
        "name": "Set up Node",
        "uses": "actions/setup-node@49933ea5288caeca8642d1e84afbd3f7d6820020",
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
                "community-cpu.yml",
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
                "community-gpu-ci.yml", "provision-and-test", "Record the step conclusion"
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
    ("job_result", "conclusion", "expected_state"),
    [
        ("success", "success", "success"),
        ("failure", "success", "failure"),
        ("cancelled", "success", "failure"),
        ("skipped", "", "failure"),
        ("success", "failure", "failure"),
        ("success", "cancelled", "failure"),
        ("success", "", "failure"),
    ],
)
def test_gpu_published_status_requires_job_and_test_success(
    tmp_path: Path, job_result: str, conclusion: str, expected_state: str
) -> None:
    output = tmp_path / "status-args"
    gh = tmp_path / "gh"
    gh.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$STATUS_ARGS"\n', encoding="utf-8")
    gh.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script("community-gpu-ci.yml", "publish", "Publish the terminal status"),
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "STATUS_ARGS": str(output),
            "JOB_RESULT": job_result,
            "CONCLUSION": conclusion,
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


def test_gpu_status_contexts_and_cleanup_remain_unconditional() -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/community-gpu-ci.yml").read_text(encoding="utf-8")
    )
    job = workflow["jobs"]["provision-and-test"]
    steps = {step["name"]: step for step in job["steps"]}
    result = steps["Record the step conclusion"]
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
    publish = workflow["jobs"]["publish"]["steps"][0]
    assert publish["env"]["JOB_RESULT"] == "${{ needs.provision-and-test.result }}"


@pytest.mark.parametrize(
    ("changed_path", "expected_scope", "expected_families"),
    [
        ("families/bert/model.py", "families", ["bert"]),
        ("families/new_family/model.py", "all", ["bert", "gpt2"]),
        ("README.md", "docs", []),
    ],
)
def test_gpu_impact_executes_only_trusted_base_code(
    tmp_path: Path,
    changed_path: str,
    expected_scope: str,
    expected_families: list[str],
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
        (REPO_ROOT / "tools/test_impact.py").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (repository / "README.md").write_text("Trusted documentation\n", encoding="utf-8")

    def git(*arguments: str) -> str:
        """Run fixture Git commands, returning stdout and raising on failure."""
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
    # A poison import on the PR branch must never execute on the trusted runner.
    sentinel = tmp_path / "untrusted-code-executed"
    (tools / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\nraise RuntimeError('untrusted')\n",
        encoding="utf-8",
    )
    git("add", "tools/__init__.py")
    git("-c", "core.hooksPath=/dev/null", "commit", "-m", "poison fixture")
    poisoned_head = git("rev-parse", "HEAD")
    git("checkout", "--detach", base)
    output = tmp_path / "output"
    script = _workflow_step_script(
        "community-gpu-ci.yml", "authorize", "Resolve the changed model families"
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
        assert values["scope"] == summary["scope"]


def test_gpu_authorization_uses_exact_trusted_base_and_cpu_prerequisite() -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/community-gpu-ci.yml").read_text(encoding="utf-8")
    )
    authorize = workflow["jobs"]["authorize"]
    steps = {step["name"]: step for step in authorize["steps"]}
    checkout = steps["Check out the base branch"]
    assert checkout["with"] == {
        "ref": "${{ steps.snapshot.outputs.base_sha }}",
        "fetch-depth": 0,
        "persist-credentials": False,
    }
    snapshot = steps["Capture the exact pull-request snapshot"]["run"]
    assert 'test "$base_repo" = "$GITHUB_REPOSITORY"' in snapshot
    assert 'test "$base_ref" = "main"' in snapshot
    assert "community-cpu.yml/runs?event=pull_request&head_sha=$head_sha" in snapshot
    assert 'select(.conclusion == "success")' in snapshot
    assert "Validate the changed family names" in steps


@pytest.mark.parametrize("create_exitcode", [0, 1])
def test_gpu_cleanup_can_delete_instance_after_reservation_failure(
    tmp_path: Path, create_exitcode: int
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
                "community-gpu-ci.yml", "provision-and-test", "Reserve a GPU instance"
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
                "community-gpu-ci.yml", "provision-and-test", "Always tear down the GPU instance"
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
