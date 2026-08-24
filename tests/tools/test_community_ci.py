# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contributor-visible Community CPU entrypoint."""

from __future__ import annotations

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
    for hook_id in (
        "trailing-whitespace",
        "end-of-file-fixer",
        "check-yaml",
        "check-json",
        "check-toml",
        "check-merge-conflict",
        "check-case-conflict",
        "check-symlinks",
    ):
        assert hooks[hook_id]["stages"] == ["pre-commit"]
    assert hooks["ruff-check"]["stages"] == ["pre-commit"]
    assert hooks["clang-format"]["stages"] == ["pre-commit"]
    assert hooks["clang-format"]["entry"] == "clang-format --dry-run --Werror"
    assert all(hook["stages"] == ["pre-commit"] for hook in hooks.values())

    source = (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "python3 -m tools.community_ci format-" not in source
    assert "pre-push" not in source


@pytest.mark.parametrize(
    "path",
    [
        REPO_ROOT / "CONTRIBUTING.md",
        REPO_ROOT / "website" / "docs" / "extend" / "contributing.md",
    ],
)
def test_contributor_guides_match_the_live_ci_flow(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    ordered_markers = [
        "pre-commit install --install-hooks",
        "git commit --signoff",
        "git push --set-upstream origin",
        "Community CPU / Required",
        "run-internal-ci",
        "trtmc/premerge/required",
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
    monkeypatch.setattr(community_ci, "discover_catalog", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        community_ci,
        "calculate_impact",
        lambda *_args, **_kwargs: {
            "mode": "unit",
            "run_unit_tests": True,
            "unit_scope": "cli",
            "changes": [
                {
                    "classifications": [
                        {"path": "src/runtime/config/cli_support.cpp", "kind": "unit_cli"}
                    ]
                }
            ],
        },
    )

    result = runner.impact(None)

    assert result["unit_scope"] == "cli"
    assert github_output.read_text(encoding="utf-8") == ("run_unit_tests=true\nunit_scope=cli\n")
    summary = github_summary.read_text(encoding="utf-8")
    assert "Unit scope: `cli`" in summary
    assert "src/runtime/config/cli_support.cpp" in summary


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
    assert "types: [opened, synchronize, reopened]" in source
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
    assert source.count("CI_HEAD_REF: ${{ github.event.pull_request.head.sha }}") == 1
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
    assert jobs["unit"]["if"] == "${{ !cancelled() }}"
    assert jobs["unit"]["needs"] == "ownership-impact"
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
        "Check documentation file references",
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
    assert docs_steps["Check documentation file references"] == {
        "name": "Check documentation file references",
        "run": "python3 tools/check_doc_file_references.py --strict website/docs",
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
    assert "pip install --no-deps" in dockerfile
    assert '"tensorrt_cu13_bindings==${TENSORRT_VERSION}"' in dockerfile
    assert '"tensorrt==${TENSORRT_VERSION}"' not in dockerfile
    assert 'multiarch="$(gcc -dumpmachine)"' in dockerfile
    assert "ENV TRT_LIB_DIR=/opt/trtmc-tensorrt-lib" in dockerfile
    assert "ENV TRT_INC_DIR=/opt/trtmc-tensorrt-include" in dockerfile
    assert "/usr/lib/x86_64-linux-gnu" not in dockerfile
    assert "/usr/include/x86_64-linux-gnu" not in dockerfile
    assert "NVIDIA_VISIBLE_DEVICES" not in dockerfile
    assert "!requirements/" not in dockerignore
    assert "!requirements/community-ci.txt" not in dockerignore


def test_cpu_image_builds_from_the_minimal_requirements_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = community_ci.CommunityCI(REPO_ROOT, dict(os.environ))
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(
            command,
            1 if command[:3] == ["docker", "image", "inspect"] else 0,
        )

    monkeypatch.setattr(runner.commands, "run", run)

    runner._ensure_cpu_image()

    assert calls[1][:3] == ["docker", "build", "--file"]
    assert calls[1][-1] == "requirements"
