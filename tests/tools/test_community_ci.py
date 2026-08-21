# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the contributor-visible Community CPU entrypoint."""

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
        (REPO_ROOT / ".github" / "workflows" / workflow_name).read_text(
            encoding="utf-8"
        )
    )
    return next(
        step["run"]
        for step in workflow["jobs"][job_name]["steps"]
        if step["name"] == step_name
    )


def _write_public_ci_fake_gh(fake_bin: Path) -> None:
    fake_jq = fake_bin / "jq"
    fake_jq.write_text(
        """#!/usr/bin/env python3
import json
import sys

arguments = sys.argv[1:]
variables = {}
expression = ""
null_input = False
raw_output = False
exit_status = False
index = 0
while index < len(arguments):
    argument = arguments[index]
    if argument == "--arg":
        variables[arguments[index + 1]] = arguments[index + 2]
        index += 3
        continue
    if argument.startswith("-"):
        null_input = null_input or "n" in argument[1:]
        raw_output = raw_output or "r" in argument[1:]
        exit_status = exit_status or "e" in argument[1:]
    else:
        expression = argument
    index += 1

if null_input:
    if 'status: "in_progress"' in expression:
        result = {
            "name": variables["name"],
            "head_sha": variables["head_sha"],
            "status": "in_progress",
            "details_url": variables["details_url"],
            "external_id": variables["external_id"],
            "output": {
                "title": "Contributor-requested public CPU validation",
                "summary": (
                    "Public CPU validation is running for this exact PR merge revision."
                ),
            },
        }
    elif 'status: "completed"' in expression:
        result = {
            "status": "completed",
            "conclusion": variables["conclusion"],
            "completed_at": variables["completed_at"],
            "details_url": variables["details_url"],
            "output": {
                "title": variables["title"],
                "summary": variables["summary"],
            },
        }
    else:
        raise SystemExit(f"unsupported null-input jq expression: {expression}")
else:
    result = json.load(sys.stdin)
    optional_empty = expression.endswith(" // empty")
    path = expression.removesuffix(" // empty").removeprefix(".").split(".")
    for part in path:
        if not isinstance(result, dict) or part not in result:
            result = None
            break
        result = result[part]
    if optional_empty and result is None:
        result = ""

if exit_status and result is None:
    raise SystemExit(1)
if raw_output:
    if isinstance(result, bool):
        print(str(result).lower())
    elif result is not None:
        print(result)
else:
    print(json.dumps(result))
""",
        encoding="utf-8",
    )
    fake_jq.chmod(0o755)

    fake_gh = fake_bin / "gh"
    fake_gh.write_text(
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

arguments = sys.argv[1:]
endpoint = next(
    (argument for argument in arguments if argument.startswith("/repos/")),
    "",
)
if "/collaborators/" in endpoint:
    print(os.environ["FAKE_ACTOR_ROLE"])
elif "/pulls/" in endpoint:
    print(os.environ["FAKE_PULL_JSON"])
elif "/commits/" in endpoint and "/check-runs" in endpoint:
    print(os.environ.get("FAKE_EXISTING_REQUIRED", ""))
elif endpoint.endswith("/check-runs") and "POST" in arguments:
    input_path = arguments[arguments.index("--input") + 1]
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    capture = Path(os.environ["FAKE_PENDING_CHECK_CAPTURE"])
    existing = (
        capture.read_text(encoding="utf-8").splitlines()
        if capture.exists()
        else []
    )
    with capture.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload) + "\\n")
    print(len(existing) + 1)
elif "/check-runs/" in endpoint and "PATCH" in arguments:
    input_path = arguments[arguments.index("--input") + 1]
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    with Path(os.environ["FAKE_CHECK_CAPTURE"]).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload) + "\\n")
else:
    print(f"unexpected gh invocation: {arguments}", file=sys.stderr)
    raise SystemExit(2)
""",
        encoding="utf-8",
    )
    fake_gh.chmod(0o755)


def _public_ci_environment(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_public_ci_fake_gh(fake_bin)
    base_sha = "a" * 40
    head_sha = "b" * 40
    merge_sha = "c" * 40
    pull = {
        "state": "open",
        "base": {
            "ref": "main",
            "sha": base_sha,
            "repo": {"full_name": "NVIDIA/TensorRT-Model-Connect"},
        },
        "head": {"sha": head_sha},
        "user": {"login": "pr-author"},
        "merge_commit_sha": merge_sha,
    }
    environment = os.environ.copy()
    environment.update(
        {
            "ACTOR": "pr-author",
            "BASE_SHA": base_sha,
            "FAKE_ACTOR_ROLE": "maintain",
            "FAKE_CHECK_CAPTURE": str(tmp_path / "checks.jsonl"),
            "FAKE_PENDING_CHECK_CAPTURE": str(tmp_path / "pending-checks.jsonl"),
            "FAKE_PULL_JSON": json.dumps(pull),
            "GH_TOKEN": "test-token",
            "GITHUB_OUTPUT": str(tmp_path / "github-output"),
            "GITHUB_REPOSITORY": "NVIDIA/TensorRT-Model-Connect",
            "GITHUB_RUN_ATTEMPT": "1",
            "GITHUB_RUN_ID": "12345",
            "GITHUB_SERVER_URL": "https://github.com",
            "HEAD_SHA": head_sha,
            "MERGE_SHA": merge_sha,
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "PR_NUMBER": "980",
            "RUNNER_TEMP": str(tmp_path),
        }
    )
    return environment


def test_pre_commit_config_installs_only_lightweight_commit_hooks() -> None:
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    assert "default_install_hook_types" not in config

    repositories = {repository["repo"]: repository for repository in config["repos"]}
    assert repositories["https://github.com/astral-sh/ruff-pre-commit"]["rev"] == "v0.16.4"
    assert (
        repositories["https://github.com/pre-commit/mirrors-clang-format"]["rev"]
        == "v22.1.8"
    )

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
        "```text\n/run-ci\n```",
        "Community CPU / Required",
        "run-internal-ci",
        "trtmc/premerge/required",
    ]

    positions = [source.index(marker) for marker in ordered_markers]
    assert positions == sorted(positions)
    for marker in (
            "GitHub-hosted",
            "ubuntu-24.04",
            "read-only repository permission",
            "no access to private",
            "runners, secrets, or",
            "GPUs",
            "py -3 -m pip",
    ):
        assert marker in source
    assert "trusted request workflow" not in source
    assert "Community CPU Request" not in source


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
    assert github_output.read_text(encoding="utf-8") == (
        "run_unit_tests=true\nunit_scope=cli\n"
    )
    summary = github_summary.read_text(encoding="utf-8")
    assert "Unit scope: `cli`" in summary
    assert "src/runtime/config/cli_support.cpp" in summary


def test_public_cpu_workflow_authorizes_pr_author_comments() -> None:
    path = REPO_ROOT / ".github" / "workflows" / "community-cpu.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    source = path.read_text(encoding="utf-8")
    authorize_source = source.split("\n  authorize:", maxsplit=1)[1].split(
        "\n  initialize:", maxsplit=1
    )[0]

    assert workflow["permissions"] == {}
    assert "issue_comment:" in source
    assert "types: [created]" in source
    assert "github.event.issue.pull_request != null" in source
    assert "github.event.comment.body == '/run-ci'" in source
    assert "github.ref == 'refs/heads/main'" in source
    assert 'if [ "$ACTOR" != "$pr_author" ]; then' in source
    assert "maintain|admin)" in source
    assert "Only the PR author or a maintainer/admin" in source
    assert "actions/workflows/community-cpu.yml/dispatches" not in source
    assert "/labels/run-ci" not in source
    assert "pull_request_target" not in source
    assert "actions/checkout@" not in authorize_source
    assert "secrets." not in source
    assert "self-hosted" not in source

    authorize = workflow["jobs"]["authorize"]
    assert authorize["runs-on"] == "ubuntu-24.04"
    assert authorize["permissions"] == {
        "checks": "read",
        "contents": "read",
        "pull-requests": "read",
    }


def test_public_cpu_request_captures_the_exact_commented_snapshot(
    tmp_path: Path,
) -> None:
    environment = _public_ci_environment(tmp_path)
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-cpu.yml",
                "authorize",
                "Authorize the current PR merge",
            ),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "github-output").read_text(encoding="utf-8") == (
        "run_tests=true\n"
        "pr_number=980\n"
        f"base_sha={'a' * 40}\n"
        f"head_sha={'b' * 40}\n"
        f"merge_sha={'c' * 40}\n"
    )


def test_public_cpu_request_rejects_an_unrelated_commenter(
    tmp_path: Path,
) -> None:
    environment = _public_ci_environment(tmp_path)
    environment["ACTOR"] = "unrelated-user"
    environment["FAKE_ACTOR_ROLE"] = "read"
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-cpu.yml",
                "authorize",
                "Authorize the current PR merge",
            ),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Only the PR author or a maintainer/admin" in result.stdout + result.stderr
    assert not (tmp_path / "github-output").exists()


def test_public_cpu_request_allows_a_maintainer_to_help(
    tmp_path: Path,
) -> None:
    environment = _public_ci_environment(tmp_path)
    environment["ACTOR"] = "trusted-maintainer"
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-cpu.yml",
                "authorize",
                "Authorize the current PR merge",
            ),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "github-output").read_text(encoding="utf-8").startswith(
        "run_tests=true\n"
    )


@pytest.mark.parametrize("existing", ["queued", "in_progress", "success"])
def test_public_cpu_request_deduplicates_the_current_merge(
    tmp_path: Path,
    existing: str,
) -> None:
    environment = _public_ci_environment(tmp_path)
    environment["FAKE_EXISTING_REQUIRED"] = existing
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-cpu.yml",
                "authorize",
                "Authorize the current PR merge",
            ),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"already {existing}" in result.stdout
    assert (tmp_path / "github-output").read_text(encoding="utf-8") == (
        "run_tests=false\n"
    )


def test_public_cpu_initialization_creates_pending_exact_merge_checks(
    tmp_path: Path,
) -> None:
    environment = _public_ci_environment(tmp_path)
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-cpu.yml",
                "initialize",
                "Publish pending exact-merge checks",
            ),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    checks = [
        json.loads(line)
        for line in (tmp_path / "pending-checks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(checks) == 4
    assert {check["head_sha"] for check in checks} == {"c" * 40}
    assert {check["status"] for check in checks} == {"in_progress"}
    assert all(
        check["external_id"].startswith("community-cpu:12345:1:")
        for check in checks
    )
    assert (tmp_path / "github-output").read_text(encoding="utf-8") == (
        "source_quality_check_id=1\n"
        "ownership_impact_check_id=2\n"
        "unit_check_id=3\n"
        "required_check_id=4\n"
    )


def test_public_cpu_verdict_neutralizes_a_stale_snapshot(tmp_path: Path) -> None:
    environment = _public_ci_environment(tmp_path)
    pull = json.loads(environment["FAKE_PULL_JSON"])
    pull["head"]["sha"] = "d" * 40
    environment.update(
        {
            "FAKE_PULL_JSON": json.dumps(pull),
            "SOURCE_QUALITY_RESULT": "success",
            "OWNERSHIP_IMPACT_RESULT": "success",
            "UNIT_RESULT": "success",
            "SOURCE_QUALITY_CHECK_ID": "1",
            "OWNERSHIP_IMPACT_CHECK_ID": "2",
            "UNIT_CHECK_ID": "3",
            "REQUIRED_CHECK_ID": "4",
        }
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-cpu.yml",
                "required",
                "Publish exact-merge CPU checks",
            ),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    checks = [
        json.loads(line)
        for line in (tmp_path / "checks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(checks) == 4
    assert {check["conclusion"] for check in checks} == {"neutral"}
    assert all("PR changed" in check["output"]["summary"] for check in checks)


def test_public_cpu_verdict_publishes_success_for_the_exact_snapshot(
    tmp_path: Path,
) -> None:
    environment = _public_ci_environment(tmp_path)
    environment.update(
        {
            "SOURCE_QUALITY_RESULT": "success",
            "OWNERSHIP_IMPACT_RESULT": "success",
            "UNIT_RESULT": "success",
            "SOURCE_QUALITY_CHECK_ID": "1",
            "OWNERSHIP_IMPACT_CHECK_ID": "2",
            "UNIT_CHECK_ID": "3",
            "REQUIRED_CHECK_ID": "4",
        }
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-cpu.yml",
                "required",
                "Publish exact-merge CPU checks",
            ),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    checks = [
        json.loads(line)
        for line in (tmp_path / "checks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(checks) == 4
    assert {check["conclusion"] for check in checks} == {"success"}
    assert checks[-1]["output"]["title"] == "Community CPU: success"


def test_public_workflow_is_a_single_comment_driven_exact_merge_gate() -> None:
    path = REPO_ROOT / ".github" / "workflows" / "community-cpu.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    source = path.read_text(encoding="utf-8")

    assert workflow["run-name"] == (
        "PR #${{ github.event.issue.number }} · public CPU validation"
    )
    assert workflow["permissions"] == {}
    assert "issue_comment:" in source
    assert "workflow_dispatch:" not in source
    assert "\n  pull_request:" not in source
    assert not (
        REPO_ROOT / ".github" / "workflows" / "community-cpu-request.yml"
    ).exists()
    assert "pull_request_target" not in source
    assert "self-hosted" not in source
    assert "secrets." not in source
    assert "persist-credentials: false" in source
    assert "--gpus" not in source
    assert "upload-pages" not in source
    assert "ref: ${{ needs.authorize.outputs.merge_sha }}" in source
    assert "github.event.pull_request" not in source
    assert "inputs." not in source
    assert 'external_id="community-cpu:' in source
    assert '"/repos/$GITHUB_REPOSITORY/check-runs"' in source
    assert '"/repos/$GITHUB_REPOSITORY/check-runs/$check_id"' in source
    assert "The PR changed after public CPU validation was requested" in source

    jobs = workflow["jobs"]
    assert [job["name"] for job in jobs.values()] == [
        "Authorize public CPU request",
        "Initialize exact-merge public checks",
        "Community CPU / Source quality",
        "Community CPU / Ownership and impact",
        "Community CPU / Unit / C++ and Python",
        "Community CPU / Required",
    ]
    assert jobs["authorize"]["permissions"] == {
        "checks": "read",
        "contents": "read",
        "pull-requests": "read",
    }
    assert jobs["initialize"]["permissions"] == {
        "checks": "write",
        "contents": "read",
    }
    assert all(job["runs-on"] == "ubuntu-24.04" for job in jobs.values())
    for job_name in ("source-quality", "ownership-impact", "unit"):
        assert jobs[job_name]["permissions"] == {"contents": "read"}
    assert jobs["unit"]["needs"] == [
        "authorize",
        "initialize",
        "ownership-impact",
    ]
    assert jobs["required"]["needs"] == [
        "authorize",
        "initialize",
        "source-quality",
        "ownership-impact",
        "unit",
    ]
    assert jobs["required"]["permissions"] == {
        "checks": "write",
        "contents": "read",
        "pull-requests": "read",
    }


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
