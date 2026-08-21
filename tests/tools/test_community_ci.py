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
from tools.ci.process import CiError


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
    if 'ref: "main"' in expression:
        result = {
            "ref": "main",
            "inputs": {
                key: variables[key]
                for key in ("pr_number", "base_sha", "head_sha", "merge_sha")
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
elif "/actions/workflows/community-cpu.yml/dispatches" in endpoint:
    input_path = arguments[arguments.index("--input") + 1]
    Path(os.environ["FAKE_DISPATCH_CAPTURE"]).write_text(
        Path(input_path).read_text(encoding="utf-8"),
        encoding="utf-8",
    )
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
            "FAKE_DISPATCH_CAPTURE": str(tmp_path / "dispatch.json"),
            "FAKE_PULL_JSON": json.dumps(pull),
            "GH_TOKEN": "test-token",
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


def test_pre_commit_config_installs_fast_commit_and_complete_push_hooks() -> None:
    config = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8"))
    assert config["default_install_hook_types"] == ["pre-commit", "pre-push"]

    hooks = {hook["id"]: hook for repository in config["repos"] for hook in repository["hooks"]}
    assert hooks["trtmc-python-quality"]["stages"] == ["pre-commit"]
    assert hooks["trtmc-cpp-format"]["stages"] == ["pre-commit"]
    assert hooks["trtmc-community-pre-push"]["stages"] == ["pre-push"]
    assert hooks["trtmc-community-pre-push"]["always_run"] is True
    assert hooks["trtmc-community-pre-push"]["pass_filenames"] is False


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


def test_pre_push_collects_source_and_unit_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = community_ci.CommunityCI(REPO_ROOT, dict(os.environ))
    calls = []
    monkeypatch.setattr(runner, "resolve_base", lambda _base: "base-sha")

    def fail_source(_base: str) -> None:
        calls.append("source")
        raise CiError("format failed")

    def run_impact(_base: str) -> dict[str, object]:
        calls.append("impact")
        return {"unit_scope": "cli"}

    def fail_unit(scope: str) -> None:
        calls.append(f"unit:{scope}")
        raise CiError("test_config_cli_support failed")

    monkeypatch.setattr(runner, "source_quality", fail_source)
    monkeypatch.setattr(runner, "impact", run_impact)
    monkeypatch.setattr(runner, "unit", fail_unit)

    with pytest.raises(CiError, match="format failed") as error:
        runner.pre_push(None)

    assert calls == ["source", "impact", "unit:cli"]
    assert "test_config_cli_support failed" in str(error.value)


def test_public_cpu_request_is_a_pr_author_comment_dispatcher() -> None:
    path = REPO_ROOT / ".github" / "workflows" / "community-cpu-request.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    source = path.read_text(encoding="utf-8")

    assert workflow["permissions"] == {}
    assert "issue_comment:" in source
    assert "types: [created]" in source
    assert "github.event.issue.pull_request != null" in source
    assert "github.event.comment.body == '/run-ci'" in source
    assert "github.ref == 'refs/heads/main'" in source
    assert 'if [ "$ACTOR" != "$pr_author" ]; then' in source
    assert "maintain|admin)" in source
    assert "Only the PR author or a maintainer/admin" in source
    assert "actions/workflows/community-cpu.yml/dispatches" in source
    assert 'ref: "main"' in source
    for input_name in ("pr_number", "base_sha", "head_sha", "merge_sha"):
        assert f"{input_name}: ${input_name}" in source
    assert "/labels/run-ci" not in source
    assert "pull_request_target" not in source
    assert "actions/checkout@" not in source
    assert "secrets." not in source
    assert "self-hosted" not in source

    authorize = workflow["jobs"]["authorize"]
    assert authorize["runs-on"] == "ubuntu-24.04"
    assert authorize["permissions"] == {
        "actions": "write",
        "checks": "read",
        "contents": "read",
        "pull-requests": "read",
    }


def test_public_cpu_request_dispatches_the_exact_commented_snapshot(
    tmp_path: Path,
) -> None:
    environment = _public_ci_environment(tmp_path)
    result = subprocess.run(
        [
            "bash",
            "-c",
            _workflow_step_script(
                "community-cpu-request.yml",
                "authorize",
                "Authorize and dispatch the current PR merge",
            ),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    dispatch = json.loads((tmp_path / "dispatch.json").read_text(encoding="utf-8"))
    assert dispatch == {
        "ref": "main",
        "inputs": {
            "pr_number": "980",
            "base_sha": "a" * 40,
            "head_sha": "b" * 40,
            "merge_sha": "c" * 40,
        },
    }


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
                "community-cpu-request.yml",
                "authorize",
                "Authorize and dispatch the current PR merge",
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
    assert not (tmp_path / "dispatch.json").exists()


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
                "community-cpu-request.yml",
                "authorize",
                "Authorize and dispatch the current PR merge",
            ),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "dispatch.json").is_file()


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
                "community-cpu-request.yml",
                "authorize",
                "Authorize and dispatch the current PR merge",
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
    assert not (tmp_path / "dispatch.json").exists()


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


def test_public_workflow_supports_safe_transition_and_exact_merge_checks() -> None:
    path = REPO_ROOT / ".github" / "workflows" / "community-cpu.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    source = path.read_text(encoding="utf-8")

    assert workflow["run-name"] == (
        "PR #${{ github.event.pull_request.number || inputs.pr_number }} · public CPU validation"
    )
    assert workflow["permissions"] == {}
    assert "workflow_dispatch:" in source
    assert "\n  pull_request:" in source
    assert "Remove pull_request after the /run-ci broker is available on main" in source
    assert "pull_request_target" not in source
    assert "self-hosted" not in source
    assert "secrets." not in source
    assert "persist-credentials: false" in source
    assert "--gpus" not in source
    assert "upload-pages" not in source
    assert "ref: ${{ inputs.merge_sha || github.event.pull_request.merge_commit_sha }}" in source
    assert 'external_id="community-cpu:' in source
    assert '"/repos/$GITHUB_REPOSITORY/check-runs"' in source
    assert '"/repos/$GITHUB_REPOSITORY/check-runs/$check_id"' in source
    assert "The PR changed after public CPU validation was requested" in source

    jobs = workflow["jobs"]
    assert [job["name"] for job in jobs.values()] == [
        "Initialize exact-merge public checks",
        "Community CPU / Source quality",
        "Community CPU / Ownership and impact",
        "Community CPU / Unit / C++ and Python",
        "Community CPU / Required",
    ]
    assert jobs["initialize"]["permissions"] == {
        "checks": "write",
        "contents": "read",
        "pull-requests": "read",
    }
    assert all(job["runs-on"] == "ubuntu-24.04" for job in jobs.values())
    for job_name in ("source-quality", "ownership-impact", "unit"):
        assert jobs[job_name]["permissions"] == {"contents": "read"}
    assert jobs["unit"]["needs"] == ["initialize", "ownership-impact"]
    assert jobs["required"]["needs"] == [
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
