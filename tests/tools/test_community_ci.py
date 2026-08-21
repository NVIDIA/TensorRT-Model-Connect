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
from tools.ci.process import CiError


REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_public_workflow_is_read_only_cpu_only_and_has_one_required_verdict() -> None:
    path = REPO_ROOT / ".github" / "workflows" / "community-cpu.yml"
    workflow = yaml.safe_load(path.read_text(encoding="utf-8"))
    source = path.read_text(encoding="utf-8")

    assert workflow["run-name"] == (
        "PR #${{ github.event.pull_request.number }} · public CPU validation"
    )
    assert workflow["permissions"] == {"contents": "read"}
    assert "pull_request_target" not in source
    assert "self-hosted" not in source
    assert "secrets." not in source
    assert "persist-credentials: false" in source
    assert "--gpus" not in source
    assert "upload-pages" not in source

    jobs = workflow["jobs"]
    assert [job["name"] for job in jobs.values()] == [
        "Community CPU / Source quality",
        "Community CPU / Ownership and impact",
        "Community CPU / Unit / C++ and Python",
        "Community CPU / Required",
    ]
    assert jobs["unit"]["needs"] == "ownership-impact"
    assert jobs["required"]["needs"] == ["source-quality", "ownership-impact", "unit"]


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
