# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contracts for keeping trusted-container credentials out of Docker argv."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tools.ci.container import CiContainer
from tools.ci.process import CiError


class _Commands:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def run(self, command, **_kwargs):
        rendered = [str(item) for item in command]
        self.calls.append(rendered)
        returncode = 1 if rendered[:2] == ["docker", "exec"] else 0
        return subprocess.CompletedProcess(rendered, returncode, "", "")


def _container(tmp_path: Path, **extra: str) -> CiContainer:
    workspace = tmp_path / "workspace"
    runner_temp = tmp_path / "runner-temp"
    workspace.mkdir()
    runner_temp.mkdir()
    environment = {
        "GITHUB_WORKSPACE": str(workspace),
        "RUNNER_TEMP": str(runner_temp),
        "TRTMC_CI_IMAGE": "ci-image",
        **extra,
    }
    return CiContainer(environment)


def test_hf_token_uses_an_ephemeral_read_only_file_mount(tmp_path: Path) -> None:
    token = "hf_test_secret_value"
    container = _container(tmp_path, HF_TOKEN=token, HUGGING_FACE_HUB_TOKEN=token)
    commands = _Commands()
    container.commands = commands

    container.start()

    docker_run = next(call for call in commands.calls if call[:3] == ["docker", "run", "-d"])
    rendered = " ".join(docker_run)
    assert token not in rendered
    assert "HF_TOKEN=" not in rendered
    assert "HUGGING_FACE_HUB_TOKEN=" not in rendered
    assert "HF_TOKEN_PATH=/run/secrets/hf-token" in docker_run
    mount = next(item for item in docker_run if "dst=/run/secrets/hf-token" in item)
    source = Path(mount.split("src=", 1)[1].split(",dst=", 1)[0])
    assert not source.exists()
    assert not source.parent.exists()


def test_conflicting_hf_token_values_fail_before_docker_run(tmp_path: Path) -> None:
    container = _container(
        tmp_path,
        HF_TOKEN="first",
        HUGGING_FACE_HUB_TOKEN="second",
    )
    commands = _Commands()
    container.commands = commands

    with pytest.raises(CiError, match="values disagree"):
        container.start()

    assert not any(call[:3] == ["docker", "run", "-d"] for call in commands.calls)


def test_hardened_container_never_receives_an_hf_secret(tmp_path: Path) -> None:
    container = _container(
        tmp_path,
        HF_TOKEN="must-not-cross-boundary",
        TRTMC_CI_HARDENED="true",
    )
    commands = _Commands()
    container.commands = commands

    container.start()

    docker_run = next(call for call in commands.calls if call[:3] == ["docker", "run", "-d"])
    rendered = " ".join(docker_run)
    assert "must-not-cross-boundary" not in rendered
    assert "HF_TOKEN" not in rendered
    assert "/run/secrets/hf-token" not in rendered
