# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Sequence

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    REPO_ROOT
    / "plugins"
    / "trtmc-agent-skills"
    / "skills"
    / "trtmc-agent-container"
    / "scripts"
    / "trtmc_agent_container.py"
)


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("trtmc_agent_container_skill", RUNNER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner_module = _load_runner()


@pytest.fixture(autouse=True)
def _clear_container_overrides(monkeypatch) -> None:
    for name in (
        "TRTMC_CONTAINER_IMAGE",
        "TRTMC_CONTAINER_NAME",
        "TRTMC_CONTAINER_WORKDIR",
        "TRTMC_HF_CACHE",
        "TRTMC_STORAGE_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)


def _repo(tmp_path: Path, agent: str = "agent-8") -> Path:
    repo = tmp_path / "workspaces" / agent / "TensorRT-Model-Connect"
    (repo / ".git").mkdir(parents=True)
    (repo / ".agents" / "plugins").mkdir(parents=True)
    (repo / ".agents" / "plugins" / "marketplace.json").write_text("{}\n")
    (repo / "python" / "tensorrt_model_connect").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "scripts" / "docker_build_gb300.sh").write_text("#!/bin/sh\n")
    return repo


def _completed(command: Sequence[str], code: int = 0, output: str = ""):
    return subprocess.CompletedProcess(tuple(command), code, stdout=output, stderr="")


def _container_payload(
    *, name: str, repo: Path, destination: str, running: bool
) -> str:
    return json.dumps(
        [
            {
                "Name": f"/{name}",
                "Config": {"Image": "trtmc-dev-gb300:latest"},
                "State": {"Running": running},
                "Mounts": [
                    {"Source": str(repo), "Destination": destination},
                ],
            }
        ]
    )


def test_expected_name_uses_agent_workspace(tmp_path: Path, monkeypatch) -> None:
    repo = _repo(tmp_path, "agent-12")
    monkeypatch.delenv("TRTMC_CONTAINER_NAME", raising=False)

    manager = runner_module.ContainerManager(
        repo, runner=lambda command: _completed(command, output="")
    )

    assert manager.workspace_id == "agent-12"
    assert manager.expected_name == "trtmc-dev-gb300-agent-12"


def test_ensure_reuses_matching_mount_without_renaming(tmp_path: Path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    monkeypatch.delenv("TRTMC_CONTAINER_NAME", raising=False)
    calls: list[tuple[str, ...]] = []

    def fake(command: Sequence[str]):
        command = tuple(command)
        calls.append(command)
        if command == ("docker", "ps", "-aq"):
            return _completed(command, output="abc\n")
        if command == ("docker", "inspect", "abc"):
            return _completed(
                command,
                output=_container_payload(
                    name="custom-model-connect-container",
                    repo=repo,
                    destination="/workspace/current-checkout",
                    running=True,
                ),
            )
        raise AssertionError(command)

    context = runner_module.ContainerManager(repo, runner=fake).ensure()

    assert context.container_name == "custom-model-connect-container"
    assert context.container_workdir == "/workspace/current-checkout"
    assert context.created is False
    assert not any(command[:2] == ("docker", "run") for command in calls)


def test_ensure_starts_stopped_matching_container(tmp_path: Path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    monkeypatch.delenv("TRTMC_CONTAINER_NAME", raising=False)
    calls: list[tuple[str, ...]] = []

    def fake(command: Sequence[str]):
        command = tuple(command)
        calls.append(command)
        if command == ("docker", "ps", "-aq"):
            return _completed(command, output="abc\n")
        if command == ("docker", "inspect", "abc"):
            return _completed(
                command,
                output=_container_payload(
                    name="custom-stopped-container",
                    repo=repo,
                    destination="/workspace/current-checkout",
                    running=False,
                ),
            )
        if command == ("docker", "start", "custom-stopped-container"):
            return _completed(command, output="custom-stopped-container\n")
        raise AssertionError(command)

    context = runner_module.ContainerManager(repo, runner=fake).ensure()

    assert context.state == "running"
    assert ("docker", "start", "custom-stopped-container") in calls


def test_ensure_rebuilds_standard_image_then_creates_container(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _repo(tmp_path)
    storage = tmp_path / "storage"
    cache = tmp_path / "hf-cache"
    monkeypatch.setenv("TRTMC_STORAGE_ROOT", str(storage))
    monkeypatch.setenv("TRTMC_HF_CACHE", str(cache))
    monkeypatch.delenv("TRTMC_CONTAINER_NAME", raising=False)
    calls: list[tuple[str, ...]] = []

    def fake(command: Sequence[str]):
        command = tuple(command)
        calls.append(command)
        if command == ("docker", "ps", "-aq"):
            return _completed(command, output="")
        if command == ("docker", "image", "inspect", "trtmc-dev-gb300:latest"):
            return _completed(command)
        if command == ("bash", str(repo / "scripts" / "docker_build_gb300.sh")):
            return _completed(command)
        if command[:3] == ("docker", "run", "-d"):
            return _completed(command, output="new-container-id\n")
        raise AssertionError(command)

    context = runner_module.ContainerManager(repo, runner=fake).ensure()

    assert context.created is True
    assert context.container_name == "trtmc-dev-gb300-agent-8"
    build_command = ("bash", str(repo / "scripts" / "docker_build_gb300.sh"))
    assert build_command in calls
    assert next(index for index, value in enumerate(calls) if value == build_command) < next(
        index for index, value in enumerate(calls) if value[:3] == ("docker", "run", "-d")
    )


def test_ensure_uses_existing_custom_image_without_rebuilding(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _repo(tmp_path)
    monkeypatch.setenv("TRTMC_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("TRTMC_HF_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("TRTMC_CONTAINER_IMAGE", "example.invalid/custom:latest")
    monkeypatch.delenv("TRTMC_CONTAINER_NAME", raising=False)
    calls: list[tuple[str, ...]] = []

    def fake(command: Sequence[str]):
        command = tuple(command)
        calls.append(command)
        if command == ("docker", "ps", "-aq"):
            return _completed(command, output="")
        if command == (
            "docker",
            "image",
            "inspect",
            "example.invalid/custom:latest",
        ):
            return _completed(command)
        if command[:3] == ("docker", "run", "-d"):
            return _completed(command)
        raise AssertionError(command)

    runner_module.ContainerManager(repo, runner=fake).ensure()

    assert not any(command[0] == "bash" for command in calls)


def test_resolve_rejects_ambiguous_mount_ownership(tmp_path: Path, monkeypatch) -> None:
    repo = _repo(tmp_path)
    monkeypatch.delenv("TRTMC_CONTAINER_NAME", raising=False)

    payload = json.dumps(
        [
            json.loads(
                _container_payload(
                    name=name,
                    repo=repo,
                    destination=f"/workspace/{name}",
                    running=True,
                )
            )[0]
            for name in ("candidate-a", "candidate-b")
        ]
    )

    def fake(command: Sequence[str]):
        command = tuple(command)
        if command == ("docker", "ps", "-aq"):
            return _completed(command, output="a\nb\n")
        if command == ("docker", "inspect", "a", "b"):
            return _completed(command, output=payload)
        raise AssertionError(command)

    with pytest.raises(RuntimeError, match="multiple containers mount"):
        runner_module.ContainerManager(repo, runner=fake).resolve()


def test_explicit_name_does_not_silently_select_another_container(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _repo(tmp_path)
    monkeypatch.setenv("TRTMC_CONTAINER_NAME", "requested-container")

    def fake(command: Sequence[str]):
        command = tuple(command)
        if command == ("docker", "ps", "-aq"):
            return _completed(command, output="abc\n")
        if command == ("docker", "inspect", "abc"):
            return _completed(
                command,
                output=_container_payload(
                    name="different-container",
                    repo=repo,
                    destination="/workspace/current-checkout",
                    running=True,
                ),
            )
        raise AssertionError(command)

    with pytest.raises(RuntimeError, match="requested container"):
        runner_module.ContainerManager(repo, runner=fake).resolve()


def test_build_docker_exec_quotes_command_and_uses_discovered_workdir() -> None:
    context = runner_module.ContainerContext(
        repo_root="/host/repo",
        workspace_id="agent-8",
        expected_container_name="trtmc-dev-gb300-agent-8",
        container_name="custom-container",
        container_workdir="/workspace/current checkout",
        container_image="trtmc-dev-gb300:latest",
        state="running",
    )

    command = runner_module.build_docker_exec(context, ["python3", "-c", "print('ok')"])

    assert command[:3] == ["docker", "exec", "custom-container"]
    assert "cd '/workspace/current checkout'" in command[-1]
    assert "exec python3 -c" in command[-1]
