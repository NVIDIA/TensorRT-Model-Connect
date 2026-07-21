# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MANAGER = REPO_ROOT / "scripts" / "manage_gpu_workspace.sh"
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap_workspace.sh"


def test_workspace_shell_scripts_parse() -> None:
    subprocess.run(
        ["bash", "-n", str(MANAGER), str(BOOTSTRAP)],
        check=True,
        cwd=REPO_ROOT,
    )


def test_bootstrap_help_does_not_require_host_configuration() -> None:
    env = {**os.environ, "TRTMC_HOST_CONFIG": "/dev/null"}
    env.pop("TRTMC_HOST_ROOT", None)
    result = subprocess.run(
        [str(BOOTSTRAP), "--help"],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert "--id ID" in result.stdout


def test_manager_starts_with_canonical_mounts_and_state(tmp_path: Path) -> None:
    host_root = tmp_path / "host"
    repo = host_root / "workspaces" / "fix-runtime" / "repo"
    repo.mkdir(parents=True)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [ \"$1\" = container ] && [ \"$2\" = inspect ]; then
    exit 1
fi
if [ \"$1\" = run ]; then
    printf '%s\\n' \"$@\" > \"$FAKE_DOCKER_LOG\"
    exit 0
fi
exit 1
""",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_DOCKER_LOG": str(docker_log),
        "TRTMC_HOST_CONFIG": "/dev/null",
        "TRTMC_HOST_ROOT": str(host_root),
        "TRTMC_DOCKER_IMAGE": "test-image:latest",
    }
    result = subprocess.run(
        [str(MANAGER), "start", "fix-runtime"],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert "Started trtmc-dev-gb300-fix-runtime" in result.stdout
    for path in ("engines", "results", "logs", "tmp"):
        assert (host_root / "runs" / "fix-runtime" / path).is_dir()
    assert (host_root / "huggingface" / "hub").is_dir()
    assert (host_root / "data").is_dir()

    state = (host_root / "state" / "fix-runtime" / "workspace.env").read_text()
    assert "TRTMC_WORKSPACE_ID=fix-runtime" in state
    assert "TRTMC_DOCKER_IMAGE=test-image:latest" in state

    docker_args = docker_log.read_text(encoding="utf-8").splitlines()
    assert "trtmc-dev-gb300-fix-runtime" in docker_args
    assert f"{repo}:/workspace/tensorrt-model-connect" in docker_args
    assert f"{host_root}/runs/fix-runtime:/work" in docker_args
    assert f"{host_root}/huggingface:/cache/huggingface" in docker_args
    assert f"{host_root}/data:/mnt/data:ro" in docker_args
    assert "ENGINE_DIR=/work/engines" in docker_args
    assert "HF_HUB_CACHE=/cache/huggingface/hub" in docker_args


def test_manager_has_no_delete_operation() -> None:
    text = MANAGER.read_text(encoding="utf-8")
    assert "docker rm" not in text
    assert "rm -rf" not in text
    assert "system prune" not in text
    assert "image prune" not in text
    assert "builder prune" not in text
