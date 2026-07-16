# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start the long-lived container used by non-model CI stages."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from .environment import (
    COMMON_ENVIRONMENT,
    OPTIONAL_HUGGING_FACE_ENVIRONMENT,
    TRUSTED_ENVIRONMENT,
)
from .process import CiError, CommandRunner, GitHubFiles


@dataclass(frozen=True)
class ContainerConfig:
    """Resolved settings for a run-owned CI container."""

    name: str
    workspace: Path
    image: str
    hardened: bool

    @classmethod
    def from_environment(cls, env: dict[str, str]) -> "ContainerConfig":
        run_id = env.get("GITHUB_RUN_ID", "local")
        attempt = env.get("GITHUB_RUN_ATTEMPT", "0")
        workspace_text = env.get("TRTMC_CI_WORKSPACE") or env.get("GITHUB_WORKSPACE", "")
        workspace = Path(workspace_text) if workspace_text else Path()
        if not workspace_text or not workspace.is_dir():
            raise CiError(f"CI workspace does not exist: {workspace_text or 'unset'}")
        return cls(
            name=env.get("TRTMC_CI_CONTAINER_NAME", f"trtmc-ci-{run_id}-{attempt}"),
            workspace=workspace.resolve(),
            image=env.get("TRTMC_CI_IMAGE", ""),
            hardened=env.get("TRTMC_CI_HARDENED", "false") == "true",
        )


class CiContainer:
    """Build the explicit Docker command for trusted or hardened CI stages."""

    def __init__(self, env: dict[str, str] | None = None):
        self.env = dict(env or os.environ)
        self.config = ContainerConfig.from_environment(self.env)
        self.commands = CommandRunner(cwd=self.config.workspace, env=self.env)
        self.github = GitHubFiles(self.env)

    def start(self) -> str:
        """Replace any stale run-owned container and start a clean one."""
        if not self.config.image:
            raise CiError("TRTMC_CI_IMAGE is not set")
        self.github.environment("TRTMC_CI_CONTAINER_NAME", self.config.name)
        if self.commands.run(
            ["docker", "image", "inspect", self.config.image], check=False, capture_output=True
        ).returncode:
            raise CiError(
                f"Docker image '{self.config.image}' is not present on the self-hosted runner. "
                "Set repository variable TRTMC_MANYLINUX_CI_IMAGE if the runner uses a "
                "different local manylinux image tag."
            )
        self.commands.run(
            ["docker", "rm", "-f", self.config.name], check=False, capture_output=True
        )

        options, mounts = self._runtime_boundary()
        self._prepare_host_paths()
        command = [
            "docker",
            "run",
            "-d",
            "--name",
            self.config.name,
            *options,
            *mounts,
            "-v",
            self._workspace_mount(),
            "-w",
            str(self.config.workspace),
            *self._environment_arguments(),
            self.config.image,
            "sleep",
            "infinity",
        ]
        self.commands.run(command)
        if self.config.hardened and self._has_nvidia_devices():
            self.commands.run(
                ["docker", "rm", "-f", self.config.name], check=False, capture_output=True
            )
            raise CiError("Hardened unit container unexpectedly exposes NVIDIA devices.")
        print(f"Started CI container: {self.config.name}")
        return self.config.name

    def _runtime_boundary(self) -> tuple[list[str], list[str]]:
        if not self.config.hardened:
            options = shlex.split(self.env.get("TRTMC_CONTAINER_OPTIONS", ""))
            mounts = []
            shared_users = Path("/workspace/users/yifeif")
            if shared_users.is_dir():
                mounts.extend(["-v", f"{shared_users}:{shared_users}"])
            return options, mounts

        scratch_parent = Path(self.env.get("RUNNER_TEMP", "/tmp")).resolve()
        default_scratch = (
            scratch_parent / f"trtmc-premerge-unit-{self.env.get('GITHUB_RUN_ID', 'local')}-"
            f"{self.env.get('GITHUB_RUN_ATTEMPT', '0')}"
        )
        scratch_input = Path(self.env.get("TRTMC_CI_SCRATCH_HOST", str(default_scratch)))
        if scratch_input.is_symlink():
            raise CiError(f"Hardened unit scratch must not be a symlink: {scratch_input}")
        scratch = scratch_input.resolve()
        if scratch_parent not in scratch.parents:
            raise CiError(f"Hardened unit scratch must be inside RUNNER_TEMP: {scratch}")
        (scratch / "tmp").mkdir(parents=True, exist_ok=True)
        options = [
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,exec,nosuid,nodev,size=16g",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--ipc",
            "private",
            "--runtime",
            "runc",
            "-e",
            "HOME=/tmp",
            "-e",
            "TMPDIR=/work/tmp",
            "-e",
            "PIP_NO_INDEX=1",
            "-e",
            "TRTMC_CI_SCRATCH_DIR=/work",
            "-e",
            "NVIDIA_VISIBLE_DEVICES=void",
            "-e",
            "CUDA_VISIBLE_DEVICES=",
        ]
        return options, ["-v", f"{scratch}:/work"]

    def _workspace_mount(self) -> str:
        mount = f"{self.config.workspace}:{self.config.workspace}"
        return f"{mount}:ro" if self.config.hardened else mount

    def _prepare_host_paths(self) -> None:
        if self.config.hardened:
            return
        for name in (
            "TRTMC_STORAGE_ROOT",
            "ENGINE_DIR",
            "HF_HOME",
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "HF_MODULES_CACHE",
        ):
            value = self.env.get(name, "")
            if not value:
                continue
            try:
                Path(value).mkdir(parents=True, exist_ok=True)
            except OSError:
                print(
                    f"::warning::Could not create '{value}' on the host; the CI container "
                    "will try through mounted storage."
                )
        try:
            self.commands.run(
                ["chmod", "-R", "a+rwX", str(self.config.workspace)], capture_output=True
            )
        except CiError:
            print(
                "::warning::Could not normalize workspace permissions before entering the CI container."
            )

    def _environment_arguments(self) -> list[str]:
        names = COMMON_ENVIRONMENT if self.config.hardened else TRUSTED_ENVIRONMENT
        arguments = [item for name in names for item in ("-e", f"{name}={self.env.get(name, '')}")]
        if not self.config.hardened:
            arguments.extend(
                item
                for name in OPTIONAL_HUGGING_FACE_ENVIRONMENT
                if self.env.get(name, "")
                for item in ("-e", f"{name}={self.env[name]}")
            )
        return arguments

    def _has_nvidia_devices(self) -> bool:
        result = self.commands.run(
            [
                "docker",
                "exec",
                self.config.name,
                "python3",
                "-c",
                "from pathlib import Path; "
                "raise SystemExit(0 if next(Path('/dev').glob('nvidia*'), None) else 1)",
            ],
            check=False,
            capture_output=True,
        )
        return result.returncode == 0
