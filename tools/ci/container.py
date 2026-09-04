# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start the run-owned container used by non-model CI stages.

Boundary: Docker mounts, devices, and forwarded environment variables are decided here.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from .environment import (
    COMMON_ENVIRONMENT,
    OPTIONAL_HUGGING_FACE_ENVIRONMENT,
    TRUSTED_ENVIRONMENT,
    forwarded_environment,
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
            raise CiError(f"Docker image is missing: {self.config.image}")
        self.commands.run(
            ["docker", "rm", "-f", self.config.name], check=False, capture_output=True
        )

        self._prepare_host_paths()
        options, mounts = self._runtime_boundary()
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
            raw_read_only_mounts = [
                self.env.get(name, "")
                for name in (
                    "HF_HUB_CACHE",
                    "HUGGINGFACE_HUB_CACHE",
                )
            ]
            raw_writable_mounts = [
                self.env.get("HF_MODULES_CACHE", ""),
                self.env.get("TRTMC_RUNTIME_ROOT", ""),
                self.env.get("TRTMC_NATIVE_BUILD_DIR", ""),
            ]
            binary = self.env.get("TRTMC_BINARY", "")
            if binary:
                raw_writable_mounts.append(str(Path(binary).parent))
            raw_writable_mounts.extend(
                value
                for name, value in self.env.items()
                if name.startswith("TRTMC_") and name.endswith("_MODEL_DIR")
            )

            def host_paths(raw_paths: list[str]) -> list[Path]:
                resolved: list[Path] = []
                for raw_path in raw_paths:
                    if not raw_path:
                        continue
                    candidate = Path(raw_path).expanduser()
                    if not candidate.is_absolute():
                        continue
                    host_path = candidate.resolve()
                    if not host_path.is_dir():
                        continue
                    if (
                        host_path == Path.home().resolve()
                        or host_path in self.config.workspace.parents
                        or host_path == self.config.workspace
                        or host_path.is_relative_to(self.config.workspace)
                    ):
                        continue
                    if any(
                        host_path == parent or host_path.is_relative_to(parent)
                        for parent in resolved
                    ):
                        continue
                    resolved = [
                        child for child in resolved if not child.is_relative_to(host_path)
                    ]
                    resolved.append(host_path)
                return sorted(resolved)

            read_only = host_paths(raw_read_only_mounts)
            writable = [
                path for path in host_paths(raw_writable_mounts) if path not in read_only
            ]
            for host_path in writable:
                mounts.extend(["-v", f"{host_path}:{host_path}"])
            for host_path in read_only:
                mounts.extend(["-v", f"{host_path}:{host_path}:ro"])
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
            "USER=trtmc-ci",
            "-e",
            "LOGNAME=trtmc-ci",
            "-e",
            "TMPDIR=/work/tmp",
            "-e",
            "TEMP=/work/tmp",
            "-e",
            "TMP=/work/tmp",
            "-e",
            "XDG_CACHE_HOME=/work/cache",
            "-e",
            "TORCHINDUCTOR_CACHE_DIR=/work/torch-cache",
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

    def _environment_arguments(self) -> list[str]:
        names = forwarded_environment(
            COMMON_ENVIRONMENT if self.config.hardened else TRUSTED_ENVIRONMENT,
            self.env,
        )
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
