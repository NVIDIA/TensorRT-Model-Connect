# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the trusted or hardened containers used by non-model CI stages.

Boundary: Docker mounts, devices, and forwarded environment variables are decided here.
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import stat
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Callable, Iterator

from .environment import (
    COMMON_ENVIRONMENT,
    OPTIONAL_HUGGING_FACE_ENVIRONMENT,
    TRUSTED_ENVIRONMENT,
)
from .process import CiError, CommandRunner, GitHubFiles


_SAFE_CONTAINER_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_HUGGING_FACE_TOKEN_NAMES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
_HOST_SECRET_ROOT = Path("/tmp")


class _ContainerInterrupted(CiError):
    """Report an interrupt only after the run-owned secret has been cleaned."""

    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"One-shot CI container interrupted by {signal.Signals(signum).name}")


@dataclass
class _InterruptionState:
    signum: int | None = None
    cleanup_started: bool = False
    cleanup_complete: bool = False


@contextmanager
def _cleanup_on_interrupt(cleanup: Callable[[], None]) -> Iterator[_InterruptionState]:
    """Clean synchronously on INT/TERM and defer signals during normal cleanup."""
    state = _InterruptionState()
    handled = (signal.SIGINT, signal.SIGTERM)
    previous = {signum: signal.getsignal(signum) for signum in handled}

    def handle(signum: int, _frame: FrameType | None) -> None:
        if state.signum is None:
            state.signum = signum
        if state.cleanup_started:
            return
        state.cleanup_started = True
        cleanup()
        state.cleanup_complete = True
        raise _ContainerInterrupted(signum)

    installed: list[signal.Signals] = []
    for signum in handled:
        try:
            signal.signal(signum, handle)
            installed.append(signum)
        except ValueError as error:
            for installed_signum in installed:
                signal.signal(installed_signum, previous[installed_signum])
            raise CiError("CI container cleanup must run on the main thread") from error
    try:
        yield state
    finally:
        for signum in installed:
            signal.signal(signum, previous[signum])


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
        name = env.get("TRTMC_CI_CONTAINER_NAME", f"trtmc-ci-{run_id}-{attempt}")
        if not _SAFE_CONTAINER_NAME.fullmatch(name):
            raise CiError(f"CI container name is unsafe: {name!r}")
        return cls(
            name=name,
            workspace=workspace.resolve(),
            image=env.get("TRTMC_CI_IMAGE", ""),
            hardened=env.get("TRTMC_CI_HARDENED", "false") == "true",
        )


class CiContainer:
    """Build the explicit Docker command for trusted or hardened CI stages."""

    def __init__(self, env: dict[str, str] | None = None):
        self.env = dict(env or os.environ)
        self.config = ContainerConfig.from_environment(self.env)
        command_environment = {
            name: value for name, value in self.env.items() if name not in _HUGGING_FACE_TOKEN_NAMES
        }
        self.commands = CommandRunner(cwd=self.config.workspace, env=command_environment)
        self.github = GitHubFiles(self.env)

    def start(self) -> str:
        """Replace any stale run-owned container and start a clean one."""
        if self._has_hugging_face_token():
            raise CiError(
                "Long-lived CI containers must not receive a Hugging Face token; "
                "use 'python3 -m tools.ci container run -- <command>' instead."
            )
        self.cleanup()
        self._verify_image()
        self.github.environment("TRTMC_CI_CONTAINER_NAME", self.config.name)

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
            self._remove_container()
            raise CiError("Hardened unit container unexpectedly exposes NVIDIA devices.")
        print(f"Started CI container: {self.config.name}")
        return self.config.name

    def run_once(self, command: Sequence[str]) -> None:
        """Run one bounded command and remove its container and temporary secret."""
        if not command:
            raise CiError("One-shot CI container command is empty")
        self.cleanup()
        self._verify_image()

        options, mounts = self._runtime_boundary()
        self._prepare_host_paths()
        with _cleanup_on_interrupt(self._cleanup_owned_resources) as interruption:
            try:
                secret_arguments = self._hugging_face_secret_arguments(
                    [*options, *mounts, "-v", self._workspace_mount()]
                )
                docker_command = [
                    "docker",
                    "run",
                    "--rm",
                    "--name",
                    self.config.name,
                    *options,
                    *([] if self.config.hardened else ["--user", f"{os.getuid()}:{os.getgid()}"]),
                    *mounts,
                    "-v",
                    self._workspace_mount(),
                    "-w",
                    str(self.config.workspace),
                    *self._environment_arguments(),
                    *secret_arguments,
                    self.config.image,
                    *command,
                ]
                self.commands.run(docker_command)
            finally:
                if not interruption.cleanup_complete:
                    interruption.cleanup_started = True
                    self._cleanup_owned_resources()
                    interruption.cleanup_complete = True
            if interruption.signum is not None:
                raise _ContainerInterrupted(interruption.signum)

    def cleanup(self) -> None:
        """Remove the exact run-owned container, then its deterministic secret residue."""
        with _cleanup_on_interrupt(self._cleanup_owned_resources) as interruption:
            if not interruption.cleanup_complete:
                interruption.cleanup_started = True
                self._cleanup_owned_resources()
                interruption.cleanup_complete = True
            if interruption.signum is not None:
                raise _ContainerInterrupted(interruption.signum)

    def _cleanup_owned_resources(self) -> None:
        self._remove_container()
        self._remove_secret_directory()

    def _verify_image(self) -> None:
        if not self.config.image:
            raise CiError("TRTMC_CI_IMAGE is not set")
        if self.commands.run(
            ["docker", "image", "inspect", self.config.image],
            check=False,
            capture_output=True,
        ).returncode:
            raise CiError(
                f"Docker image '{self.config.image}' is not present on the self-hosted runner. "
                "Set repository variable TRTMC_MANYLINUX_CI_IMAGE if the runner uses a "
                "different local manylinux image tag."
            )

    def _remove_container(self) -> None:
        container_ids = self._exact_container_ids()
        if container_ids:
            result = self.commands.run(
                ["docker", "rm", "-f", self.config.name],
                check=False,
                capture_output=True,
            )
            if result.returncode:
                detail = (result.stderr or result.stdout or "").strip()
                raise CiError(
                    f"Could not remove exact CI container {self.config.name!r}: {detail or 'docker rm failed'}"
                )
        remaining = self._exact_container_ids()
        if remaining:
            raise CiError(f"Exact CI container {self.config.name!r} remains after cleanup")

    def _exact_container_ids(self) -> list[str]:
        result = self.commands.run(
            [
                "docker",
                "ps",
                "--all",
                "--quiet",
                "--filter",
                f"name=^/{self.config.name}$",
            ],
            capture_output=True,
        )
        container_ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(container_ids) > 1:
            raise CiError(
                f"Exact CI container lookup returned multiple IDs for {self.config.name!r}"
            )
        return container_ids

    def _has_hugging_face_token(self) -> bool:
        return bool(self.env.get("HF_TOKEN", "") or self.env.get("HUGGING_FACE_HUB_TOKEN", ""))

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

    def _hugging_face_secret_arguments(self, bind_arguments: Sequence[str]) -> list[str]:
        if self.config.hardened:
            return []
        primary = self.env.get("HF_TOKEN", "")
        legacy = self.env.get("HUGGING_FACE_HUB_TOKEN", "")
        if primary and legacy and primary != legacy:
            raise CiError("HF token environment values disagree")
        token = primary or legacy
        if not token:
            return []
        directory = self._secret_directory()
        token_file = directory / "token"
        self._verify_secret_is_not_aliased(token_file, bind_arguments)
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError as error:
            raise CiError(f"Run-owned CI secret directory already exists: {directory}") from error
        directory.chmod(0o700)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(token_file, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(token)
        return [
            "--mount",
            f"type=bind,src={token_file},dst=/run/secrets/hf-token,readonly",
            "-e",
            "HF_TOKEN_PATH=/run/secrets/hf-token",
        ]

    def _secret_directory(self) -> Path:
        secret_root_input = _HOST_SECRET_ROOT
        try:
            secret_root_stat = secret_root_input.lstat()
        except OSError as error:
            raise CiError("Host CI secret root is unavailable") from error
        if (
            not secret_root_input.is_absolute()
            or secret_root_input.is_symlink()
            or not stat.S_ISDIR(secret_root_stat.st_mode)
        ):
            raise CiError("Host CI secret root must be an absolute, real directory")
        if secret_root_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH) and not (
            secret_root_stat.st_mode & stat.S_ISVTX
        ):
            raise CiError("Writable host CI secret root must have the sticky bit")
        secret_root = secret_root_input.resolve()
        directory = secret_root / f"trtmc-hf-secret-{self.config.name}"
        if directory.parent != secret_root:
            raise CiError("Run-owned CI secret directory escapes the host secret root")
        return directory

    def _verify_secret_is_not_aliased(
        self, token_file: Path, bind_arguments: Sequence[str]
    ) -> None:
        """Reject an ordinary bind that would expose the token by a second path."""
        try:
            token_path = token_file.resolve()
            bind_sources = self._bind_mount_sources(bind_arguments)
        except CiError:
            raise
        except (OSError, RuntimeError) as error:
            raise CiError("Could not validate CI container bind sources") from error
        for source in bind_sources:
            if token_path == source or source in token_path.parents:
                raise CiError(
                    f"Hugging Face token path would be exposed by container bind source: {source}"
                )

    @staticmethod
    def _bind_mount_sources(arguments: Sequence[str]) -> list[Path]:
        """Return canonical host paths from Docker volume and bind-mount options."""
        sources: list[Path] = []
        index = 0
        while index < len(arguments):
            argument = arguments[index]
            volume_spec: str | None = None
            mount_spec: str | None = None
            if argument == "--volumes-from" or argument.startswith("--volumes-from="):
                raise CiError(
                    "Credentialed CI containers cannot inherit unverified container volumes"
                )
            if argument in ("-v", "--volume", "--mount"):
                index += 1
                if index >= len(arguments):
                    raise CiError(f"Docker {argument} option is missing its value")
                if argument == "--mount":
                    mount_spec = arguments[index]
                else:
                    volume_spec = arguments[index]
            elif argument.startswith("--volume="):
                volume_spec = argument.removeprefix("--volume=")
            elif argument.startswith("--mount="):
                mount_spec = argument.removeprefix("--mount=")
            elif argument.startswith("-v") and argument != "-v":
                volume_spec = argument[2:]

            if volume_spec is not None:
                source_text, separator, _destination = volume_spec.partition(":")
                if separator and source_text:
                    source = Path(source_text)
                    if source.is_absolute():
                        sources.append(source.resolve())
                    elif "/" in source_text or source_text.startswith("."):
                        raise CiError(
                            f"Docker bind source must be an absolute host path: {source_text!r}"
                        )
            if mount_spec is not None:
                fields: dict[str, str] = {}
                for field in mount_spec.split(","):
                    key, separator, value = field.partition("=")
                    if separator:
                        fields[key.lower()] = value
                if fields.get("type", "").lower() == "bind":
                    source_text = fields.get("src") or fields.get("source")
                    if not source_text:
                        raise CiError("Docker bind mount is missing its host source")
                    source = Path(source_text)
                    if not source.is_absolute():
                        raise CiError(
                            f"Docker bind source must be an absolute host path: {source_text!r}"
                        )
                    sources.append(source.resolve())
            index += 1
        return sources

    def _remove_secret_directory(self) -> None:
        directory = self._secret_directory()
        try:
            directory_stat = directory.lstat()
        except FileNotFoundError:
            return
        if not stat.S_ISDIR(directory_stat.st_mode) or directory.is_symlink():
            raise CiError(f"Run-owned CI secret path is not a safe directory: {directory}")
        if directory_stat.st_uid != os.getuid() or stat.S_IMODE(directory_stat.st_mode) != 0o700:
            raise CiError(
                f"Run-owned CI secret directory has unsafe ownership or mode: {directory}"
            )

        token_file = directory / "token"
        try:
            entries = list(directory.iterdir())
            if entries and entries != [token_file]:
                raise CiError(f"Run-owned CI secret directory has unexpected contents: {directory}")
            if token_file.exists() or token_file.is_symlink():
                token_stat = token_file.lstat()
                if (
                    not stat.S_ISREG(token_stat.st_mode)
                    or token_file.is_symlink()
                    or token_stat.st_uid != os.getuid()
                    or stat.S_IMODE(token_stat.st_mode) != 0o600
                    or token_stat.st_nlink != 1
                ):
                    raise CiError(f"Run-owned Hugging Face token file is unsafe: {token_file}")
                token_file.unlink()
            directory.rmdir()
        except CiError:
            raise
        except OSError as error:
            raise CiError("Could not remove the run-owned Hugging Face token file") from error

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
