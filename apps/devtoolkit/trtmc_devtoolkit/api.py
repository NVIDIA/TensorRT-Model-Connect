# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare only the checkout that owns this toolkit."""

from __future__ import annotations

import platform
import re
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

from .docker_target import (
    DockerLifecycle,
    DockerMount,
    DockerTargetPolicy,
    DockerTargetRequest,
)
from .models import DevToolkitError, PreparedEnvironment
from .runner import CommandRunner


_FAMILY = re.compile(r"[a-z][a-z0-9_]*\Z")


class DevToolkit:
    def __init__(self, repository: Path, runner: CommandRunner | None = None):
        self.repository = repository.resolve()
        self.runner = runner or CommandRunner()
        if not (self.repository / "pyproject.toml").is_file():
            raise DevToolkitError(f"not a TensorRT-Model-Connect checkout: {self.repository}")
        if not (self.repository / "families").is_dir():
            raise DevToolkitError(f"checkout has no families directory: {self.repository}")

    @classmethod
    def from_checkout(
        cls,
        repository: Path | None = None,
        *,
        runner: CommandRunner | None = None,
    ) -> "DevToolkit":
        return cls(repository or Path.cwd(), runner=runner)

    def prepare_docker(
        self,
        *,
        family: str | None = None,
        gpu: str = "all",
        image: str = "trtmc-dev:current",
        container: str = "trtmc-dev",
        policy: DockerTargetPolicy = DockerTargetPolicy.ENSURE,
        environment: Mapping[str, str] | None = None,
        mounts: Sequence[DockerMount] = (),
        command: Sequence[str] = ("sleep", "infinity"),
        ipc: str | None = None,
    ) -> PreparedEnvironment:
        """Prepare one exact checkout container without replacing collisions."""
        machine = platform.machine()
        dockerfile = {
            "x86_64": "Dockerfile.dev.x86",
            "aarch64": "Dockerfile.dev.aarch64",
        }.get(machine)
        if dockerfile is None:
            raise DevToolkitError(f"unsupported Docker development host architecture: {machine}")
        if not (self.repository / dockerfile).is_file():
            raise DevToolkitError(f"checkout does not provide {dockerfile}")
        requirements = self._family_requirements(family)
        target = DockerTargetRequest(
            repository=self.repository,
            name=container,
            image=image,
            gpu=gpu,
            environment=dict(environment or {}),
            mounts=tuple(mounts),
            command=tuple(command),
            ipc=ipc,
        )
        state = DockerLifecycle(self.repository, self.runner).prepare(
            target,
            policy=policy,
            build_image=lambda: self._build_docker_image(image, dockerfile),
        )
        if requirements is not None and policy is not DockerTargetPolicy.ADOPT:
            self.runner.run(
                [
                    "docker",
                    "exec",
                    "-w",
                    str(self.repository),
                    state.container_id,
                    "python3",
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "-r",
                    str(requirements.relative_to(self.repository)),
                ],
                cwd=self.repository,
            )
        return PreparedEnvironment(
            kind="docker",
            repository=self.repository,
            python="python3",
            family=family,
            container=container,
            container_id=state.container_id,
            image_id=state.image_id,
        )

    def _build_docker_image(self, image: str, dockerfile: str) -> None:
        self.runner.run(
            ["docker", "build", "--file", dockerfile, "--tag", image, "requirements"],
            cwd=self.repository,
        )

    def prepare_local(
        self,
        *,
        python: str,
        family: str | None = None,
    ) -> PreparedEnvironment:
        """Use one explicit existing system or virtual-environment interpreter."""
        requirements = self._family_requirements(family)
        executable = shutil.which(python)
        if executable is None:
            raise DevToolkitError(f"Python executable was not found: {python}")
        if requirements is not None:
            self.runner.run(
                [
                    executable,
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "-r",
                    str(requirements),
                ],
                cwd=self.repository,
            )
        return PreparedEnvironment(
            kind="local",
            repository=self.repository,
            python=executable,
            family=family,
        )

    def _family_requirements(self, family: str | None) -> Path | None:
        if family is None:
            return None
        if _FAMILY.fullmatch(family) is None:
            raise DevToolkitError(f"invalid family name: {family!r}")
        family_root = self.repository / "families" / family
        if not (family_root / "model.py").is_file():
            raise DevToolkitError(f"unknown family: {family}")
        requirements = family_root / "requirements.txt"
        return requirements if requirements.is_file() else None
