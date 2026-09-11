# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Value types shared by checkout preparation and environment capabilities."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .resolution import ExecutionTarget


class DevToolkitError(RuntimeError):
    """A user-facing development-environment error."""


@dataclass(frozen=True)
class ToolchainRuntime:
    """Normalized paths required to use one resolved toolchain."""

    python_executable: str
    cuda_root: str
    nvcc: str
    tensorrt_include_dir: str
    tensorrt_library: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value
            for value in (
                self.python_executable,
                self.cuda_root,
                self.nvcc,
                self.tensorrt_include_dir,
                self.tensorrt_library,
            )
        ):
            raise DevToolkitError("Toolchain runtime paths must be non-empty")


@dataclass(frozen=True)
class ToolchainObservation:
    """Facts measured from the environment that will execute TRTMC."""

    python_version: str
    cuda_version: str
    tensorrt_python_version: str
    tensorrt_native_version: str
    tensorrt_header_version: str
    tensorrt_include_dir: str
    tensorrt_library: str
    cuda_root: str | None = None
    image_id: str | None = None
    architecture: str | None = None
    evidence: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.evidence or any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for name, digest in self.evidence.items()
        ):
            raise DevToolkitError("Toolchain evidence requires named lowercase SHA-256 digests")
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))


@dataclass(frozen=True)
class PreparedEnvironment:
    """One usable local interpreter or persistent development container."""

    kind: Literal["docker", "local"]
    repository: Path
    python: str
    family: str | None = None
    container: str | None = None
    container_id: str | None = None
    image_id: str | None = None

    def execution_target(
        self,
        *,
        gpu: str = "0",
        state: str = "/tmp/trtmc-devtoolkit",
    ) -> ExecutionTarget:
        """Convert a prepared checkout into a capability execution target."""
        from .resolution import ExecutionTarget

        if self.kind == "docker":
            target = self.container_id or self.container
            if target is None:
                raise DevToolkitError("docker environment has no container")
            return ExecutionTarget.docker(
                python=self.python,
                container=target,
                workspace=str(self.repository),
                state=state,
            )
        return ExecutionTarget.local(python=self.python, gpu=gpu)

    def command(self, *arguments: str) -> tuple[str, ...]:
        if self.kind == "docker":
            target = self.container_id or self.container
            if target is None:
                raise DevToolkitError("docker environment has no container")
            return (
                "docker",
                "exec",
                "-w",
                str(self.repository),
                target,
                *arguments,
            )
        return tuple(arguments)
