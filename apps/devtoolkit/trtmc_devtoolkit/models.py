# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small value types returned by the checkout preparation API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


class DevToolkitError(RuntimeError):
    """The requested checkout environment cannot be prepared."""


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
