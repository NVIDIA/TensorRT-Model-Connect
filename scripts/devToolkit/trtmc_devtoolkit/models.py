# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared capability errors and observed toolchain facts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import re
from types import MappingProxyType


class DevToolkitError(RuntimeError):
    """A user-facing environment preparation error."""


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
