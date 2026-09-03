# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared capability errors and observed toolchain facts."""

from __future__ import annotations

from dataclasses import dataclass


class DevToolkitError(RuntimeError):
    """A user-facing environment preparation error."""


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
