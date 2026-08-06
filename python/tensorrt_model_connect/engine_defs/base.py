# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build backend protocol -- all backends implement this interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class BuildBackend(Protocol):
    """Protocol for engine build backends.

    Each backend takes a resolved model directory and produces a .bundle artifact.
    The backend is a build-time concern -- the resulting bundle uses standard
    runtime_strategy values and is indistinguishable at runtime.
    """

    name: str

    def is_available(self) -> bool:
        """Check whether this backend's dependencies are installed."""
        ...

    def build(
        self,
        model_dir: str,
        output_path: str,
        max_cache_length: int = 256,
        *,
        precision: str = "fp16",
        verbose: bool = False,
        parallel_config=None,
    ) -> None:
        """Build a .bundle artifact from a model directory."""
        ...
