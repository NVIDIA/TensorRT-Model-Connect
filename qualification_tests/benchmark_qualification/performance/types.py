# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stable values exchanged by Performance Qualification modules."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class PerfMatrixError(RuntimeError):
    pass


@dataclass(frozen=True)
class Environment:
    name: str
    trtmc_bench: Path
    worker: Path
    hf_runner: Path
    task_runner: Path
    results_root: Path
    scratch_root: Path
    bundle_cache: Path
    bundle_roots: tuple[Path, ...]
    runtime_root: Path
    bundle_retention: str
    local_files_only: bool
    timeout_seconds: int
    references: Mapping[str, str]
    reference_python: Path = Path(sys.executable)
    storage_root: Path | None = None
    hf_cache_mode: str = "shared"
    hf_cache_retention: str = "retain"


@dataclass(frozen=True)
class ResolvedEntry:
    spec: Mapping[str, Any]
    model: Any
    case: Any
    manifest: Mapping[str, Any]
    reference_precision: str
    baseline_timing: Mapping[str, Any]
