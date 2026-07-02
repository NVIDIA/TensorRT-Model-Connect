# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified E2E testing harness for TRT inference validation.

This package provides a DIP-first framework where high-level orchestration
depends only on abstract contracts, and concrete implementations (strategy
runners, reference backends, comparators) are pluggable adapters.

Public API re-exports the core types from contracts.py.
"""

__version__ = "0.1.0"

import os


def _case_artifact_dir(artifacts_dir: str, case_name: str) -> str:
    """Return per-model artifact subdirectory, creating it if needed."""
    if case_name:
        d = os.path.join(artifacts_dir, case_name)
    else:
        d = artifacts_dir
    os.makedirs(d, exist_ok=True)
    return d


def save_full_stderr(stderr: str, artifacts_dir: str, stage_name: str, case_name: str = "") -> tuple:
    """Write full stderr to file, return (truncated_msg, file_path) or (truncated_msg, None) if no artifacts_dir."""
    truncated = stderr[-2000:] if len(stderr) > 2000 else stderr
    if not artifacts_dir:
        return truncated, None
    d = _case_artifact_dir(artifacts_dir, case_name)
    path = os.path.join(d, f"{stage_name}_stderr.log")
    with open(path, "w") as f:
        f.write(stderr)
    return truncated, path

from .contracts import (  # noqa: E402
    ArtifactSink,
    ArtifactType,
    CILane,
    Comparator,
    CompareResult,
    ComparisonMode,
    DeterminismPolicy,
    E2ECase,
    E2EResult,
    E2EStatus,
    FailureType,
    MetricResult,
    OracleLevel,
    PreflightRequirement,
    ReferenceBackendRunner,
    ReferenceFamily,
    RunContext,
    StageOutput,
    StageSpec,
    StageStatus,
    TaskStrategyRunner,
    ThresholdProfile,
    UserContract,
)

__all__ = [
    "__version__",
    # Enums
    "FailureType",
    "OracleLevel",
    "E2EStatus",
    "StageStatus",
    "ReferenceFamily",
    "ArtifactType",
    "ComparisonMode",
    "CILane",
    "UserContract",
    # Dataclasses
    "PreflightRequirement",
    "StageSpec",
    "ThresholdProfile",
    "MetricResult",
    "E2ECase",
    "StageOutput",
    "CompareResult",
    "E2EResult",
    "RunContext",
    # Protocols
    "TaskStrategyRunner",
    "ReferenceBackendRunner",
    "Comparator",
    "ArtifactSink",
    "DeterminismPolicy",
    # Helpers
    "save_full_stderr",
    "_case_artifact_dir",
]
