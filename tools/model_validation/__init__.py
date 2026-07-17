# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Foundational contracts for the incremental Task Eval architecture migration."""

from .compatibility import LegacyTaskEvalFacade
from .contracts import (
    Assessment,
    AssessmentStatus,
    CasePlan,
    CompatibilityMode,
    PlanIntegrityError,
    SuiteContract,
    ValidationPlan,
    ValidationRequest,
    WorkloadResolution,
    WorkloadSpec,
)
from .planner import (
    UnsupportedLegacyPerformanceError,
    compile_legacy_plan,
    compile_native_plan,
)
from .registry import TaskAdapterRegistry, UnknownTaskAdapterError

__all__ = [
    "Assessment",
    "AssessmentStatus",
    "CasePlan",
    "CompatibilityMode",
    "LegacyTaskEvalFacade",
    "PlanIntegrityError",
    "SuiteContract",
    "TaskAdapterRegistry",
    "UnknownTaskAdapterError",
    "UnsupportedLegacyPerformanceError",
    "ValidationPlan",
    "ValidationRequest",
    "WorkloadResolution",
    "WorkloadSpec",
    "compile_legacy_plan",
    "compile_native_plan",
]
