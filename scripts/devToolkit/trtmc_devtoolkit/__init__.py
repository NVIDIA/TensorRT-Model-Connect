# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Repository-local environment preparation API for TensorRT-Model-Connect."""

from .api import DevToolkit
from .building import BuildArtifact, BuildResult, BuildSpec, SourceSnapshot
from .commands import (
    CommandArgument,
    CommandResult,
    CommandSpec,
    EnvironmentPath,
    repository_path,
    state_path,
    target_path,
)
from .handoff import performance_handoff, profiling_handoff, validation_handoff
from .models import (
    DockerTarget,
    EnvironmentHandle,
    HandoffPlan,
    LocalTarget,
    PrepareRequest,
    PrepareResult,
    PreparationPlan,
    ToolchainObservation,
)
from .providers import (
    ExecutionContext,
    FrozenProviderRegistry,
    ProviderRegistry,
    ToolchainSource,
)
from .provisioning import (
    AttestationFailed,
    ContextHandle,
    ProvisionedEnvironment,
    ProvisionPolicy,
)
from .qualifications import QualificationRef, QualificationRegistry
from .resolution import (
    ArtifactPin,
    ArtifactUnavailable,
    ContextLock,
    CudaPolicy,
    EnvironmentLock,
    EnvironmentRequest,
    ExecutionTarget,
    IncompatibleCombination,
    ProviderDescriptor,
    ResolutionError,
    ToolchainCandidate,
)

__all__ = [
    "DevToolkit",
    "DockerTarget",
    "EnvironmentHandle",
    "HandoffPlan",
    "LocalTarget",
    "PrepareRequest",
    "PrepareResult",
    "PreparationPlan",
    "ArtifactPin",
    "ArtifactUnavailable",
    "AttestationFailed",
    "BuildArtifact",
    "BuildResult",
    "BuildSpec",
    "CommandArgument",
    "CommandResult",
    "CommandSpec",
    "ContextHandle",
    "ContextLock",
    "CudaPolicy",
    "EnvironmentLock",
    "EnvironmentPath",
    "EnvironmentRequest",
    "ExecutionTarget",
    "ExecutionContext",
    "FrozenProviderRegistry",
    "IncompatibleCombination",
    "ProviderDescriptor",
    "ProviderRegistry",
    "ProvisionedEnvironment",
    "ProvisionPolicy",
    "QualificationRef",
    "QualificationRegistry",
    "ResolutionError",
    "SourceSnapshot",
    "ToolchainCandidate",
    "ToolchainObservation",
    "ToolchainSource",
    "repository_path",
    "state_path",
    "target_path",
    "performance_handoff",
    "profiling_handoff",
    "validation_handoff",
]
