# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extension interfaces for DevToolkit adapters and build recipes."""

from .building import BuildContext, BuildPlan, BuildRecipe
from .providers import (
    ExecutionContext,
    FrozenProviderRegistry,
    ProviderRegistry,
    TargetProvider,
    ToolchainCatalog,
    ToolchainSource,
)
from .provisioning import ContextHandle, ToolchainHandle
from .qualifications import (
    QualificationRecord,
    QualificationRegistry,
    QualificationSource,
)
from .resolution import ContextLock, ProviderDescriptor, ToolchainCandidate
from .targets import TargetHandle, TargetPlan

__all__ = [
    "BuildContext",
    "BuildPlan",
    "BuildRecipe",
    "ContextHandle",
    "ContextLock",
    "ExecutionContext",
    "FrozenProviderRegistry",
    "ProviderDescriptor",
    "ProviderRegistry",
    "TargetProvider",
    "TargetHandle",
    "TargetPlan",
    "QualificationRecord",
    "QualificationRegistry",
    "QualificationSource",
    "ToolchainCandidate",
    "ToolchainCatalog",
    "ToolchainHandle",
    "ToolchainSource",
]
