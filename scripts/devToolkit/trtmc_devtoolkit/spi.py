# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extension interfaces for DevToolkit adapters and build recipes."""

from .building import BuildContext, BuildPlan, BuildRecipe
from .providers import (
    ExecutionContext,
    FrozenProviderRegistry,
    ProviderRegistry,
    ToolchainSource,
)
from .provisioning import ContextHandle, ToolchainHandle
from .qualifications import (
    QualificationRecord,
    QualificationRegistry,
    QualificationSource,
)
from .resolution import ContextLock, ProviderDescriptor, ToolchainCandidate

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
    "QualificationRecord",
    "QualificationRegistry",
    "QualificationSource",
    "ToolchainCandidate",
    "ToolchainHandle",
    "ToolchainSource",
]
