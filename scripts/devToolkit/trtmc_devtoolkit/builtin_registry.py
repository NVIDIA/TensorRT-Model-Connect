# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composition root for the built-in DevToolkit adapters."""

from __future__ import annotations

from .builtin_providers import (
    ContainerImageToolchainSource,
    DockerExecutionContext,
    LocalExecutionContext,
    ManagedArtifactToolchainSource,
    PrefixToolchainSource,
    SystemToolchainSource,
)
from .catalogs import NvidiaPackageIndexCatalog
from .docker_target import DockerTargetProvider
from .providers import ProviderRegistry


def builtin_provider_registry() -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register_context(LocalExecutionContext())
    registry.register_context(DockerExecutionContext())
    registry.register_toolchain(SystemToolchainSource())
    registry.register_toolchain(PrefixToolchainSource())
    registry.register_toolchain(ContainerImageToolchainSource())
    registry.register_toolchain(ManagedArtifactToolchainSource())
    registry.register_catalog(NvidiaPackageIndexCatalog())
    registry.register_target(DockerTargetProvider())
    return registry
