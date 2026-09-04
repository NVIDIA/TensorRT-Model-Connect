# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit provider registration for composable DevToolkit capabilities."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .models import DevToolkitError

if TYPE_CHECKING:
    from pathlib import Path

    from .models import ToolchainObservation
    from .commands import CommandSpec
    from .provisioning import ContextHandle, ProvisionPolicy, ToolchainHandle
    from .resolution import (
        ContextLock,
        EnvironmentLock,
        EnvironmentRequest,
        ProviderDescriptor,
        ToolchainCandidate,
    )
    from .runner import Runner
    from .targets import TargetHandle, TargetPlan, TargetPolicy


class ToolchainSource(Protocol):
    descriptor: ProviderDescriptor

    def resolve(
        self,
        request: EnvironmentRequest,
        context: ContextLock,
        *,
        repository: Path,
        runner: Runner,
    ) -> tuple[ToolchainCandidate, ...]: ...

    def provision(
        self,
        lock: EnvironmentLock,
        context: ContextHandle,
        *,
        repository: Path,
        state_dir: Path,
        runner: Runner,
    ) -> ToolchainHandle: ...

    def observe(
        self,
        lock: EnvironmentLock,
        context: ContextHandle,
        toolchain: ToolchainHandle,
        *,
        repository: Path,
        runner: Runner,
    ) -> ToolchainObservation: ...


class ToolchainCatalog(Protocol):
    """Resolve version intent into immutable artifacts for a materializer."""

    descriptor: ProviderDescriptor

    def resolve(
        self,
        request: EnvironmentRequest,
        context: ContextLock,
        *,
        repository: Path,
        runner: Runner,
    ) -> tuple[ToolchainCandidate, ...]: ...


class ExecutionContext(Protocol):
    descriptor: ProviderDescriptor

    def resolve(
        self,
        request: EnvironmentRequest,
        *,
        repository: Path,
        runner: Runner,
    ) -> ContextLock: ...

    def provision(
        self,
        context: ContextLock,
        *,
        inherit_system_packages: bool,
        repository: Path,
        state_dir: Path,
        policy: ProvisionPolicy,
        runner: Runner,
    ) -> ContextHandle: ...

    def execute(
        self,
        context: ContextHandle,
        command: CommandSpec,
        *,
        repository: Path,
        state_dir: Path,
        runner: Runner,
        check: bool,
        capture_output: bool,
    ) -> subprocess.CompletedProcess[str]: ...


class TargetProvider(Protocol):
    """Resolve and materialize an execution target independently of toolchains."""

    descriptor: ProviderDescriptor

    def resolve(
        self,
        request: object,
        *,
        repository: Path,
        runner: Runner,
    ) -> TargetPlan: ...

    def provision(
        self,
        plan: TargetPlan,
        *,
        policy: TargetPolicy,
        repository: Path,
        state_dir: Path,
        runner: Runner,
    ) -> TargetHandle: ...

    def attest(
        self,
        target: TargetHandle,
        *,
        repository: Path,
        runner: Runner,
    ) -> None: ...


@dataclass(frozen=True)
class FrozenProviderRegistry:
    contexts: tuple[ExecutionContext, ...]
    toolchains: tuple[ToolchainSource, ...]
    catalogs: tuple[ToolchainCatalog, ...] = ()
    targets: tuple[TargetProvider, ...] = ()

    def context(self, name: str) -> ExecutionContext:
        matches = [provider for provider in self.contexts if provider.descriptor.name == name]
        if len(matches) != 1:
            raise DevToolkitError(f"Unknown execution context provider: {name}")
        return matches[0]

    def toolchain(self, name: str) -> ToolchainSource:
        matches = [provider for provider in self.toolchains if provider.descriptor.name == name]
        if len(matches) != 1:
            raise DevToolkitError(f"Unknown toolchain source provider: {name}")
        return matches[0]

    def target(self, name: str) -> TargetProvider:
        matches = [provider for provider in self.targets if provider.descriptor.name == name]
        if len(matches) != 1:
            raise DevToolkitError(f"Unknown target provider: {name}")
        return matches[0]


class ProviderRegistry:
    """Mutable wiring object that is frozen when a toolkit is constructed."""

    def __init__(self) -> None:
        self._contexts: dict[str, ExecutionContext] = {}
        self._toolchains: dict[str, ToolchainSource] = {}
        self._catalogs: dict[str, ToolchainCatalog] = {}
        self._targets: dict[str, TargetProvider] = {}

    @classmethod
    def with_builtins(cls) -> ProviderRegistry:
        from .builtin_providers import (
            ContainerImageToolchainSource,
            DockerExecutionContext,
            LocalExecutionContext,
            ManagedArtifactToolchainSource,
            PrefixToolchainSource,
            SystemToolchainSource,
        )
        from .catalogs import NvidiaPackageIndexCatalog
        from .targets import DockerTargetProvider

        registry = cls()
        registry.register_context(LocalExecutionContext())
        registry.register_context(DockerExecutionContext())
        registry.register_toolchain(SystemToolchainSource())
        registry.register_toolchain(PrefixToolchainSource())
        registry.register_toolchain(ContainerImageToolchainSource())
        registry.register_toolchain(ManagedArtifactToolchainSource())
        registry.register_catalog(NvidiaPackageIndexCatalog())
        registry.register_target(DockerTargetProvider())
        return registry

    def register_context(self, provider: ExecutionContext) -> None:
        self._register(self._contexts, provider, "execution context")

    def register_toolchain(self, provider: ToolchainSource) -> None:
        self._register(self._toolchains, provider, "toolchain")

    def register_catalog(self, provider: ToolchainCatalog) -> None:
        self._register(self._catalogs, provider, "toolchain catalog")

    def register_target(self, provider: TargetProvider) -> None:
        self._register(self._targets, provider, "target")

    def freeze(self) -> FrozenProviderRegistry:
        return FrozenProviderRegistry(
            contexts=tuple(self._contexts.values()),
            toolchains=tuple(self._toolchains.values()),
            catalogs=tuple(self._catalogs.values()),
            targets=tuple(self._targets.values()),
        )

    @staticmethod
    def _register(registry: dict[str, object], provider: object, kind: str) -> None:
        descriptor = getattr(provider, "descriptor", None)
        name = getattr(descriptor, "name", None)
        if not isinstance(name, str) or not name:
            raise DevToolkitError(f"{kind} provider must declare a non-empty descriptor name")
        if name in registry:
            raise DevToolkitError(f"Duplicate {kind} provider: {name}")
        registry[name] = provider
