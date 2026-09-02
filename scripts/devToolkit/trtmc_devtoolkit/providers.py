# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit provider registration for composable DevToolkit capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .models import DevToolkitError

if TYPE_CHECKING:
    from pathlib import Path

    from .models import ToolchainObservation
    from .commands import CommandSpec
    from .provisioning import ContextHandle, ProvisionPolicy
    from .resolution import (
        ContextLock,
        EnvironmentLock,
        EnvironmentRequest,
        ProviderDescriptor,
        ToolchainCandidate,
    )
    from .runner import Runner


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
        execution: ExecutionContext,
        repository: Path,
        state_dir: Path,
        runner: Runner,
    ) -> ContextHandle: ...

    def observe(
        self,
        lock: EnvironmentLock,
        context: ContextHandle,
        *,
        execution: ExecutionContext,
        repository: Path,
        runner: Runner,
    ) -> ToolchainObservation: ...


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
        lock: EnvironmentLock,
        *,
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
    ): ...


@dataclass(frozen=True)
class FrozenProviderRegistry:
    contexts: tuple[ExecutionContext, ...]
    toolchains: tuple[ToolchainSource, ...]

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


class ProviderRegistry:
    """Mutable wiring object that is frozen when a toolkit is constructed."""

    def __init__(self) -> None:
        self._contexts: dict[str, ExecutionContext] = {}
        self._toolchains: dict[str, ToolchainSource] = {}

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

        registry = cls()
        registry.register_context(LocalExecutionContext())
        registry.register_context(DockerExecutionContext())
        registry.register_toolchain(SystemToolchainSource())
        registry.register_toolchain(PrefixToolchainSource())
        registry.register_toolchain(ContainerImageToolchainSource())
        registry.register_toolchain(ManagedArtifactToolchainSource())
        return registry

    def register_context(self, provider: ExecutionContext) -> None:
        self._register(self._contexts, provider, "execution context")

    def register_toolchain(self, provider: ToolchainSource) -> None:
        self._register(self._toolchains, provider, "toolchain")

    def freeze(self) -> FrozenProviderRegistry:
        return FrozenProviderRegistry(
            contexts=tuple(self._contexts.values()),
            toolchains=tuple(self._toolchains.values()),
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
