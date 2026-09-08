# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Idempotent environment provisioning with core-owned attestation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

from .models import DevToolkitError, ToolchainObservation, ToolchainRuntime
from .providers import FrozenProviderRegistry
from .receipt import exclusive_lock, write_json
from .resolution import EnvironmentLock, ProviderDescriptor
from .runner import Runner


class ProvisionPolicy(Enum):
    ADOPT_ONLY = "adopt-only"
    ADOPT_OR_CREATE = "adopt-or-create"
    CREATE = "create"


def _freeze_json(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(name): _freeze_json(item) for name, item in sorted(value.items())}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item) for item in value)
    raise DevToolkitError(f"Provisioned identity must be JSON-compatible: {type(value).__name__}")


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    frozen = _freeze_json(value)
    assert isinstance(frozen, Mapping)
    return frozen


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(name): _plain_json(item) for name, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


@dataclass(frozen=True)
class ContextHandle:
    """Provider-owned runtime locator; secrets are deliberately not serialized."""

    provider: ProviderDescriptor
    identity: Mapping[str, object]
    execution_identity: Mapping[str, object]
    locator: Mapping[str, object] = field(default_factory=dict, compare=False)
    environment: Mapping[str, str] = field(default_factory=dict, compare=False)
    capabilities: frozenset[str] = frozenset()
    _executor: Callable[[object, bool, bool], object] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _path_mapper: Callable[[object], str] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity", _freeze_mapping(self.identity))
        object.__setattr__(
            self,
            "execution_identity",
            _freeze_mapping(self.execution_identity),
        )
        object.__setattr__(self, "locator", _freeze_mapping(self.locator))
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))
        object.__setattr__(self, "capabilities", frozenset(self.capabilities))

    def execute(
        self,
        command: object,
        *,
        check: bool = True,
        capture_output: bool = False,
    ) -> object:
        if self._executor is None:
            raise DevToolkitError(
                f"Execution context {self.provider.name} does not expose target commands"
            )
        return self._executor(command, check, capture_output)

    def map_path(self, path: object) -> str:
        if self._path_mapper is None:
            raise DevToolkitError(
                f"Execution context {self.provider.name} does not expose target path mapping"
            )
        return self._path_mapper(path)

    @property
    def supports_target_operations(self) -> bool:
        """Whether a materializer can execute commands and map paths on the target."""

        return self._executor is not None and self._path_mapper is not None


@dataclass(frozen=True)
class ToolchainHandle:
    """Provisioned toolchain state independent from an execution context."""

    provider: ProviderDescriptor
    identity: Mapping[str, object]
    runtime: ToolchainRuntime
    environment: Mapping[str, str] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity", _freeze_mapping(self.identity))
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))


@dataclass(frozen=True)
class ProvisionedEnvironment:
    environment_id: str
    lock: EnvironmentLock
    context: ContextHandle
    toolchain: ToolchainHandle
    observation: ToolchainObservation
    state_dir: Path = field(compare=False)
    receipt: Path = field(compare=False)


class AttestationFailed(DevToolkitError):
    """The provisioned environment does not satisfy its immutable lock."""


def _observation_payload(observed: ToolchainObservation) -> dict[str, object]:
    return {
        "python_version": observed.python_version,
        "cuda_version": observed.cuda_version,
        "tensorrt_python_version": observed.tensorrt_python_version,
        "tensorrt_native_version": observed.tensorrt_native_version,
        "tensorrt_header_version": observed.tensorrt_header_version,
        "tensorrt_include_dir": observed.tensorrt_include_dir,
        "tensorrt_library": observed.tensorrt_library,
        "cuda_root": observed.cuda_root,
        "image_id": observed.image_id,
        "architecture": observed.architecture,
        "evidence": dict(observed.evidence),
    }


def _runtime_payload(runtime: ToolchainRuntime) -> dict[str, str]:
    return asdict(runtime)


def _environment_id(
    lock: EnvironmentLock,
    context: ContextHandle,
    toolchain: ToolchainHandle,
    observed: ToolchainObservation,
) -> str:
    payload = {
        "schema_version": 3,
        "lock_id": lock.lock_id,
        "execution_identity": _plain_json(context.execution_identity),
        "toolchain_provider": {
            "name": toolchain.provider.name,
            "implementation": toolchain.provider.implementation,
            "lock_schema": toolchain.provider.lock_schema,
        },
        "toolchain_identity": _plain_json(toolchain.identity),
        "runtime": _runtime_payload(toolchain.runtime),
        "observed": _observation_payload(observed),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(b"trtmc-devtoolkit-environment-v3\0" + encoded).hexdigest()


def _attest(
    lock: EnvironmentLock,
    toolchain: ToolchainHandle,
    observed: ToolchainObservation,
) -> None:
    mismatches: list[str] = []
    expected_trt = lock.toolchain.tensorrt
    for name, actual in (
        ("TensorRT Python", observed.tensorrt_python_version),
        ("TensorRT native", observed.tensorrt_native_version),
        ("TensorRT headers", observed.tensorrt_header_version),
    ):
        if actual != expected_trt:
            mismatches.append(f"{name}: expected {expected_trt}, observed {actual}")
    if observed.cuda_version != lock.toolchain.cuda:
        mismatches.append(f"CUDA: expected {lock.toolchain.cuda}, observed {observed.cuda_version}")
    if observed.python_version != lock.toolchain.python:
        mismatches.append(
            f"Python: expected {lock.toolchain.python}, observed {observed.python_version}"
        )
    if observed.architecture is not None and observed.architecture != lock.context.architecture:
        mismatches.append(
            f"Architecture: expected {lock.context.architecture}, observed {observed.architecture}"
        )
    expected_include = toolchain.runtime.tensorrt_include_dir
    if observed.tensorrt_include_dir != expected_include:
        mismatches.append(
            "TensorRT include directory: "
            f"expected {expected_include}, observed {observed.tensorrt_include_dir}"
        )
    expected_library = toolchain.runtime.tensorrt_library
    if observed.tensorrt_library != expected_library:
        mismatches.append(
            f"TensorRT library: expected {expected_library}, observed {observed.tensorrt_library}"
        )
    expected_cuda_root = toolchain.runtime.cuda_root
    if observed.cuda_root != expected_cuda_root:
        mismatches.append(
            f"CUDA root: expected {expected_cuda_root}, observed {observed.cuda_root}"
        )
    expected_image = lock.context.identity.get("image_id")
    if expected_image is not None and observed.image_id != expected_image:
        mismatches.append(f"Image: expected {expected_image}, observed {observed.image_id}")
    if mismatches:
        raise AttestationFailed("Environment attestation failed: " + "; ".join(mismatches))


class EnvironmentProvisioner:
    def __init__(
        self,
        repository: Path,
        state_root: Path,
        providers: FrozenProviderRegistry,
        runner: Runner,
    ) -> None:
        self.repository = repository
        self.state_root = state_root
        self.providers = providers
        self.runner = runner

    def provision(
        self,
        lock: EnvironmentLock,
        *,
        policy: ProvisionPolicy,
    ) -> ProvisionedEnvironment:
        state_dir = self.state_root / "environments" / lock.lock_id
        state_dir.mkdir(parents=True, exist_ok=True)
        with exclusive_lock(state_dir / ".provision.lock"):
            return self._provision_locked(lock, policy=policy, state_dir=state_dir)

    def _provision_locked(
        self,
        lock: EnvironmentLock,
        *,
        policy: ProvisionPolicy,
        state_dir: Path,
    ) -> ProvisionedEnvironment:
        write_json(state_dir / "environment-lock.json", lock.as_dict())
        (state_dir / "provision-receipt.json").unlink(missing_ok=True)
        (state_dir / "provision-failure.json").unlink(missing_ok=True)
        try:
            context_provider = self.providers.context(lock.context.provider.name)
            toolchain_provider = self.providers.toolchain(lock.toolchain.provider.name)
            if context_provider.descriptor != lock.context.provider:
                raise AttestationFailed(
                    "Registered execution context provider does not match the environment lock"
                )
            if toolchain_provider.descriptor != lock.toolchain.provider:
                raise AttestationFailed(
                    "Registered toolchain provider does not match the environment lock"
                )
            if policy is ProvisionPolicy.ADOPT_ONLY and lock.toolchain.origin == "managed":
                raise AttestationFailed(
                    "adopt-only provisioning cannot materialize a managed toolchain"
                )
            context = context_provider.provision(
                lock.context,
                inherit_system_packages=lock.toolchain.origin != "managed",
                repository=self.repository,
                state_dir=state_dir,
                policy=policy,
                runner=self.runner,
            )
            if context.provider != lock.context.provider:
                raise AttestationFailed(
                    "Provisioned context provider does not match the environment lock"
                )
            if dict(context.identity) != dict(lock.context.identity):
                raise AttestationFailed(
                    "Provisioned context identity does not match the environment lock"
                )
            toolchain = toolchain_provider.provision(
                lock,
                context,
                repository=self.repository,
                state_dir=state_dir,
                runner=self.runner,
            )
            if toolchain.provider != lock.toolchain.provider:
                raise AttestationFailed(
                    "Provisioned toolchain provider does not match the environment lock"
                )
            context = replace(
                context,
                environment={**dict(context.environment), **dict(toolchain.environment)},
            )
            observed = toolchain_provider.observe(
                lock,
                context,
                toolchain,
                repository=self.repository,
                runner=self.runner,
            )
            _attest(lock, toolchain, observed)
            environment_id = _environment_id(lock, context, toolchain, observed)
            (state_dir / "provision-failure.json").unlink(missing_ok=True)
            receipt = write_json(
                state_dir / "provision-receipt.json",
                {
                    "schema_version": 3,
                    "status": "ready",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "lock_id": lock.lock_id,
                    "environment_id": environment_id,
                    "cuda_origin": lock.cuda_origin,
                    "toolchain_source": lock.toolchain.provider.name,
                    "context_provider": lock.context.provider.name,
                    "context_identity": _plain_json(context.identity),
                    "execution_identity": _plain_json(context.execution_identity),
                    "toolchain_identity": _plain_json(toolchain.identity),
                    "toolchain_runtime": _runtime_payload(toolchain.runtime),
                    "qualifications": [
                        {
                            "name": item.name,
                            "digest": item.digest,
                            "status": item.status,
                        }
                        for item in lock.qualifications
                    ],
                    "observed": _observation_payload(observed),
                },
            )
            return ProvisionedEnvironment(
                environment_id=environment_id,
                lock=lock,
                context=context,
                toolchain=toolchain,
                observation=observed,
                state_dir=state_dir,
                receipt=receipt,
            )
        except Exception as error:
            (state_dir / "provision-receipt.json").unlink(missing_ok=True)
            write_json(
                state_dir / "provision-failure.json",
                {
                    "schema_version": 3,
                    "status": "failed",
                    "lock_id": lock.lock_id,
                    "error_type": type(error).__name__,
                },
            )
            raise


def attest_environment(
    environment: ProvisionedEnvironment,
    *,
    repository: Path,
    providers: FrozenProviderRegistry,
    runner: Runner,
) -> ToolchainObservation:
    """Re-observe a provisioned environment before mutable target execution."""
    context_provider = providers.context(environment.context.provider.name)
    toolchain_provider = providers.toolchain(environment.lock.toolchain.provider.name)
    if context_provider.descriptor != environment.lock.context.provider:
        raise AttestationFailed(
            "Registered execution context provider does not match the environment lock"
        )
    if toolchain_provider.descriptor != environment.lock.toolchain.provider:
        raise AttestationFailed("Registered toolchain provider does not match the environment lock")
    if environment.context.provider != environment.lock.context.provider or dict(
        environment.context.identity
    ) != dict(environment.lock.context.identity):
        raise AttestationFailed("Provisioned context does not match the environment lock")
    observed = toolchain_provider.observe(
        environment.lock,
        environment.context,
        environment.toolchain,
        repository=repository,
        runner=runner,
    )
    _attest(environment.lock, environment.toolchain, observed)
    if observed != environment.observation:
        raise AttestationFailed("Environment evidence changed after provisioning")
    expected_id = _environment_id(
        environment.lock,
        environment.context,
        environment.toolchain,
        observed,
    )
    if expected_id != environment.environment_id:
        raise AttestationFailed("Provisioned environment identity is inconsistent")
    return observed
