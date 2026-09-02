# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Idempotent environment provisioning with core-owned attestation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from .models import DevToolkitError, ToolchainObservation
from .providers import FrozenProviderRegistry
from .receipt import write_json
from .resolution import EnvironmentLock, ProviderDescriptor
from .runner import Runner


class ProvisionPolicy(Enum):
    ADOPT_ONLY = "adopt-only"
    ADOPT_OR_CREATE = "adopt-or-create"
    CREATE = "create"


@dataclass(frozen=True)
class ContextHandle:
    """Provider-owned runtime locator; secrets are deliberately not serialized."""

    provider: ProviderDescriptor
    identity: Mapping[str, object]
    locator: Mapping[str, object] = field(default_factory=dict, compare=False)
    environment: Mapping[str, str] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity", MappingProxyType(dict(self.identity)))
        object.__setattr__(self, "locator", MappingProxyType(dict(self.locator)))
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))


@dataclass(frozen=True)
class ProvisionedEnvironment:
    environment_id: str
    lock: EnvironmentLock
    context: ContextHandle
    observation: ToolchainObservation
    state_dir: Path = field(compare=False)
    receipt: Path = field(compare=False)


class AttestationFailed(DevToolkitError):
    """The provisioned environment does not satisfy its immutable lock."""


def _attest(lock: EnvironmentLock, observed: ToolchainObservation) -> None:
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
    expected_include = lock.toolchain.identity.get("tensorrt_include_dir")
    if expected_include is not None and observed.tensorrt_include_dir != expected_include:
        mismatches.append(
            "TensorRT include directory: "
            f"expected {expected_include}, observed {observed.tensorrt_include_dir}"
        )
    expected_library = lock.toolchain.identity.get("tensorrt_library")
    if expected_library is not None and observed.tensorrt_library != expected_library:
        mismatches.append(
            f"TensorRT library: expected {expected_library}, observed {observed.tensorrt_library}"
        )
    expected_cuda_root = lock.toolchain.identity.get("cuda_root") or lock.toolchain.identity.get(
        "system_cuda_root"
    )
    if expected_cuda_root is not None and observed.cuda_root != expected_cuda_root:
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
        write_json(state_dir / "environment-lock.json", lock.as_dict())
        context_provider = self.providers.context(lock.context.provider.name)
        toolchain_provider = self.providers.toolchain(lock.toolchain.provider.name)
        try:
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
                lock,
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
            context = toolchain_provider.provision(
                lock,
                context,
                execution=context_provider,
                repository=self.repository,
                state_dir=state_dir,
                runner=self.runner,
            )
            if context.provider != lock.context.provider:
                raise AttestationFailed(
                    "Toolchain provisioning changed the execution context provider"
                )
            if dict(context.identity) != dict(lock.context.identity):
                raise AttestationFailed(
                    "Toolchain provisioning changed the resolved context identity"
                )
            observed = toolchain_provider.observe(
                lock,
                context,
                execution=context_provider,
                repository=self.repository,
                runner=self.runner,
            )
            _attest(lock, observed)
            (state_dir / "provision-failure.json").unlink(missing_ok=True)
            receipt = write_json(
                state_dir / "provision-receipt.json",
                {
                    "schema_version": 2,
                    "status": "ready",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "lock_id": lock.lock_id,
                    "environment_id": lock.lock_id,
                    "cuda_origin": lock.cuda_origin,
                    "toolchain_source": lock.toolchain.provider.name,
                    "context_provider": lock.context.provider.name,
                    "context_identity": dict(context.identity),
                    "qualifications": [
                        {
                            "name": item.name,
                            "digest": item.digest,
                            "status": item.status,
                        }
                        for item in lock.qualifications
                    ],
                    "observed": asdict(observed),
                },
            )
            return ProvisionedEnvironment(
                environment_id=lock.lock_id,
                lock=lock,
                context=context,
                observation=observed,
                state_dir=state_dir,
                receipt=receipt,
            )
        except Exception as error:
            (state_dir / "provision-receipt.json").unlink(missing_ok=True)
            write_json(
                state_dir / "provision-failure.json",
                {
                    "schema_version": 2,
                    "status": "failed",
                    "lock_id": lock.lock_id,
                    "error_type": type(error).__name__,
                },
            )
            raise
