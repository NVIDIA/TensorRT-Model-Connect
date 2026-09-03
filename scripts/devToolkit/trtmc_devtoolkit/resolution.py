# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only resolution of environment intent into an immutable lock."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from .models import DevToolkitError, ToolchainRuntime
from .providers import FrozenProviderRegistry
from .qualifications import QualificationRef, QualificationRegistry
from .runner import Runner


JsonScalar = str | int | float | bool | None
FrozenJson = JsonScalar | tuple[object, ...] | Mapping[str, object]
ToolchainOrigin = Literal["system", "managed", "image", "prefix"]
CudaSource = Literal["system", "managed", "image", "prefix"]
CudaOrigin = Literal["system", "managed-default", "explicit"]
ResolutionKind = Literal["system-first", "exact", "system-only", "managed"]


def _freeze_json(value: object) -> FrozenJson:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item) for item in value)
    raise DevToolkitError(f"Provider state must be JSON-compatible, got {type(value).__name__}")


def _plain_json(value: FrozenJson) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


@dataclass(frozen=True)
class ProviderDescriptor:
    name: str
    implementation: str
    lock_schema: int

    def __post_init__(self) -> None:
        if not self.name or not self.implementation or self.lock_schema < 1:
            raise DevToolkitError("Provider descriptors require name, implementation, and schema")


@dataclass(frozen=True)
class CudaPolicy:
    kind: ResolutionKind
    version: str | None = None
    fallback: str | None = None

    @classmethod
    def system_first(cls, *, fallback: str = "13.3") -> CudaPolicy:
        return cls("system-first", fallback=fallback)

    @classmethod
    def exact(cls, version: str) -> CudaPolicy:
        return cls("exact", version=version)

    @classmethod
    def system_only(cls, version: str | None = None) -> CudaPolicy:
        return cls("system-only", version=version)

    @classmethod
    def managed(cls, version: str) -> CudaPolicy:
        return cls("managed", version=version)

    def __post_init__(self) -> None:
        if self.kind not in {"system-first", "exact", "system-only", "managed"}:
            raise DevToolkitError(f"Unsupported CUDA policy kind: {self.kind}")
        if self.kind == "system-first" and not self.fallback:
            raise DevToolkitError("system-first CUDA policy requires a fallback version")
        if self.kind in {"exact", "managed"} and not self.version:
            raise DevToolkitError(f"{self.kind} CUDA policy requires an exact version")
        for value in (self.version, self.fallback):
            if value is not None and re.fullmatch(r"[0-9]+\.[0-9]+", value) is None:
                raise DevToolkitError("CUDA versions must have exact major.minor form")


@dataclass(frozen=True)
class ExecutionTarget:
    provider: str
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.provider:
            raise DevToolkitError("Execution target requires a provider name")
        object.__setattr__(self, "options", _freeze_json(self.options))

    @classmethod
    def local(
        cls,
        *,
        python: str = "python3",
        gpu: str = "0",
    ) -> ExecutionTarget:
        return cls("local", {"python": python, "gpu": gpu})

    @classmethod
    def docker(
        cls,
        *,
        python: str = "python3",
        docker_context: str | None = None,
        container: str | None = None,
        workspace: str = "/workspace/tensorrt-model-connect",
        state: str = "/tmp/trtmc-devtoolkit",
    ) -> ExecutionTarget:
        options: dict[str, object] = {
            "python": python,
            "workspace": workspace,
            "state": state,
        }
        if docker_context is not None:
            options["docker_context"] = docker_context
        if container is not None:
            options["container"] = container
        return cls("docker", options)


@dataclass(frozen=True)
class EnvironmentRequest:
    tensorrt: str
    target: ExecutionTarget
    cuda: CudaPolicy = field(default_factory=CudaPolicy.system_first)
    python: str = "3.12"
    architecture: str | None = None
    toolchain: str | None = None
    toolchain_options: Mapping[str, object] = field(default_factory=dict)
    preset: str | None = None
    require_qualification: bool = False
    artifacts: tuple[ArtifactPin, ...] = ()

    def __post_init__(self) -> None:
        value = self.tensorrt.strip()
        if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+", value) is None:
            raise DevToolkitError(
                "TensorRT must be one exact four-part version, not a range or alias"
            )
        object.__setattr__(self, "tensorrt", value)
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "toolchain_options", _freeze_json(self.toolchain_options))
        names = [artifact.name for artifact in self.artifacts]
        if len(names) != len(set(names)):
            raise DevToolkitError("Artifact pin names must be unique")


@dataclass(frozen=True)
class ArtifactPin:
    name: str
    uri: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.name or not self.uri:
            raise DevToolkitError("Artifact pins require a name and URI")
        parsed = urllib.parse.urlsplit(self.uri)
        if parsed.username is not None or parsed.password is not None:
            raise DevToolkitError(f"Artifact {self.name} URI cannot contain credentials")
        if re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None:
            raise DevToolkitError(f"Artifact {self.name} requires a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ContextLock:
    """Resolved context identity, semantic execution mapping, and private locator."""

    provider: ProviderDescriptor
    operating_system: str
    architecture: str
    identity: Mapping[str, object]
    execution: Mapping[str, object] = field(default_factory=dict)
    locator: Mapping[str, object] = field(default_factory=dict, compare=False)
    capabilities: frozenset[str] = frozenset()
    qualification: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity", _freeze_json(self.identity))
        object.__setattr__(self, "execution", _freeze_json(self.execution))
        object.__setattr__(self, "locator", _freeze_json(self.locator))
        capabilities = frozenset(self.capabilities)
        if any(not isinstance(value, str) or not value for value in capabilities):
            raise DevToolkitError("Context capabilities must be non-empty strings")
        object.__setattr__(self, "capabilities", capabilities)
        qualification = dict(self.qualification)
        if any(
            not isinstance(name, str) or not name or not isinstance(value, str) or not value
            for name, value in qualification.items()
        ):
            raise DevToolkitError("Context qualification facts must be non-empty strings")
        object.__setattr__(self, "qualification", MappingProxyType(qualification))


@dataclass(frozen=True)
class ToolchainCandidate:
    provider: ProviderDescriptor
    origin: ToolchainOrigin
    tensorrt: str
    cuda: str
    python: str
    identity: Mapping[str, object]
    runtime: ToolchainRuntime | None = None
    artifacts: tuple[ArtifactPin, ...] = ()
    cuda_source: CudaSource | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity", _freeze_json(self.identity))
        if self.cuda_source is None:
            object.__setattr__(self, "cuda_source", self.origin)
        if self.origin not in {"system", "managed", "image", "prefix"}:
            raise DevToolkitError(f"Unsupported toolchain origin: {self.origin}")
        if self.cuda_source not in {"system", "managed", "image", "prefix"}:
            raise DevToolkitError(f"Unsupported CUDA source: {self.cuda_source}")
        if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+", self.tensorrt) is None:
            raise DevToolkitError("Toolchain candidates require exact four-part TensorRT")
        if re.fullmatch(r"[0-9]+\.[0-9]+", self.cuda) is None:
            raise DevToolkitError("Toolchain candidates require exact major.minor CUDA")
        if re.fullmatch(r"[0-9]+\.[0-9]+", self.python) is None:
            raise DevToolkitError("Toolchain candidates require exact major.minor Python")
        if self.origin == "managed" and not self.artifacts:
            raise DevToolkitError(
                f"Managed toolchain candidate {self.provider.name} requires trusted artifacts"
            )
        if self.origin != "managed" and self.runtime is None:
            raise DevToolkitError(
                f"Adopted toolchain candidate {self.provider.name} requires a runtime"
            )


@dataclass(frozen=True)
class EnvironmentLock:
    schema_version: Literal[3]
    lock_id: str
    request: EnvironmentRequest
    context: ContextLock
    toolchain: ToolchainCandidate
    cuda_origin: CudaOrigin
    qualifications: tuple[QualificationRef, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != 3 or self.lock_id != _environment_lock_id(
            self.context, self.toolchain
        ):
            raise DevToolkitError("Environment lock ID does not match its resolved identity")

    @property
    def tensorrt(self) -> str:
        return self.toolchain.tensorrt

    @property
    def cuda(self) -> str:
        return self.toolchain.cuda

    def as_dict(self) -> dict[str, object]:
        return {
            **_resolved_payload(self.context, self.toolchain),
            "lock_id": self.lock_id,
            "cuda_origin": self.cuda_origin,
            "request": {
                "tensorrt": self.request.tensorrt,
                "cuda": {
                    "kind": self.request.cuda.kind,
                    "version": self.request.cuda.version,
                    "fallback": self.request.cuda.fallback,
                },
                "python": self.request.python,
                "architecture": self.request.architecture,
                "target": {
                    "provider": self.request.target.provider,
                    "option_names": sorted(self.request.target.options),
                },
                "toolchain": self.request.toolchain,
                "toolchain_option_names": sorted(self.request.toolchain_options),
                "preset": self.request.preset,
                "require_qualification": self.request.require_qualification,
            },
            "qualifications": [
                {
                    "name": item.name,
                    "digest": item.digest,
                    "status": item.status,
                }
                for item in self.qualifications
            ],
        }


class ResolutionError(DevToolkitError):
    def __init__(self, message: str, *, attempts: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.attempts = attempts


class ArtifactUnavailable(ResolutionError):
    """No provider could supply an environment matching the request."""


class IncompatibleCombination(ResolutionError):
    """Providers found candidates, but none formed one unambiguous exact match."""


class EnvironmentResolver:
    def __init__(
        self,
        repository: Path,
        providers: FrozenProviderRegistry,
        runner: Runner,
        qualifications: QualificationRegistry | None = None,
    ):
        self.repository = repository
        self.providers = providers
        self.runner = runner
        self.qualifications = qualifications or QualificationRegistry(())

    def resolve(self, request: EnvironmentRequest) -> EnvironmentLock:
        context_provider = self.providers.context(request.target.provider)
        context = context_provider.resolve(
            request,
            repository=self.repository,
            runner=self.runner,
        )
        sources = tuple(
            source
            for source in self.providers.toolchains
            if request.toolchain is None or source.descriptor.name == request.toolchain
        )
        if not sources:
            raise ResolutionError("No toolchain source is registered for this request")
        candidates: list[ToolchainCandidate] = []
        attempts: list[str] = []
        for source in sources:
            resolved = source.resolve(
                request,
                context,
                repository=self.repository,
                runner=self.runner,
            )
            candidates.extend(resolved)
            attempts.append(f"{source.descriptor.name}: {len(resolved)} candidate(s)")
        selected, decision = self._select(request, tuple(candidates), tuple(attempts))
        qualifications = self._qualifications(request, context, selected)
        lock_id = self._lock_id(request, context, selected)
        return EnvironmentLock(
            3,
            lock_id,
            request,
            context,
            selected,
            decision,
            qualifications,
        )

    def _qualifications(
        self,
        request: EnvironmentRequest,
        context: ContextLock,
        candidate: ToolchainCandidate,
    ) -> tuple[QualificationRef, ...]:
        try:
            records = self.qualifications.load_all()
        except DevToolkitError:
            if request.preset is not None or request.require_qualification:
                raise
            return ()
        if request.preset is not None:
            named = [record for record in records if record.reference.name == request.preset]
            if len(named) != 1 or not named[0].matches(context, candidate):
                raise IncompatibleCombination(
                    f"Requested qualification {request.preset!r} does not match the environment"
                )
            return (named[0].reference,)
        matches = tuple(
            record.reference for record in records if record.matches(context, candidate)
        )
        if request.require_qualification and not matches:
            raise IncompatibleCombination(
                "Environment resolution requires a matching qualification record"
            )
        return matches

    @staticmethod
    def _select(
        request: EnvironmentRequest,
        candidates: tuple[ToolchainCandidate, ...],
        attempts: tuple[str, ...],
    ) -> tuple[ToolchainCandidate, CudaOrigin]:
        exact_trt = [
            candidate
            for candidate in candidates
            if candidate.tensorrt == request.tensorrt and candidate.python == request.python
        ]
        policy = request.cuda
        existing_sources = {"system", "image", "prefix"}
        found_any = bool(candidates)
        if policy.kind == "system-first":
            existing = [
                candidate for candidate in exact_trt if candidate.cuda_source in existing_sources
            ]
            if existing:
                complete = [
                    candidate for candidate in existing if candidate.origin in existing_sources
                ]
                return (
                    EnvironmentResolver._one(complete or existing, attempts, found_any),
                    "system",
                )
            managed = [
                candidate
                for candidate in exact_trt
                if candidate.cuda_source == "managed" and candidate.cuda == policy.fallback
            ]
            if not managed and not found_any:
                raise ArtifactUnavailable(
                    "No complete target CUDA/toolchain was found, and the managed "
                    f"CUDA {policy.fallback} fallback has no digest-pinned artifacts. "
                    "Supply EnvironmentRequest.artifacts or register a ToolchainSource.",
                    attempts=attempts,
                )
            return EnvironmentResolver._one(managed, attempts, found_any), "managed-default"
        if policy.kind == "system-only":
            existing = [
                candidate
                for candidate in exact_trt
                if candidate.cuda_source in existing_sources
                and (policy.version is None or candidate.cuda == policy.version)
            ]
            complete = [candidate for candidate in existing if candidate.origin in existing_sources]
            return (
                EnvironmentResolver._one(complete or existing, attempts, found_any),
                "system",
            )
        if policy.kind == "managed":
            managed = [
                candidate
                for candidate in exact_trt
                if candidate.cuda_source == "managed" and candidate.cuda == policy.version
            ]
            return EnvironmentResolver._one(managed, attempts, found_any), "explicit"
        matches = [candidate for candidate in exact_trt if candidate.cuda == policy.version]
        existing = [candidate for candidate in matches if candidate.cuda_source in existing_sources]
        complete = [candidate for candidate in existing if candidate.origin in existing_sources]
        return (
            EnvironmentResolver._one(complete or existing or matches, attempts, found_any),
            "explicit",
        )

    @staticmethod
    def _one(
        candidates: list[ToolchainCandidate],
        attempts: tuple[str, ...],
        found_any: bool,
    ) -> ToolchainCandidate:
        if len(candidates) != 1:
            reason = "no exact candidate" if not candidates else "ambiguous exact candidates"
            error_type = IncompatibleCombination if found_any else ArtifactUnavailable
            raise error_type(
                f"Environment resolution failed: {reason}",
                attempts=attempts,
            )
        return candidates[0]

    @staticmethod
    def _lock_id(
        request: EnvironmentRequest,
        context: ContextLock,
        candidate: ToolchainCandidate,
    ) -> str:
        del request
        return _environment_lock_id(context, candidate)


def _provider_payload(provider: ProviderDescriptor) -> dict[str, object]:
    return {
        "name": provider.name,
        "implementation": provider.implementation,
        "lock_schema": provider.lock_schema,
    }


def _identity_payload(
    context: ContextLock,
    candidate: ToolchainCandidate,
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "context": {
            "provider": _provider_payload(context.provider),
            "operating_system": context.operating_system,
            "architecture": context.architecture,
            "identity": _plain_json(context.identity),
            "execution": _plain_json(context.execution),
            "capabilities": sorted(context.capabilities),
            "qualification": dict(context.qualification),
        },
        "toolchain": {
            "provider": _provider_payload(candidate.provider),
            "origin": candidate.origin,
            "cuda_source": candidate.cuda_source,
            "tensorrt": candidate.tensorrt,
            "cuda": candidate.cuda,
            "python": candidate.python,
            "identity": _plain_json(candidate.identity),
            "runtime": (
                {
                    "python_executable": candidate.runtime.python_executable,
                    "cuda_root": candidate.runtime.cuda_root,
                    "nvcc": candidate.runtime.nvcc,
                    "tensorrt_include_dir": candidate.runtime.tensorrt_include_dir,
                    "tensorrt_library": candidate.runtime.tensorrt_library,
                }
                if candidate.runtime is not None
                else None
            ),
            "artifacts": [
                {
                    "name": artifact.name,
                    "sha256": artifact.sha256,
                }
                for artifact in candidate.artifacts
            ],
        },
    }


def _environment_lock_id(
    context: ContextLock,
    candidate: ToolchainCandidate,
) -> str:
    payload = _identity_payload(context, candidate)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(b"trtmc-devtoolkit-environment-lock-v3\0" + encoded).hexdigest()


def _resolved_payload(
    context: ContextLock,
    candidate: ToolchainCandidate,
) -> dict[str, object]:
    payload = _identity_payload(context, candidate)
    context_payload = payload["context"]
    toolchain_payload = payload["toolchain"]
    assert isinstance(context_payload, dict)
    assert isinstance(toolchain_payload, dict)
    context_payload["locator_names"] = sorted(context.locator)
    toolchain_payload["artifacts"] = [
        {
            "name": artifact.name,
            "uri": urllib.parse.urlunsplit(
                urllib.parse.urlsplit(artifact.uri)._replace(query="", fragment="")
            ),
            "sha256": artifact.sha256,
        }
        for artifact in candidate.artifacts
    ]
    return payload
