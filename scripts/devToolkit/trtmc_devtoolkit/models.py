# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable request, plan, result, and environment-contract models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


PrepareMode = Literal["development", "installed"]
CohortStatus = Literal["supported", "experimental"]


class DevToolkitError(RuntimeError):
    """A user-facing environment preparation error."""


@dataclass(frozen=True)
class ArchitectureContract:
    dockerfile: str
    docker_context: str
    container_python_version: str
    wheel_platform: str
    tensorrt_include_dir: str
    tensorrt_library_dir: str


@dataclass(frozen=True)
class EnvironmentCohort:
    schema_version: int
    id: str
    status: CohortStatus
    tensorrt_version: str
    tensorrt_apt_version: str
    cuda_version: str
    python_versions: tuple[str, ...]
    architectures: dict[str, ArchitectureContract]
    source: Path = field(compare=False)


@dataclass(frozen=True)
class DockerTarget:
    """Prepare a persistent development container backed by a cohort image."""

    gpu: str = "0"
    image: str | None = None
    container_name: str | None = None
    hf_cache: str | None = None
    storage_root: str | None = None
    forward_environment: tuple[str, ...] = (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_HUB_DISABLE_XET",
        "HF_HUB_DOWNLOAD_TIMEOUT",
        "HF_HUB_ETAG_TIMEOUT",
    )
    rebuild_image: bool = False
    kind: Literal["docker"] = "docker"


@dataclass(frozen=True)
class LocalTarget:
    """Prepare a managed venv without modifying system CUDA or TensorRT."""

    python: str = "python3"
    gpu: str = "0"
    kind: Literal["local"] = "local"


@dataclass(frozen=True)
class ModelRequest:
    """Optional model smoke requested after the environment is ready."""

    model_id: str
    revision: str | None = None
    precision: str = "bf16"
    max_cache_length: int = 16384
    prompt: str = "What is the capital of France? Answer in one word."
    max_new_tokens: int = 16


@dataclass(frozen=True)
class PrepareRequest:
    tensorrt: str
    cuda: str
    target: DockerTarget | LocalTarget
    mode: PrepareMode = "development"
    python_version: str = "3.12"
    architecture: str | None = None
    allow_experimental: bool = False
    model: ModelRequest | None = None


@dataclass(frozen=True)
class PlanStep:
    id: str
    description: str
    mutates: bool


@dataclass(frozen=True)
class PreparationPlan:
    request: PrepareRequest
    cohort: EnvironmentCohort
    repository: Path
    architecture: str
    source_revision: str
    run_id: str
    state_dir: Path
    image_fingerprint: str | None
    steps: tuple[PlanStep, ...]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["repository"] = str(self.repository)
        payload["state_dir"] = str(self.state_dir)
        payload["cohort"]["source"] = str(self.cohort.source)  # type: ignore[index]
        return payload


@dataclass(frozen=True)
class ProbeResult:
    name: str
    status: Literal["pass", "fail", "warning"]
    detail: str


@dataclass(frozen=True)
class EnvironmentHandle:
    kind: Literal["docker", "local"]
    fingerprint: str
    trtmc: str
    python: str
    activate_command: str
    image_ref: str | None = None
    container_name: str | None = None
    environment: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PrepareResult:
    plan: PreparationPlan
    environment: EnvironmentHandle
    receipt: Path
    wheel: Path | None = None
    bundle: Path | None = None


@dataclass(frozen=True)
class HandoffPlan:
    name: str
    command: tuple[str, ...]
    environment: dict[str, str]
