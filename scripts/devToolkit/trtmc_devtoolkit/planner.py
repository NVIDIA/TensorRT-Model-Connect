# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve one immutable request into a deterministic preparation plan."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict
from pathlib import Path

from .cohorts import CohortRegistry, normalize_architecture
from .models import DevToolkitError, PlanStep, PrepareRequest, PreparationPlan


_SOURCE_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


def source_revision(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def image_fingerprint(repository: Path, cohort_path: Path, architecture_contract) -> str:
    digest = hashlib.sha256(b"trtmc-devtoolkit-image-v1\0")
    inputs = [cohort_path, repository / architecture_contract.dockerfile]
    context = repository / architecture_contract.docker_context
    inputs.extend(path for path in sorted(context.rglob("*")) if path.is_file())
    for path in inputs:
        digest.update(str(path.relative_to(repository)).encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def request_fingerprint(
    request: PrepareRequest,
    *,
    cohort_id: str,
    architecture: str,
    revision: str,
) -> str:
    payload = {
        "schema_version": 1,
        "cohort": cohort_id,
        "architecture": architecture,
        "revision": revision,
        "request": asdict(request),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class Planner:
    def __init__(
        self,
        repository: Path,
        state_root: Path | None = None,
        source_revision_override: str | None = None,
    ):
        self.repository = repository.resolve()
        self.state_root = (state_root or self.repository / ".devtoolkit" / "runs").resolve()
        if source_revision_override is not None and not _SOURCE_REVISION_PATTERN.fullmatch(
            source_revision_override
        ):
            raise DevToolkitError(
                "source_revision_override must be a 40- or 64-character lowercase "
                "hexadecimal source identifier"
            )
        self.source_revision_override = source_revision_override
        self.registry = CohortRegistry(
            self.repository / "configs" / "environment-cohorts"
        )

    def create(self, request: PrepareRequest) -> PreparationPlan:
        architecture = normalize_architecture(request.architecture)
        cohort = self.registry.resolve(
            tensorrt=request.tensorrt,
            cuda=request.cuda,
            architecture=architecture,
            python_version=request.python_version,
            allow_experimental=request.allow_experimental,
        )
        contract = cohort.architectures[architecture]
        if (
            request.target.kind == "docker"
            and request.python_version != contract.container_python_version
        ):
            raise DevToolkitError(
                f"Environment cohort {cohort.id} Docker image uses Python "
                f"{contract.container_python_version}; requested Python "
                f"{request.python_version}"
            )
        revision = self.source_revision_override or source_revision(self.repository)
        fingerprint = request_fingerprint(
            request,
            cohort_id=cohort.id,
            architecture=architecture,
            revision=revision,
        )
        state_dir = self.state_root / fingerprint[:16]
        docker_fingerprint = None
        if request.target.kind == "docker" and request.target.image is None:
            docker_fingerprint = image_fingerprint(
                self.repository,
                cohort.source,
                cohort.architectures[architecture],
            )
        steps = [
            PlanStep("doctor", "Inspect the host and selected target prerequisites", False),
            PlanStep("provision", f"Prepare the {request.target.kind} target", True),
            PlanStep(
                "build-install",
                f"Build TRTMC and prepare the {request.mode} layout",
                True,
            ),
            PlanStep("verify-install", "Verify Python, CLI, native DSOs, and TensorRT ABI", False),
        ]
        if request.model is not None:
            steps.append(
                PlanStep(
                    "model-smoke",
                    f"Build, inspect, and run {request.model.model_id}",
                    True,
                )
            )
        steps.append(PlanStep("receipt", "Write a reproducible preparation receipt", True))
        return PreparationPlan(
            request=request,
            cohort=cohort,
            repository=self.repository,
            architecture=architecture,
            source_revision=revision,
            run_id=fingerprint[:16],
            state_dir=state_dir,
            image_fingerprint=docker_fingerprint,
            steps=tuple(steps),
        )
