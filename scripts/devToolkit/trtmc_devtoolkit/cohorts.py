# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load and resolve exact TensorRT/CUDA environment cohorts."""

from __future__ import annotations

import json
import platform
import re
from pathlib import Path

from .models import ArchitectureContract, DevToolkitError, EnvironmentCohort


EXACT_TENSORRT_VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+")
EXACT_CUDA_VERSION = re.compile(r"[0-9]+\.[0-9]+")
ARCHITECTURE_ALIASES = {
    "amd64": "x86_64",
    "arm64": "aarch64",
    "x86_64": "x86_64",
    "aarch64": "aarch64",
}


def normalize_architecture(value: str | None = None) -> str:
    raw = (value or platform.machine()).strip().lower()
    try:
        return ARCHITECTURE_ALIASES[raw]
    except KeyError as error:
        raise DevToolkitError(f"Unsupported host architecture: {raw or '<empty>'}") from error


def load_cohort(path: Path) -> EnvironmentCohort:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DevToolkitError(f"Could not load environment cohort {path}: {error}") from error
    if payload.get("schema_version") != 1:
        raise DevToolkitError(f"{path}: schema_version must be 1")
    cohort_id = payload.get("id")
    status = payload.get("status")
    if not isinstance(cohort_id, str) or not cohort_id:
        raise DevToolkitError(f"{path}: id must be a non-empty string")
    if status not in {"supported", "experimental"}:
        raise DevToolkitError(f"{path}: status must be supported or experimental")
    tensorrt = payload.get("tensorrt", {})
    cuda = payload.get("cuda", {})
    trt_version = tensorrt.get("version")
    apt_version = tensorrt.get("apt_version")
    cuda_version = cuda.get("version")
    if not isinstance(trt_version, str) or not EXACT_TENSORRT_VERSION.fullmatch(trt_version):
        raise DevToolkitError(f"{path}: TensorRT version must have four numeric parts")
    if not isinstance(apt_version, str) or not apt_version:
        raise DevToolkitError(f"{path}: TensorRT apt version must be non-empty")
    if not isinstance(cuda_version, str) or not EXACT_CUDA_VERSION.fullmatch(cuda_version):
        raise DevToolkitError(f"{path}: CUDA version must have major.minor form")
    raw_python = payload.get("python_versions")
    if not isinstance(raw_python, list) or not raw_python or not all(
        isinstance(item, str) and re.fullmatch(r"[0-9]+\.[0-9]+", item)
        for item in raw_python
    ):
        raise DevToolkitError(f"{path}: python_versions must contain major.minor strings")
    raw_architectures = payload.get("architectures")
    if not isinstance(raw_architectures, dict) or not raw_architectures:
        raise DevToolkitError(f"{path}: architectures must be a non-empty object")
    architectures: dict[str, ArchitectureContract] = {}
    fields = (
        "dockerfile",
        "docker_context",
        "container_python_version",
        "wheel_platform",
        "tensorrt_include_dir",
        "tensorrt_library_dir",
    )
    for raw_name, raw_contract in raw_architectures.items():
        name = normalize_architecture(raw_name)
        if not isinstance(raw_contract, dict) or any(
            not isinstance(raw_contract.get(field), str) or not raw_contract[field]
            for field in fields
        ):
            raise DevToolkitError(f"{path}: architecture {name} has incomplete fields")
        architectures[name] = ArchitectureContract(
            **{field: raw_contract[field] for field in fields}
        )
    return EnvironmentCohort(
        schema_version=1,
        id=cohort_id,
        status=status,
        tensorrt_version=trt_version,
        tensorrt_apt_version=apt_version,
        cuda_version=cuda_version,
        python_versions=tuple(raw_python),
        architectures=architectures,
        source=path.resolve(),
    )


class CohortRegistry:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def load_all(self) -> tuple[EnvironmentCohort, ...]:
        paths = sorted(path for path in self.root.glob("*.json") if path.name != "schema.json")
        if not paths:
            raise DevToolkitError(f"No environment cohorts found under {self.root}")
        cohorts = tuple(load_cohort(path) for path in paths)
        ids = [cohort.id for cohort in cohorts]
        if len(ids) != len(set(ids)):
            raise DevToolkitError(f"Duplicate environment cohort ids under {self.root}")
        return cohorts

    def resolve(
        self,
        *,
        tensorrt: str,
        cuda: str,
        architecture: str,
        python_version: str,
        allow_experimental: bool,
    ) -> EnvironmentCohort:
        matches = [
            cohort
            for cohort in self.load_all()
            if cohort.tensorrt_version == tensorrt and cohort.cuda_version == cuda
        ]
        if len(matches) != 1:
            supported = ", ".join(
                f"TRT {cohort.tensorrt_version} / CUDA {cohort.cuda_version} ({cohort.status})"
                for cohort in self.load_all()
            )
            raise DevToolkitError(
                f"No exact environment cohort for TensorRT {tensorrt} and CUDA {cuda}. "
                f"Available cohorts: {supported}"
            )
        cohort = matches[0]
        if cohort.status == "experimental" and not allow_experimental:
            raise DevToolkitError(
                f"Environment cohort {cohort.id} is experimental; opt in explicitly"
            )
        if architecture not in cohort.architectures:
            raise DevToolkitError(
                f"Environment cohort {cohort.id} does not support architecture {architecture}"
            )
        if python_version not in cohort.python_versions:
            raise DevToolkitError(
                f"Environment cohort {cohort.id} does not support Python {python_version}; "
                f"supported versions: {', '.join(cohort.python_versions)}"
            )
        return cohort
