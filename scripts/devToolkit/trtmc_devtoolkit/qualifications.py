# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional cohort-backed qualification provenance for resolved environments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .models import DevToolkitError

if TYPE_CHECKING:
    from .resolution import ContextLock, EnvironmentRequest, ToolchainCandidate


@dataclass(frozen=True)
class QualificationRef:
    name: str
    digest: str
    status: str


@dataclass(frozen=True)
class QualificationRecord:
    reference: QualificationRef
    tensorrt: str
    cuda: str
    python_versions: tuple[str, ...]
    architectures: tuple[str, ...]
    targets: tuple[str, ...]
    source: Path = field(compare=False)

    def matches(
        self,
        request: EnvironmentRequest,
        context: ContextLock,
        candidate: ToolchainCandidate,
    ) -> bool:
        logical_target = request.target.options.get("qualification_target")
        if not isinstance(logical_target, str):
            logical_target = "docker" if "docker" in request.target.provider else "local"
        return (
            candidate.tensorrt == self.tensorrt
            and candidate.cuda == self.cuda
            and candidate.python in self.python_versions
            and context.architecture in self.architectures
            and logical_target in self.targets
        )


class QualificationRegistry:
    def __init__(self, roots: tuple[Path, ...]) -> None:
        self.roots = roots

    def load_all(self) -> tuple[QualificationRecord, ...]:
        paths = sorted(
            path
            for root in self.roots
            if root.is_dir()
            for path in root.glob("*.json")
            if path.name != "schema.json"
        )
        records = tuple(self._load(path) for path in paths)
        names = [record.reference.name for record in records]
        if len(set(names)) != len(names):
            raise DevToolkitError("Duplicate qualification record names")
        return records

    @staticmethod
    def _load(path: Path) -> QualificationRecord:
        try:
            content = path.read_bytes()
            payload = json.loads(content)
            name = payload["id"]
            status = payload.get("status", "qualified")
            tensorrt = payload["tensorrt"]["version"]
            cuda = payload["cuda"]["version"]
            python_versions = tuple(payload["python_versions"])
            architectures = tuple(payload["architectures"])
            targets = tuple(payload["targets"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise DevToolkitError(f"Invalid qualification record {path}: {error}") from error
        values = (name, status, tensorrt, cuda, *python_versions, *architectures, *targets)
        if not all(isinstance(value, str) and value for value in values):
            raise DevToolkitError(f"Invalid qualification record values in {path}")
        return QualificationRecord(
            reference=QualificationRef(
                name=name,
                digest=hashlib.sha256(content).hexdigest(),
                status=status,
            ),
            tensorrt=tensorrt,
            cuda=cuda,
            python_versions=python_versions,
            architectures=architectures,
            targets=targets,
            source=path,
        )
