# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional, source-neutral qualification evidence for resolved environments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from .models import DevToolkitError

if TYPE_CHECKING:
    from .resolution import ContextLock, ToolchainCandidate


@dataclass(frozen=True)
class QualificationRef:
    name: str
    digest: str
    status: str


@dataclass(frozen=True)
class QualificationRecord:
    reference: QualificationRef
    requirements: Mapping[str, tuple[str, ...]]
    source: str = field(compare=False)

    def __post_init__(self) -> None:
        normalized = {name: tuple(values) for name, values in sorted(self.requirements.items())}
        if any(
            not name or not values or any(not value for value in values)
            for name, values in normalized.items()
        ):
            raise DevToolkitError("Qualification requirements must be non-empty")
        object.__setattr__(self, "requirements", MappingProxyType(normalized))

    def matches(
        self,
        context: ContextLock,
        candidate: ToolchainCandidate,
    ) -> bool:
        facts = {
            "tensorrt": candidate.tensorrt,
            "cuda": candidate.cuda,
            "python": candidate.python,
            "architecture": context.architecture,
            **dict(context.qualification),
        }
        return all(facts.get(name) in accepted for name, accepted in self.requirements.items())


class QualificationSource(Protocol):
    def load(self) -> tuple[QualificationRecord, ...]: ...


class JsonQualificationSource:
    """Load generic qualification facts from caller-owned JSON directories."""

    def __init__(self, roots: tuple[Path, ...]) -> None:
        self.roots = tuple(Path(root).resolve() for root in roots)

    def load(self) -> tuple[QualificationRecord, ...]:
        paths = sorted(
            path
            for root in self.roots
            if root.is_dir()
            for path in root.glob("*.json")
            if path.name != "schema.json"
        )
        return tuple(self._load(path) for path in paths)

    @staticmethod
    def _load(path: Path) -> QualificationRecord:
        try:
            content = path.read_bytes()
            payload = json.loads(content)
            name = payload["id"]
            status = payload.get("status", "qualified")
            raw_requirements = payload["requirements"]
            if not isinstance(raw_requirements, dict):
                raise TypeError("requirements must be an object")
            requirements: dict[str, tuple[str, ...]] = {}
            for fact, raw_values in raw_requirements.items():
                if isinstance(raw_values, str):
                    values = (raw_values,)
                elif isinstance(raw_values, list):
                    values = tuple(raw_values)
                else:
                    raise TypeError(f"requirement {fact} must be text or an array")
                if not isinstance(fact, str) or any(not isinstance(value, str) for value in values):
                    raise TypeError("requirement names and values must be text")
                requirements[fact] = values
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise DevToolkitError(f"Invalid qualification record {path}: {error}") from error
        if not isinstance(name, str) or not name or not isinstance(status, str) or not status:
            raise DevToolkitError(f"Invalid qualification record values in {path}")
        return QualificationRecord(
            reference=QualificationRef(
                name=name,
                digest=hashlib.sha256(content).hexdigest(),
                status=status,
            ),
            requirements=requirements,
            source=str(path),
        )


class QualificationRegistry:
    def __init__(self, sources: tuple[QualificationSource, ...]) -> None:
        self.sources = sources

    def load_all(self) -> tuple[QualificationRecord, ...]:
        records = tuple(record for source in self.sources for record in source.load())
        names = [record.reference.name for record in records]
        if len(set(names)) != len(names):
            raise DevToolkitError("Duplicate qualification record names")
        return records
