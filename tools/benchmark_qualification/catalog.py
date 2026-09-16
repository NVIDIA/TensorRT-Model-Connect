# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover family-local benchmark cases for internal qualification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

SCHEMA = "trtmc.qualification/v1"


class QualificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class QualificationCase:
    kind: str
    model: str
    family: str
    name: str
    benchmark: str
    candidate: Mapping[str, Any]
    values: Mapping[str, Any]
    source: Path
    reference_requirements: Path | None

    @property
    def id(self) -> str:
        return f"{self.model}/{self.kind}/{self.name}"


def discover(repository: Path) -> tuple[QualificationCase, ...]:
    repository = repository.resolve()
    families = repository / "families"
    cases: list[QualificationCase] = []
    seen: set[str] = set()
    pattern = "*/tests/benchmark/*.yaml"
    for path in sorted(families.glob(pattern)):
        raw = _yaml_object(path, "qualification config")
        if raw.get("schema_version") != SCHEMA:
            raise QualificationError(f"{path}: schema_version must be {SCHEMA}")
        model = _string(raw.get("model"), "model", path)
        family = path.parents[2].name
        if path.stem != model:
            raise QualificationError(f"{path}: file name must match model {model!r}")
        candidate = _candidate(raw.get("candidate"), family, path)
        requirements = _requirements(raw.get("reference_environment"), path)
        for kind in ("accuracy", "performance"):
            configured = raw.get(kind, [])
            if not isinstance(configured, list):
                raise QualificationError(f"{path}: {kind} must be a list")
            for value in configured:
                if not isinstance(value, Mapping):
                    raise QualificationError(f"{path}: every {kind} case must be an object")
                name = _string(value.get("name"), f"{kind}.name", path)
                benchmark = _string(value.get("benchmark"), f"{kind}.benchmark", path)
                case = QualificationCase(
                    kind=kind,
                    model=model,
                    family=family,
                    name=name,
                    benchmark=benchmark,
                    candidate=candidate,
                    values=dict(value),
                    source=path.resolve(),
                    reference_requirements=requirements,
                )
                if case.id in seen:
                    raise QualificationError(f"duplicate qualification case {case.id!r}")
                seen.add(case.id)
                cases.append(case)
    return tuple(cases)


def select(
    cases: Sequence[QualificationCase], requested: Sequence[str]
) -> tuple[QualificationCase, ...]:
    names = {item.strip() for raw in requested for item in str(raw).split(",") if item.strip()}
    if not names:
        return tuple(cases)
    selected = tuple(
        case
        for case in cases
        if names & {case.model, case.family, case.id, f"{case.model}/{case.name}"}
    )
    matched = {
        requested_name
        for requested_name in names
        if any(
            requested_name in {case.model, case.family, case.id, f"{case.model}/{case.name}"}
            for case in selected
        )
    }
    if missing := sorted(names - matched):
        raise QualificationError("unknown qualification selection: " + ", ".join(missing))
    return selected


def load_benchmark(repository: Path, case: QualificationCase) -> dict[str, Any]:
    path = repository / "tools/benchmark_qualification/benchmarks" / f"{case.benchmark}.yaml"
    value = _yaml_object(path, "benchmark definition")
    if value.get("kind") != case.kind:
        raise QualificationError(
            f"{path}: benchmark kind {value.get('kind')!r} does not match {case.kind!r}"
        )
    return value


def _candidate(raw: Any, family: str, path: Path) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise QualificationError(f"{path}: candidate must be an object")
    value = dict(raw)
    configured_family = _string(value.get("family"), "candidate.family", path)
    if configured_family != family:
        raise QualificationError(
            f"{path}: candidate family {configured_family!r} does not match {family!r}"
        )
    for field in ("checkpoint", "task", "precision"):
        _string(value.get(field), f"candidate.{field}", path)
    revision = value.get("revision")
    if revision is not None and not isinstance(revision, str):
        raise QualificationError(f"{path}: candidate.revision must be a string")
    build = value.get("build", {})
    if not isinstance(build, Mapping):
        raise QualificationError(f"{path}: candidate.build must be an object")
    value["build"] = dict(build)
    return value


def _requirements(raw: Any, path: Path) -> Path | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise QualificationError(f"{path}: reference_environment must be an object")
    value = raw.get("requirements")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise QualificationError(f"{path}: reference_environment.requirements must be a path")
    result = (path.parent / value).resolve()
    if not result.is_file():
        raise QualificationError(f"{path}: reference requirements do not exist: {result}")
    return result


def _yaml_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise QualificationError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise QualificationError(f"{label} must contain an object: {path}")
    return value


def _string(value: Any, name: str, path: Path) -> str:
    if not isinstance(value, str) or not value:
        raise QualificationError(f"{path}: {name} must be a non-empty string")
    return value
