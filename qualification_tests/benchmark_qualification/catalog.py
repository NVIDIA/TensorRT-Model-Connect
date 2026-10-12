# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover family-local benchmark cases for internal qualification."""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import yaml

SCHEMA = "trtmc.qualification/v1"


class QualificationError(RuntimeError):
    pass


class _UniqueKeyLoader(yaml.SafeLoader):
    """Load policy YAML while rejecting duplicate explicit mapping keys."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=False)
            try:
                duplicate = key in seen
                seen.add(key)
            except TypeError as error:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable mapping key",
                    key_node.start_mark,
                ) from error
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
        return super().construct_mapping(node, deep=deep)


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
    reference_paths: Mapping[str, str] = dataclass_field(default_factory=dict)
    reference_build_isolation: bool = True
    environment_hook: Path | None = None

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
        reference_environment = raw.get("reference_environment")
        requirements = _requirements(reference_environment, path)
        reference_paths = _reference_paths(reference_environment, path)
        build_isolation = _build_isolation(reference_environment, path)
        hook_path = path.parent / "prepare_environment.py"
        environment_hook = hook_path.resolve() if hook_path.is_file() else None
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
                    reference_paths=reference_paths,
                    reference_build_isolation=build_isolation,
                    environment_hook=environment_hook,
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
    path = repository / "qualification_tests/benchmark_qualification/benchmarks" / f"{case.benchmark}.yaml"
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
    selected_task = value.get("selected_task")
    if selected_task is not None:
        _string(selected_task, "candidate.selected_task", path)
    model_directory = value.get("model_directory")
    if model_directory is not None:
        configured = Path(_string(model_directory, "candidate.model_directory", path))
        if configured.is_absolute() or ".." in configured.parts:
            raise QualificationError(
                f"{path}: candidate.model_directory must stay inside the family environment"
            )
    revision = value.get("revision")
    if revision is not None and not isinstance(revision, str):
        raise QualificationError(f"{path}: candidate.revision must be a string")
    trust_remote_code = value.get("trust_remote_code")
    if trust_remote_code is not None and not isinstance(trust_remote_code, bool):
        raise QualificationError(f"{path}: candidate.trust_remote_code must be a boolean")
    if trust_remote_code and (
        not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None
    ):
        raise QualificationError(
            f"{path}: trusted remote code requires an immutable 40-character revision"
        )
    if model_directory is None and (
        not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None
    ):
        raise QualificationError(
            f"{path}: remote checkpoint requires an immutable 40-character revision"
        )
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


def _build_isolation(raw: Any, path: Path) -> bool:
    if raw is None:
        return True
    if not isinstance(raw, Mapping):
        raise QualificationError(f"{path}: reference_environment must be an object")
    value = raw.get("build_isolation", True)
    if not isinstance(value, bool):
        raise QualificationError(f"{path}: reference_environment.build_isolation must be a boolean")
    return value


def _reference_paths(raw: Any, path: Path) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise QualificationError(f"{path}: reference_environment must be an object")
    configured = raw.get("paths", {})
    if not isinstance(configured, Mapping):
        raise QualificationError(f"{path}: reference_environment.paths must be an object")
    result = {}
    for name, value in configured.items():
        if not isinstance(name, str) or not name:
            raise QualificationError(f"{path}: reference_environment.paths keys must be strings")
        relative = Path(_string(value, f"reference_environment.paths.{name}", path))
        if relative.is_absolute() or ".." in relative.parts:
            raise QualificationError(
                f"{path}: reference_environment.paths.{name} must stay inside the environment"
            )
        result[name] = relative.as_posix()
    return result


def _yaml_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as error:
        raise QualificationError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise QualificationError(f"{label} must contain an object: {path}")
    return value


def _string(value: Any, name: str, path: Path) -> str:
    if not isinstance(value, str) or not value:
        raise QualificationError(f"{path}: {name} must be a non-empty string")
    return value
