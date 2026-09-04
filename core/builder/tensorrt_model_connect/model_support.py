# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve one model to one family through family-owned support declarations."""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


_ID = re.compile(r"[a-z][a-z0-9_]*\Z")


@dataclass(frozen=True)
class ModelMetadata:
    """Raw Hugging Face metadata used only for family discovery."""

    config: dict[str, Any]
    model_index: dict[str, Any]
    files: tuple[str, ...] = ()

    @property
    def model_type(self) -> str:
        value = self.config.get("model_type", "")
        return value if isinstance(value, str) else ""

    @property
    def architectures(self) -> tuple[str, ...]:
        value = self.config.get("architectures", ())
        if isinstance(value, str):
            result = (value,)
        elif isinstance(value, list):
            result = tuple(item for item in value if isinstance(item, str))
        else:
            result = ()
        singular = self.config.get("architecture")
        if isinstance(singular, str) and singular not in result:
            return (*result, singular)
        return result

    @property
    def pipeline_class(self) -> str:
        value = self.model_index.get("_class_name", "")
        return value if isinstance(value, str) else ""


@dataclass(frozen=True)
class FamilySupport:
    """Task capabilities returned by one matching family."""

    tasks: tuple[str, ...]
    default_task: str

    def __post_init__(self) -> None:
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("family support tasks must be non-empty and unique")
        if any(_ID.fullmatch(task) is None for task in self.tasks):
            raise ValueError("family support tasks must be lowercase identifiers")
        if self.default_task not in self.tasks:
            raise ValueError("default_task must be one of the supported tasks")


DescribeSupport = Callable[[ModelMetadata], FamilySupport | None]


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def family_support(
    *,
    model_types: tuple[str, ...] = (),
    architectures: tuple[str, ...] = (),
    pipeline_classes: tuple[str, ...] = (),
    required_files: tuple[str, ...] = (),
    tasks: tuple[str, ...],
    default_task: str,
) -> DescribeSupport:
    """Create one exact, family-owned support function."""

    model_type_keys = frozenset(key for value in model_types if (key := _key(value)))
    architecture_keys = frozenset(
        key for value in architectures if (key := _key(value))
    )
    pipeline_keys = frozenset(
        key for value in pipeline_classes if (key := _key(value))
    )
    file_keys = frozenset(value for value in required_files if value)
    if not model_type_keys and not architecture_keys and not pipeline_keys and not file_keys:
        raise ValueError("family support must declare at least one model identity")
    support = FamilySupport(tasks=tasks, default_task=default_task)

    def describe(metadata: ModelMetadata) -> FamilySupport | None:
        if _key(metadata.model_type) in model_type_keys:
            return support
        if architecture_keys.intersection(_key(value) for value in metadata.architectures):
            return support
        if _key(metadata.pipeline_class) in pipeline_keys:
            return support
        if file_keys and file_keys.issubset(metadata.files):
            return support
        return None

    return describe


def _read_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read model metadata {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"model metadata must be a JSON object: {path}")
    return value


def load_model_metadata(model_dir: str | Path) -> ModelMetadata:
    """Read the standard Hugging Face identity files from a local snapshot."""

    root = Path(model_dir)
    config = _read_object(root / "config.json")
    model_index = _read_object(root / "model_index.json")
    files = tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        )
    )
    if not files:
        raise ValueError(f"model snapshot is empty: {root}")
    return ModelMetadata(config=config, model_index=model_index, files=files)


def _family_directories() -> list[Path]:
    package = importlib.import_module("families")
    root = Path(next(iter(package.__path__)))
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith("_") and _ID.fullmatch(path.name)
    )


def resolve_family(metadata: ModelMetadata) -> tuple[str, FamilySupport]:
    """Ask every lightweight family support module and require one owner."""

    matches: list[tuple[str, FamilySupport]] = []
    for family in _family_directories():
        module = importlib.import_module(f"families.{family.name}.support")
        describe = getattr(module, "describe", None)
        if not callable(describe):
            raise RuntimeError(f"family {family.name!r} does not define support.describe()")
        support = describe(metadata)
        if support is not None:
            if not isinstance(support, FamilySupport):
                raise TypeError(
                    f"family {family.name!r} support.describe() returned an invalid value"
                )
            matches.append((family.name, support))

    if not matches:
        identity = metadata.model_type or metadata.pipeline_class or "unknown"
        raise ValueError(f"no family supports model {identity!r}")
    if len(matches) > 1:
        names = ", ".join(family for family, _ in matches)
        raise ValueError(f"multiple families support this model: {names}")
    return matches[0]
