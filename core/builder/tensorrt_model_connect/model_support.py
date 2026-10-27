# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve one model to one family through family-owned support declarations."""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping


_ID = re.compile(r"[a-z][a-z0-9_]*\Z")


@dataclass(frozen=True)
class ModelMetadata:
    """Raw Hugging Face metadata used only for family discovery.

    A document is ``None`` when it was not read. Support rules refuse to decide
    from a document that the file list names but that was not read.
    """

    config: dict[str, Any] | None
    model_index: dict[str, Any] | None
    files: tuple[str, ...] = ()
    adapter_config: dict[str, Any] | None = field(default_factory=dict)
    repo_id: str = ""
    revision: str = ""

    @property
    def model_type(self) -> str:
        value = (self.config or {}).get("model_type", "")
        return value if isinstance(value, str) else ""

    @property
    def architectures(self) -> tuple[str, ...]:
        config = self.config or {}
        value = config.get("architectures", ())
        if isinstance(value, str):
            result = (value,)
        elif isinstance(value, list):
            result = tuple(item for item in value if isinstance(item, str))
        else:
            result = ()
        singular = config.get("architecture")
        if isinstance(singular, str) and singular not in result:
            return (*result, singular)
        return result

    @property
    def pipeline_class(self) -> str:
        value = (self.model_index or {}).get("_class_name", "")
        return value if isinstance(value, str) else ""


@dataclass(frozen=True)
class FamilySupport:
    """Task capabilities returned by one matching family."""

    tasks: tuple[str, ...]
    default_task: str
    default_precision: str = "fp32"

    def __post_init__(self) -> None:
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("family support tasks must be non-empty and unique")
        if any(_ID.fullmatch(task) is None for task in self.tasks):
            raise ValueError("family support tasks must be lowercase identifiers")
        if self.default_task not in self.tasks:
            raise ValueError("default_task must be one of the supported tasks")
        if self.default_precision not in {"fp16", "bf16", "fp32"}:
            raise ValueError("default_precision must be fp16, bf16, or fp32")


DescribeSupport = Callable[[ModelMetadata], FamilySupport | None]


class FamilyResolutionError(ValueError):
    """A model family cannot be selected safely."""


class IncompleteMetadataError(FamilyResolutionError):
    """A support rule needs a model document that was not read."""


class AmbiguousFamilyError(FamilyResolutionError):
    """More than one family declares support for the same model metadata."""

    def __init__(self, matches: tuple[tuple[str, FamilySupport], ...]) -> None:
        self.matches = matches
        names = ", ".join(family for family, _ in matches)
        super().__init__(f"multiple families support this model: {names}")


SUPPORT_SCHEMA_VERSION = 1
_DOCUMENT_FILES = {
    "config": "config.json",
    "model_index": "model_index.json",
    "adapter_config": "adapter_config.json",
}
_SCALARS = ("repo_id", "revision")
_TYPES: dict[str, type | tuple[type, ...]] = {
    "object": dict,
    "array": list,
    "string": str,
    "number": (int, float),
    "boolean": bool,
}
_TESTS = frozenset({"in", "contains_any", "equals", "exists", "type", "nonempty", "prefix"})
_MISSING = object()


def _pointer_segments(pointer: str) -> list[str]:
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer.split("/")[1:]]


def _validate_test(where: str, test: Any) -> None:
    if not isinstance(test, dict) or not test:
        raise ValueError(f"{where}: a condition must be a non-empty object of tests")
    for name, expected in test.items():
        if name not in _TESTS:
            raise ValueError(f"{where}: unknown test {name!r}")
        if name in {"in", "contains_any"} and not (
            isinstance(expected, list) and expected and all(isinstance(v, str) for v in expected)
        ):
            raise ValueError(f"{where}: {name!r} needs a non-empty list of strings")
        if name in {"exists", "nonempty"} and not isinstance(expected, bool):
            raise ValueError(f"{where}: {name!r} needs true or false")
        if name == "prefix" and not (isinstance(expected, str) and expected):
            raise ValueError(f"{where}: 'prefix' needs a non-empty string")
        if name == "type" and expected not in _TYPES:
            raise ValueError(f"{where}: 'type' must be one of {sorted(_TYPES)}")


def _validate_rules(document: Any, source: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(document, dict) or set(document) != {"schema_version", "rules"}:
        raise ValueError(f"{source}: must contain exactly schema_version and rules")
    if document["schema_version"] != SUPPORT_SCHEMA_VERSION:
        raise ValueError(f"{source}: unsupported schema_version {document['schema_version']!r}")
    rules = document["rules"]
    if not isinstance(rules, list) or not rules:
        raise ValueError(f"{source}: rules must be a non-empty list")
    for index, rule in enumerate(rules):
        where = f"{source} rule {index}"
        if not isinstance(rule, dict) or set(rule) != {"if", "then"}:
            raise ValueError(f"{where}: must contain exactly 'if' and 'then'")
        conditions = rule["if"]
        if not isinstance(conditions, dict) or not conditions:
            raise ValueError(f"{where}: 'if' needs at least one condition")
        for key, test in conditions.items():
            if key in {"files_all", "files_none"}:
                if not (isinstance(test, list) and test and all(isinstance(v, str) and v for v in test)):
                    raise ValueError(f"{where}: {key!r} needs a non-empty list of file paths")
            elif key.startswith("/"):
                root = _pointer_segments(key)[0]
                if root not in _DOCUMENT_FILES and root not in _SCALARS:
                    raise ValueError(f"{where}: unknown document {root!r} in {key!r}")
                _validate_test(f"{where} {key}", test)
            else:
                raise ValueError(f"{where}: unknown condition {key!r}")
        then = rule["then"]
        if then != "reject":
            if not isinstance(then, dict) or not {"tasks", "default_task"} <= set(then) or not set(
                then
            ) <= {"tasks", "default_task", "default_precision"}:
                raise ValueError(f"{where}: 'then' must be \"reject\" or tasks/default_task")
            FamilySupport(
                tuple(then["tasks"]), then["default_task"], then.get("default_precision", "fp32")
            )
    return tuple(rules)


def _resolve(metadata: ModelMetadata, pointer: str) -> Any:
    root, *path = _pointer_segments(pointer)
    if root in _SCALARS:
        value: Any = getattr(metadata, root) or _MISSING
        return value if not path else _MISSING
    value = getattr(metadata, root)
    if value is None:
        if _DOCUMENT_FILES[root] in metadata.files:
            raise IncompleteMetadataError(
                f"{_DOCUMENT_FILES[root]} is listed in the repository but was not read"
            )
        value = {}
    for key in path:
        if isinstance(value, dict) and key in value:
            value = value[key]
        elif isinstance(value, list) and key.isdigit() and int(key) < len(value):
            value = value[int(key)]
        else:
            return _MISSING
    return value


def _json_equal(left: Any, right: Any) -> bool:
    """JSON equality: numbers by value, booleans never equal numbers."""

    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_json_equal(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_json_equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def _passes(value: Any, test: dict[str, Any]) -> bool:
    for name, expected in test.items():
        if name == "exists":
            ok = (value is not _MISSING) == expected
        elif name == "equals":
            ok = value is not _MISSING and _json_equal(value, expected)
        elif name == "in":
            ok = isinstance(value, str) and value in expected
        elif name == "contains_any":
            ok = isinstance(value, list) and any(item in expected for item in value if isinstance(item, str))
        elif name == "prefix":
            ok = isinstance(value, str) and value.startswith(expected)
        elif name == "type":
            ok = isinstance(value, _TYPES[expected]) and not (
                expected == "number" and isinstance(value, bool)
            )
        else:
            empty = value is _MISSING or value in (None, "", [], {})
            ok = empty != expected
        if not ok:
            return False
    return True


def _rule_holds(conditions: dict[str, Any], metadata: ModelMetadata) -> bool:
    files = set(metadata.files)
    for key, test in conditions.items():
        if key == "files_all":
            if not set(test) <= files:
                return False
        elif key == "files_none":
            if set(test) & files:
                return False
        elif not _passes(_resolve(metadata, key), test):
            return False
    return True


@dataclass(frozen=True)
class _RuleSupport:
    """Family support evaluated from a declarative ``support.json``."""

    path: Path
    rules: tuple[dict[str, Any], ...]

    def __call__(self, metadata: ModelMetadata) -> FamilySupport | None:
        for rule in self.rules:
            if _rule_holds(rule["if"], metadata):
                then = rule["then"]
                if then == "reject":
                    return None
                return FamilySupport(
                    tuple(then["tasks"]),
                    then["default_task"],
                    then.get("default_precision", "fp32"),
                )
        return None


def support_from_json(support_module: str | Path) -> _RuleSupport:
    """Load the ``support.json`` that sits next to a family's ``support.py``."""

    path = Path(support_module).with_suffix(".json")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read support rules {path}: {error}") from error
    return _RuleSupport(path, _validate_rules(document, str(path)))


@dataclass(frozen=True)
class SupportDecision:
    """The shared selection result: one status plus the selected family."""

    status: str
    family: str | None = None
    support: FamilySupport | None = None
    matches: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()
    candidates: tuple[FamilySupport, ...] = field(default=(), repr=False)


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
    default_precision: str = "fp32",
) -> DescribeSupport:
    """Create one exact, family-owned support function."""

    model_type_keys = frozenset(key for value in model_types if (key := _key(value)))
    architecture_keys = frozenset(key for value in architectures if (key := _key(value)))
    pipeline_keys = frozenset(key for value in pipeline_classes if (key := _key(value)))
    file_keys = frozenset(value for value in required_files if value)
    if not model_type_keys and not architecture_keys and not pipeline_keys and not file_keys:
        raise ValueError("family support must declare at least one model identity")
    support = FamilySupport(
        tasks=tasks,
        default_task=default_task,
        default_precision=default_precision,
    )

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


def load_model_metadata(
    model_dir: str | Path, repo_id: str = "", revision: str = ""
) -> ModelMetadata:
    """Read the standard Hugging Face identity files from a local snapshot."""

    root = Path(model_dir)
    files = tuple(
        sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file())
    )
    if not files:
        raise ValueError(f"model snapshot is empty: {root}")
    return ModelMetadata(
        config=_read_object(root / "config.json"),
        model_index=_read_object(root / "model_index.json"),
        files=files,
        adapter_config=_read_object(root / "adapter_config.json"),
        repo_id=repo_id,
        revision=revision,
    )


def resolve_model(model: str, revision: str | None = None) -> Path:
    """Resolve a local directory or download one explicit checkpoint revision."""
    local = Path(model)
    if local.is_dir():
        return local
    if local.exists():
        raise ValueError(f"model path is not a directory: {local}")
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model,
            revision=revision,
            ignore_patterns=["*flax_model*", "*tf_model*"],
        )
    )


def _family_directories() -> list[Path]:
    package = importlib.import_module("families")
    root = Path(next(iter(package.__path__)))
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith("_") and _ID.fullmatch(path.name)
    )


def _family_describer(family: Path) -> DescribeSupport:
    module = importlib.import_module(f"families.{family.name}.support")
    describe = getattr(module, "describe", None)
    if not callable(describe):
        raise RuntimeError(f"family {family.name!r} does not define support.describe()")
    return describe


def _describe_family(family: Path, metadata: ModelMetadata) -> FamilySupport | None:
    support = _family_describer(family)(metadata)
    if support is not None and not isinstance(support, FamilySupport):
        raise TypeError(f"family {family.name!r} support.describe() returned an invalid value")
    return support


def resolve_family(
    metadata: ModelMetadata, requested_family: str | None = None
) -> tuple[str, FamilySupport]:
    """Select one compatible family automatically or by an explicit family ID."""

    directories = _family_directories()
    if requested_family is not None:
        if _ID.fullmatch(requested_family) is None:
            raise FamilyResolutionError(
                "family must be a lowercase identifier containing only letters, digits, "
                "and underscores"
            )
        selected = next(
            (family for family in directories if family.name == requested_family), None
        )
        if selected is None:
            raise FamilyResolutionError(
                f"unknown family {requested_family!r}; omit --family to discover compatible families"
            )
        support = _describe_family(selected, metadata)
        if support is None:
            raise FamilyResolutionError(
                f"family {requested_family!r} does not support this model; "
                "omit --family to discover compatible families"
            )
        return requested_family, support

    decision = decide_support(metadata)
    if decision.status == "incomplete":
        raise IncompleteMetadataError(
            "cannot decide support until every listed model document is read; "
            f"families waiting on metadata: {', '.join(decision.missing)}"
        )
    if decision.status == "conflict":
        raise AmbiguousFamilyError(tuple(zip(decision.matches, decision.candidates)))
    if decision.status == "unsupported":
        identity = metadata.model_type or metadata.pipeline_class or "unknown"
        raise FamilyResolutionError(f"no family supports model {identity!r}")
    return decision.family, decision.support


def decide_support(
    metadata: ModelMetadata, families: Mapping[str, DescribeSupport] | None = None
) -> SupportDecision:
    """Apply every family's rules and the shared selection policy.

    ``families`` maps family IDs to support functions; it defaults to every
    family package in the repository.
    """

    describers = (
        families
        if families is not None
        else {path.name: _family_describer(path) for path in _family_directories()}
    )
    matches: list[tuple[str, FamilySupport]] = []
    missing: list[str] = []
    for family, describe in describers.items():
        try:
            support = describe(metadata)
        except IncompleteMetadataError:
            missing.append(family)
            continue
        if support is not None and not isinstance(support, FamilySupport):
            raise TypeError(f"family {family!r} support.describe() returned an invalid value")
        if support is not None:
            matches.append((family, support))
    if missing:
        return SupportDecision("incomplete", missing=tuple(missing))
    if len(matches) > 1:
        return SupportDecision(
            "conflict",
            matches=tuple(family for family, _ in matches),
            candidates=tuple(support for _, support in matches),
        )
    if not matches:
        return SupportDecision("unsupported")
    family, support = matches[0]
    return SupportDecision("supported", family, support, (family,))
