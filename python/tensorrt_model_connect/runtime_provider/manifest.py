# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filesystem manifest contract for isolated runtime-provider capsules."""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility.
    import tomli as tomllib


MANIFEST_NAME = "IMPLEMENTATION.toml"
MANIFEST_SCHEMA_VERSION = 1
RUNTIME_ABI_VERSIONS = frozenset({1})

_IDENTIFIER_RE = re.compile(r"^[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?$")
_TARGET_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*$")

_TOP_LEVEL_KEYS = {
    "schema_version",
    "implementation_id",
    "downstream_runtime",
    "downstream_version",
    "downstream_commit",
    "model",
    "target",
    "build",
    "runtime",
}
_MODEL_KEYS = {"id", "revisions"}
_BUILD_KEYS = {"entrypoint", "timeout_seconds"}
_RUNTIME_KEYS = {"library", "abi"}

TargetScalar = str | int | float | bool
RequestValue = (
    str | int | float | bool | None | tuple["RequestValue", ...] | Mapping[str, "RequestValue"]
)


class ManifestValidationError(ValueError):
    """An ``IMPLEMENTATION.toml`` violates the capsule contract."""


class ManifestDiscoveryError(ValueError):
    """Filesystem discovery found a structurally ambiguous capsule set."""


class AmbiguousImplementationError(LookupError):
    """More than one implementation is authoritative for a request."""


def _fail(path: Path, message: str) -> ManifestValidationError:
    return ManifestValidationError(f"{path}: {message}")


def _require_table(value: Any, path: Path, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _fail(path, f"{field} must be a TOML table")
    return dict(value)


def _reject_unknown_keys(
    table: Mapping[str, Any], allowed: set[str], path: Path, field: str
) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise _fail(path, f"{field} contains unknown field(s): {', '.join(unknown)}")


def _require_string(value: Any, path: Path, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(path, f"{field} must be a non-empty string")
    if value != value.strip():
        raise _fail(path, f"{field} must not contain leading or trailing whitespace")
    return value


def _require_identifier(value: Any, path: Path, field: str) -> str:
    text = _require_string(value, path, field)
    if len(text) > 255 or _IDENTIFIER_RE.fullmatch(text) is None:
        raise _fail(
            path,
            f"{field} must use lowercase letters, digits, '.', '_' or '-'",
        )
    return text


def _require_exact_int(value: Any, path: Path, field: str) -> int:
    if type(value) is not int:
        raise _fail(path, f"{field} must be an integer")
    return value


def _require_string_list(
    value: Any,
    path: Path,
    field: str,
    *,
    item_pattern: re.Pattern[str] | None = None,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise _fail(path, f"{field} must be a non-empty array of strings")
    values: list[str] = []
    for index, item in enumerate(value):
        text = _require_string(item, path, f"{field}[{index}]")
        if item_pattern is not None and item_pattern.fullmatch(text) is None:
            raise _fail(path, f"{field}[{index}] has an invalid identifier")
        values.append(text)
    if len(values) != len(set(values)):
        raise _fail(path, f"{field} must not contain duplicate values")
    return tuple(values)


def _validate_target_scalar(value: Any, path: Path, field: str) -> TargetScalar:
    if isinstance(value, bool) or isinstance(value, str):
        if isinstance(value, str) and not value:
            raise _fail(path, f"{field} must not be an empty string")
        return value
    if type(value) is int:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise _fail(path, f"{field} must be a finite string, integer, float, or boolean")


def _validate_target(value: Any, path: Path, field: str = "target") -> Mapping[str, TargetScalar]:
    table = _require_table(value, path, field)
    if not table:
        raise _fail(path, f"{field} must contain at least one exact target fact")
    normalized: dict[str, TargetScalar] = {}
    for key in sorted(table):
        if _TARGET_KEY_RE.fullmatch(key) is None:
            raise _fail(path, f"{field} contains invalid fact name {key!r}")
        normalized[key] = _validate_target_scalar(table[key], path, f"{field}.{key}")
    return MappingProxyType(normalized)


def _validate_request_value(value: Any, path: Path, field: str) -> RequestValue:
    """Normalize JSON-compatible request data without interpreting it."""

    if value is None or isinstance(value, (bool, str)) or type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _fail(path, f"{field} must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(
            _validate_request_value(item, path, f"{field}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        normalized: dict[str, RequestValue] = {}
        for key in sorted(value):
            if not isinstance(key, str) or not key:
                raise _fail(path, f"{field} keys must be non-empty strings")
            normalized[key] = _validate_request_value(value[key], path, f"{field}.{key}")
        return MappingProxyType(normalized)
    raise _fail(path, f"{field} must be JSON-compatible")


def _request_value_to_json(value: RequestValue) -> Any:
    if isinstance(value, Mapping):
        return {key: _request_value_to_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_request_value_to_json(item) for item in value]
    return value


def _validate_relative_entrypoint(capsule_root: Path, value: Any, path: Path) -> Path:
    text = _require_string(value, path, "build.entrypoint")
    relative = Path(text)
    if relative.is_absolute() or ".." in relative.parts or relative == Path("."):
        raise _fail(path, "build.entrypoint must be a normalized capsule-relative path")
    entrypoint = capsule_root / relative
    try:
        resolved = entrypoint.resolve(strict=True)
    except FileNotFoundError as exc:
        raise _fail(path, f"build.entrypoint does not exist: {text}") from exc
    try:
        resolved.relative_to(capsule_root)
    except ValueError as exc:
        raise _fail(path, "build.entrypoint resolves outside the capsule") from exc
    if not resolved.is_file():
        raise _fail(path, "build.entrypoint must resolve to a regular file")
    if resolved.suffix != ".py" and not os.access(resolved, os.X_OK):
        raise _fail(path, "non-Python build.entrypoint must be executable")
    return resolved


def _validate_runtime_library(value: Any, path: Path) -> str:
    library = _require_string(value, path, "runtime.library")
    library_path = Path(library)
    if library_path.is_absolute() or library_path.name != library or library in {".", ".."}:
        raise _fail(path, "runtime.library must be a library filename, not a path")
    if not library.startswith("libtrtmc_impl_") or not library.endswith(".so"):
        raise _fail(
            path,
            "runtime.library must name an isolated libtrtmc_impl_*.so",
        )
    if _IDENTIFIER_RE.fullmatch(library.removeprefix("libtrtmc_impl_").removesuffix(".so")) is None:
        raise _fail(path, "runtime.library contains unsafe filename characters")
    return library


@dataclass(frozen=True)
class ImplementationRequest:
    """Canonical request used for exact implementation matching."""

    model_id: str
    model_revision: str
    target: Mapping[str, TargetScalar]
    parameters: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        pseudo_path = Path("<implementation-request>")
        model_id = _require_string(self.model_id, pseudo_path, "model_id")
        revision = _require_string(self.model_revision, pseudo_path, "model_revision")
        target = _validate_target(dict(self.target), pseudo_path)
        raw_parameters = {} if self.parameters is None else self.parameters
        if not isinstance(raw_parameters, Mapping):
            raise _fail(pseudo_path, "parameters must be a mapping")
        parameters = _validate_request_value(dict(raw_parameters), pseudo_path, "parameters")
        assert isinstance(parameters, Mapping)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "model_revision", revision)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "parameters", parameters)

    def to_json(self) -> dict[str, Any]:
        """Return the versioned JSON request passed to a build adapter."""
        return {
            "schema_version": 1,
            "model": {"id": self.model_id, "revision": self.model_revision},
            "target": dict(self.target),
            "parameters": _request_value_to_json(self.parameters or {}),
        }


@dataclass(frozen=True)
class ImplementationManifest:
    """Validated declaration for one isolated optimized implementation."""

    path: Path
    capsule_root: Path
    implementation_id: str
    downstream_runtime: str
    downstream_version: str
    downstream_commit: str
    model_id: str
    model_revisions: tuple[str, ...]
    target: Mapping[str, TargetScalar]
    build_entrypoint: Path
    build_timeout_seconds: int
    runtime_library: str
    runtime_abi: int

    def matches_target(self, target: Mapping[str, TargetScalar]) -> bool:
        for key, required in self.target.items():
            actual = target.get(key)
            if type(actual) is not type(required) or actual != required:
                return False
        return True

    def matches(self, request: ImplementationRequest) -> bool:
        return (
            self.model_id == request.model_id
            and request.model_revision in self.model_revisions
            and self.matches_target(request.target)
        )


def manifest_contract_sha256(manifest: ImplementationManifest) -> str:
    """Hash the exact selected manifest bytes carried in build bindings."""

    try:
        encoded = manifest.path.read_bytes()
    except OSError as exc:
        raise ManifestValidationError(
            f"Unable to hash selected manifest {manifest.path}: {exc}"
        ) from exc
    return hashlib.sha256(encoded).hexdigest()


def load_implementation_manifest(path: str | Path) -> ImplementationManifest:
    """Load and strictly validate one ``IMPLEMENTATION.toml`` file."""
    manifest_path = Path(path)
    if manifest_path.name != MANIFEST_NAME:
        raise _fail(manifest_path, f"manifest filename must be {MANIFEST_NAME}")
    if manifest_path.is_symlink():
        raise _fail(manifest_path, "manifest file must not be a symbolic link")
    try:
        resolved_path = manifest_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise _fail(manifest_path, "manifest file does not exist") from exc
    if not resolved_path.is_file():
        raise _fail(resolved_path, "manifest must be a regular file")
    capsule_root = resolved_path.parent

    try:
        with resolved_path.open("rb") as file:
            raw = tomllib.load(file)
    except tomllib.TOMLDecodeError as exc:
        raise _fail(resolved_path, f"invalid TOML: {exc}") from exc
    if not isinstance(raw, dict):  # Defensive; tomllib always returns a dict.
        raise _fail(resolved_path, "manifest must decode to a TOML table")
    _reject_unknown_keys(raw, _TOP_LEVEL_KEYS, resolved_path, "manifest")

    missing = sorted(_TOP_LEVEL_KEYS - set(raw))
    if missing:
        raise _fail(resolved_path, f"manifest is missing field(s): {', '.join(missing)}")

    schema_version = _require_exact_int(raw["schema_version"], resolved_path, "schema_version")
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise _fail(
            resolved_path,
            f"unsupported schema_version {schema_version}; expected {MANIFEST_SCHEMA_VERSION}",
        )
    implementation_id = _require_identifier(
        raw["implementation_id"], resolved_path, "implementation_id"
    )
    downstream_runtime = _require_identifier(
        raw["downstream_runtime"], resolved_path, "downstream_runtime"
    )
    downstream_version = _require_string(
        raw["downstream_version"], resolved_path, "downstream_version"
    )
    downstream_commit = _require_string(
        raw["downstream_commit"], resolved_path, "downstream_commit"
    )
    model = _require_table(raw["model"], resolved_path, "model")
    _reject_unknown_keys(model, _MODEL_KEYS, resolved_path, "model")
    missing_model = sorted(_MODEL_KEYS - set(model))
    if missing_model:
        raise _fail(resolved_path, f"model is missing field(s): {', '.join(missing_model)}")
    model_id = _require_string(model["id"], resolved_path, "model.id")
    revisions = _require_string_list(model["revisions"], resolved_path, "model.revisions")

    target = _validate_target(raw["target"], resolved_path)

    build = _require_table(raw["build"], resolved_path, "build")
    _reject_unknown_keys(build, _BUILD_KEYS, resolved_path, "build")
    if "entrypoint" not in build:
        raise _fail(resolved_path, "build is missing field: entrypoint")
    entrypoint = _validate_relative_entrypoint(capsule_root, build["entrypoint"], resolved_path)
    timeout = _require_exact_int(
        build.get("timeout_seconds", 3600), resolved_path, "build.timeout_seconds"
    )
    if not 1 <= timeout <= 86400:
        raise _fail(resolved_path, "build.timeout_seconds must be between 1 and 86400")

    runtime = _require_table(raw["runtime"], resolved_path, "runtime")
    _reject_unknown_keys(runtime, _RUNTIME_KEYS, resolved_path, "runtime")
    missing_runtime = sorted(_RUNTIME_KEYS - set(runtime))
    if missing_runtime:
        raise _fail(
            resolved_path,
            f"runtime is missing field(s): {', '.join(missing_runtime)}",
        )
    runtime_library = _validate_runtime_library(runtime["library"], resolved_path)
    runtime_abi = _require_exact_int(runtime["abi"], resolved_path, "runtime.abi")
    if runtime_abi not in RUNTIME_ABI_VERSIONS:
        raise _fail(
            resolved_path,
            "unsupported runtime.abi "
            f"{runtime_abi}; expected one of {sorted(RUNTIME_ABI_VERSIONS)}",
        )
    return ImplementationManifest(
        path=resolved_path,
        capsule_root=capsule_root,
        implementation_id=implementation_id,
        downstream_runtime=downstream_runtime,
        downstream_version=downstream_version,
        downstream_commit=downstream_commit,
        model_id=model_id,
        model_revisions=revisions,
        target=target,
        build_entrypoint=entrypoint,
        build_timeout_seconds=timeout,
        runtime_library=runtime_library,
        runtime_abi=runtime_abi,
    )


def _manifest_paths(root: str | Path) -> tuple[Path, ...]:
    raw_root = Path(root)
    if raw_root.is_symlink():
        raise ManifestDiscoveryError(f"Discovery root must not be a symlink: {raw_root}")
    try:
        resolved = raw_root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ManifestDiscoveryError(f"Discovery root does not exist: {raw_root}") from exc
    if not resolved.is_dir():
        raise ManifestDiscoveryError(f"Discovery root is not a directory: {resolved}")
    return tuple(sorted(resolved.glob(f"*/{MANIFEST_NAME}"), key=lambda path: str(path)))


def _reject_duplicate_implementation_ids(
    manifests: Iterable[ImplementationManifest],
) -> None:
    by_id: dict[str, list[Path]] = {}
    for manifest in manifests:
        by_id.setdefault(manifest.implementation_id, []).append(manifest.path)
    duplicates = {identifier: paths for identifier, paths in by_id.items() if len(paths) > 1}
    if duplicates:
        details = "; ".join(
            f"{identifier}: {', '.join(str(path) for path in sorted(paths))}"
            for identifier, paths in sorted(duplicates.items())
        )
        raise ManifestDiscoveryError(f"Duplicate implementation_id declarations: {details}")


def _declared_model_id(path: Path) -> str | None:
    """Read only the manifest's model index before validating a candidate.

    A model family may contain many independently owned adapters. A syntax
    error in an adapter for another model must not disable the requested model.
    The small ``[model]`` table is therefore the discovery index; only an exact
    model match is subjected to full manifest validation.
    """

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        # A capsule that cannot declare its model does not participate in
        # discovery.  In particular, it must not disable a sibling model.
        return None
    starts = [index for index, line in enumerate(lines) if line.strip() == "[model]"]
    if len(starts) != 1:
        return None
    start = starts[0]
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if lines[index].lstrip().startswith("[")
        ),
        len(lines),
    )
    matches: list[str] = []
    for line in lines[start + 1 : end]:
        match = re.fullmatch(
            r'''\s*id\s*=\s*(?:"([^"\\\r\n]+)"|'([^'\r\n]+)')\s*(?:#.*)?''',
            line,
        )
        if match is not None:
            matches.append(match.group(1) or match.group(2))
    if len(matches) != 1:
        return None
    model_id = matches[0]
    if not model_id or model_id != model_id.strip():
        return None
    return model_id


def discover_implementations_for_model(
    root: str | Path,
    model_id: str,
) -> tuple[ImplementationManifest, ...]:
    """Load exact-model implementations below already selected family roots.

    Discovery reads only each sibling's ``[model]`` index. Full parsing and
    strict validation are intentionally limited to the requested model.
    """

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    manifests = tuple(
        load_implementation_manifest(path)
        for path in _manifest_paths(root)
        if _declared_model_id(path) == model_id
    )
    _reject_duplicate_implementation_ids(manifests)
    return manifests
