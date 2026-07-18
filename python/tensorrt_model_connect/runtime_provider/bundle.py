# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package one provider-produced directory as a delegated Model Connect bundle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tensorrt_model_connect.bundle_writer import (
    BundleInfo,
    BundleSection,
    _bundle_section_from_file,
    write_bundle,
)

from .provider_process import BuildArtifact
from .manifest import ImplementationManifest, ImplementationRequest


OPTIMIZED_DESCRIPTOR_SECTION = "optimized_runtime.json"
IMPLEMENTATION_METADATA_SECTION = "implementation.json"
ARTIFACT_SECTION_PREFIX = "optimized_runtime_artifacts"
OPTIMIZED_DESCRIPTOR_SCHEMA_VERSION = 2
_MAX_ARTIFACT_ENTRIES = 65536
_MAX_ARTIFACT_PATH_BYTES = 4096
_MAX_ARTIFACT_TOTAL_SIZE = 1 << 40
_MAX_METADATA_BYTES = 16 * 1024 * 1024
_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


class OptimizedBundleError(ValueError):
    """Capsule output cannot be represented by the generic bundle contract."""


@dataclass(frozen=True)
class EmbeddedArtifactTree:
    sections: tuple[BundleSection, ...]
    directories: tuple[str, ...]
    file_count: int
    total_size: int
    tree_sha256: str


def _file_size_and_digest(path: Path) -> tuple[int, bytes]:
    digest = hashlib.sha256()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise OptimizedBundleError(
            f"Unable to hash capsule artifact without following links: {path}: {exc}"
        ) from exc
    with os.fdopen(descriptor, "rb") as source:
        metadata = os.fstat(source.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise OptimizedBundleError(f"Capsule artifact is not a regular file: {path}")
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return metadata.st_size, digest.digest()


def collect_artifact_tree(root: str | Path) -> EmbeddedArtifactTree:
    """Validate and describe an opaque capsule artifact directory."""

    raw_root = Path(root)
    if raw_root.is_symlink():
        raise OptimizedBundleError(f"Capsule artifact root must not be a symlink: {root}")
    try:
        artifact_root = raw_root.resolve(strict=True)
    except OSError as exc:
        raise OptimizedBundleError(f"Capsule artifact root is unavailable: {root}: {exc}") from exc
    if not artifact_root.is_dir():
        raise OptimizedBundleError(f"Capsule artifact root is not a directory: {root}")

    directories: list[str] = []
    files: list[tuple[str, Path]] = []
    for candidate in sorted(artifact_root.rglob("*"), key=lambda path: path.as_posix()):
        relative = candidate.relative_to(artifact_root).as_posix()
        if (
            not relative
            or "\\" in relative
            or len(relative.encode("utf-8")) > _MAX_ARTIFACT_PATH_BYTES
        ):
            raise OptimizedBundleError(f"Unsafe capsule artifact path: {relative!r}")
        if candidate.is_symlink():
            raise OptimizedBundleError(
                f"Capsule artifact trees must not contain symbolic links: {candidate}"
            )
        if candidate.is_dir():
            directories.append(relative)
        elif candidate.is_file():
            files.append((relative, candidate))
        else:
            raise OptimizedBundleError(
                "Capsule artifact trees may contain only regular files and directories: "
                f"{candidate}"
            )
        if len(directories) + len(files) > _MAX_ARTIFACT_ENTRIES:
            raise OptimizedBundleError(
                f"Capsule artifact tree exceeds {_MAX_ARTIFACT_ENTRIES} entries"
            )

    digest = hashlib.sha256()
    for relative in directories:
        digest.update(b"directory\0")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")

    sections: list[BundleSection] = []
    total_size = 0
    for relative, source in files:
        size, content_digest = _file_size_and_digest(source)
        digest.update(b"file\0")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(content_digest)
        sections.append(
            _bundle_section_from_file(
                f"{ARTIFACT_SECTION_PREFIX}/{relative}",
                source,
                expected_sha256=content_digest.hex(),
            )
        )
        total_size += size
        if total_size > _MAX_ARTIFACT_TOTAL_SIZE:
            raise OptimizedBundleError(
                f"Capsule artifact tree exceeds {_MAX_ARTIFACT_TOTAL_SIZE} bytes"
            )
    if not sections:
        raise OptimizedBundleError("Capsule artifact directory contains no files")
    return EmbeddedArtifactTree(
        sections=tuple(sections),
        directories=tuple(directories),
        file_count=len(sections),
        total_size=total_size,
        tree_sha256=digest.hexdigest(),
    )


def _metadata_bytes(name: str, value: Any) -> bytes:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if len(encoded) > _MAX_METADATA_BYTES:
        raise OptimizedBundleError(
            f"Optimized-runtime metadata section {name!r} exceeds {_MAX_METADATA_BYTES} bytes"
        )
    return encoded


def _required_object(root: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = root.get(field)
    if not isinstance(value, Mapping):
        raise OptimizedBundleError(f"Capsule descriptor requires object field {field!r}")
    return value


def _required_string(root: Mapping[str, Any], field: str) -> str:
    value = root.get(field)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise OptimizedBundleError(
            f"Capsule descriptor field {field!r} must be a non-empty trimmed string"
        )
    return value


def _descriptor_values(
    descriptor: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any] | None, Mapping[str, Any] | None]:
    """Read the tiny cross-runtime fields; preserve the rest as opaque JSON."""

    binding = _required_object(descriptor, "build_binding")
    profile_id = _required_string(binding, "profile_id")
    if len(profile_id) > 255 or _SAFE_IDENTIFIER_RE.fullmatch(profile_id) is None:
        raise OptimizedBundleError("Capsule descriptor profile_id must be a safe path component")
    bundle_config = descriptor.get("bundle_config")
    if bundle_config is not None and not isinstance(bundle_config, Mapping):
        raise OptimizedBundleError("Capsule descriptor bundle_config must be a JSON object")
    bundle_info = descriptor.get("bundle_info")
    if bundle_info is not None and not isinstance(bundle_info, Mapping):
        raise OptimizedBundleError("Capsule descriptor bundle_info must be a JSON object")
    return profile_id, bundle_config, bundle_info


def write_optimized_bundle(
    output_path: str | Path,
    manifest: ImplementationManifest,
    request: ImplementationRequest,
    build: BuildArtifact,
) -> Path:
    """Write the single generic delegated-bundle format from opaque capsule output."""

    artifact = collect_artifact_tree(build.artifacts_path)
    runtime_matches = [
        section
        for section in artifact.sections
        if section.name == f"{ARTIFACT_SECTION_PREFIX}/{manifest.runtime_library}"
    ]
    if len(runtime_matches) != 1:
        raise OptimizedBundleError(
            "Capsule artifact tree must contain exactly one runtime library named "
            f"{manifest.runtime_library!r}; found {len(runtime_matches)}"
        )

    profile_id, bundle_config, capsule_bundle_info = _descriptor_values(build.descriptor)

    generic_descriptor = {
        "schema_version": OPTIMIZED_DESCRIPTOR_SCHEMA_VERSION,
        "implementation_id": manifest.implementation_id,
        "model_id": request.model_id,
        "profile_id": profile_id,
        "runtime_library": manifest.runtime_library,
        "factory_abi": manifest.runtime_abi,
        "implementation_metadata_section": IMPLEMENTATION_METADATA_SECTION,
        "runtime": {
            "name": manifest.downstream_runtime,
            "version": manifest.downstream_version,
            "commit": manifest.downstream_commit,
        },
        "artifact": {
            "section_prefix": ARTIFACT_SECTION_PREFIX,
            "directories": list(artifact.directories),
            "file_count": artifact.file_count,
            "total_size": artifact.total_size,
            "tree_sha256": artifact.tree_sha256,
        },
    }
    info_values = dict(capsule_bundle_info or {})
    declared_model = info_values.pop("model_id", request.model_id)
    if declared_model != request.model_id:
        raise OptimizedBundleError("Capsule bundle_info.model_id must match the build request")
    declared_source_model = info_values.pop("source_model_id", request.model_id)
    if declared_source_model != request.model_id:
        raise OptimizedBundleError(
            "Capsule bundle_info.source_model_id must match the build request"
        )
    declared_source_revision = info_values.pop(
        "source_revision", request.model_revision
    )
    if declared_source_revision != request.model_revision:
        raise OptimizedBundleError(
            "Capsule bundle_info.source_revision must match the build request"
        )
    # These neutral values avoid importing any modality, precision, or engine
    # assumptions into the shared packager. Capsules may explicitly override
    # them through bundle_info for display and inspection purposes.
    info_values.setdefault("model_type", "optimized_runtime")
    info_values.setdefault("family", "optimized_runtime")
    info_values.setdefault("precision", "")
    info_values.setdefault("max_cache_length", 0)
    try:
        info = BundleInfo(
            model_id=request.model_id,
            source_model_id=request.model_id,
            source_revision=request.model_revision,
            **info_values,
        )
    except TypeError as exc:
        raise OptimizedBundleError(f"Capsule bundle_info is invalid: {exc}") from exc

    sections = [
        BundleSection(
            OPTIMIZED_DESCRIPTOR_SECTION,
            _metadata_bytes(OPTIMIZED_DESCRIPTOR_SECTION, generic_descriptor),
        ),
        BundleSection(
            IMPLEMENTATION_METADATA_SECTION,
            _metadata_bytes(IMPLEMENTATION_METADATA_SECTION, dict(build.descriptor)),
        ),
        *artifact.sections,
    ]
    if bundle_config is not None:
        sections.insert(
            0,
            BundleSection(
                "config.json",
                _metadata_bytes("config.json", dict(bundle_config)),
            ),
        )
    write_bundle(output_path, info, sections)
    return Path(output_path)
