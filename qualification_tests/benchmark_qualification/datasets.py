# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve internal benchmark datasets without embedding machine paths."""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .catalog import QualificationError
from .runtime import RuntimeContext


@dataclass(frozen=True)
class Dataset:
    id: str
    path: Path
    source_mode: str
    sha256: str

    def receipt(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_mode": self.source_mode,
            "sha256": self.sha256,
        }


def resolve_dataset(
    definition: Mapping[str, Any],
    context: RuntimeContext,
    profile: Path | None = None,
) -> Dataset:
    configured = definition.get("dataset")
    if not isinstance(configured, Mapping):
        raise QualificationError("Accuracy benchmark requires a dataset object")
    dataset_id = _string(configured.get("id"), "dataset.id")
    source = configured.get("source")
    if not isinstance(source, Mapping):
        raise QualificationError("Accuracy benchmark requires dataset.source")
    mode = _string(source.get("mode"), "dataset.source.mode")
    explicit = context.datasets.get(dataset_id)
    if explicit is not None:
        path = explicit
        resolved_mode = "provided"
    elif mode == "manual":
        relative = source.get("path")
        if not isinstance(relative, str) or not relative:
            raise QualificationError(
                f"dataset {dataset_id!r} must be supplied with --dataset {dataset_id}=PATH"
            )
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise QualificationError("manual dataset source.path must stay below --data-root")
        path = (context.data_root / candidate).resolve()
        resolved_mode = "staged"
    elif mode == "download":
        if not configured.get("sha256"):
            raise QualificationError(f"downloadable dataset {dataset_id!r} requires dataset.sha256")
        path = _download_dataset(dataset_id, source, context.data_root)
        resolved_mode = "download"
    elif mode == "family":
        relative = source.get("path")
        if profile is None:
            raise QualificationError(f"family dataset {dataset_id!r} requires its model profile")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise QualificationError("family dataset source.path must be relative")
        family_root = profile.parents[2].resolve()
        path = (profile.parent / relative).resolve()
        if not path.is_relative_to(family_root):
            raise QualificationError("family dataset source.path must stay inside its family")
        resolved_mode = "family"
    else:
        raise QualificationError(
            f"dataset {dataset_id!r} has unsupported source mode {mode!r}; "
            "use manual, download, or family"
        )
    if not path.is_file():
        if mode == "manual":
            raise QualificationError(
                f"dataset {dataset_id!r} is unavailable at {path}; "
                f"provide --dataset {dataset_id}=PATH"
            )
        raise QualificationError(f"dataset {dataset_id!r} is unavailable at {path}")
    digest = _sha256(path)
    expected = configured.get("sha256")
    if expected and digest != expected:
        if resolved_mode == "download":
            path.unlink(missing_ok=True)
        raise QualificationError(f"dataset {dataset_id!r} checksum differs: {path}")
    return Dataset(dataset_id, path.resolve(), resolved_mode, digest)


def _download_dataset(dataset_id: str, source: Mapping[str, Any], root: Path) -> Path:
    url = _string(source.get("url"), "dataset.source.url")
    filename = source.get("filename")
    if filename is None:
        filename = Path(urllib.parse.urlparse(url).path).name
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise QualificationError("dataset.source.filename must be one file name")
    destination = (root / dataset_id / filename).resolve()
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        urllib.request.urlretrieve(url, temporary)
        os.replace(temporary, destination)
    except (OSError, urllib.error.URLError) as error:
        temporary.unlink(missing_ok=True)
        raise QualificationError(f"cannot download dataset {dataset_id!r}: {error}") from error
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise QualificationError(f"{field} must be a non-empty string")
    return value
