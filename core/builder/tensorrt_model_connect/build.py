# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dispatch a build to exactly one model family."""

from __future__ import annotations

import hashlib
import importlib
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from .bundle_writer import BundleWriter
from .graph_transform import GraphTransform, graph_transform


_ID = re.compile(r"[a-z][a-z0-9_]*\Z")
_EXACT_REVISION = re.compile(r"[0-9a-f]{40}\Z")


@dataclass(frozen=True)
class BuildRequest:
    """Inputs shared by the build core and one family-owned builder."""

    model_dir: Path
    output_path: Path
    family: str
    task: str
    precision: str
    backend: str = "trt"
    checkpoint_id: str = ""
    checkpoint_revision: str = ""
    source_revision: str = ""
    max_sequence_length: int | None = None
    image_height: int | None = None
    image_width: int | None = None
    video_num_frames: int | None = None
    max_batch_size: int = 1
    tensor_parallel_size: int = 1
    context_parallel_size: int = 1
    quantization: str | None = None
    fp32_layers: tuple[int, ...] = ()
    dynamic_kv_cache: bool = False
    verbose: bool = False
    graph_transform: GraphTransform | None = None

    def __post_init__(self) -> None:
        if not self.precision:
            raise ValueError("precision must be non-empty")
        for field in ("checkpoint_revision", "source_revision"):
            revision = getattr(self, field)
            if revision and _EXACT_REVISION.fullmatch(revision) is None:
                raise ValueError(f"{field} must be an exact 40-character Git SHA")
        _validate_id("family", self.family)
        _validate_id("task", self.task)
        if self.backend not in {"trt", "trt_rtx"}:
            raise ValueError("backend must be 'trt' or 'trt_rtx'")
        if self.max_sequence_length is not None and self.max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        for field in ("image_height", "image_width", "video_num_frames"):
            value = getattr(self, field)
            if value is not None and value < 1:
                raise ValueError(f"{field} must be positive")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be positive")
        if self.context_parallel_size < 1:
            raise ValueError("context_parallel_size must be positive")
        if self.quantization is not None and not self.quantization:
            raise ValueError("quantization must be non-empty when provided")
        if any(layer < 0 for layer in self.fp32_layers):
            raise ValueError("fp32_layers must contain non-negative indices")
        if not isinstance(self.dynamic_kv_cache, bool):
            raise ValueError("dynamic_kv_cache must be a bool")
        if self.graph_transform is not None and not callable(self.graph_transform):
            raise ValueError("graph_transform must be callable when provided")


def _validate_id(field: str, value: object) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError(
            f"{field} must be a lowercase identifier containing only "
            "letters, digits, and underscores"
        )
    return value


def content_cache_key(domain: str, *payloads: bytes) -> str:
    """Return a domain-separated key for model-agnostic content caches."""

    if not domain or any(not isinstance(payload, bytes) for payload in payloads):
        raise ValueError("cache-key domain and byte payloads must be valid")
    value = hashlib.sha256(domain.encode("utf-8") + b"\0")
    for payload in payloads:
        value.update(len(payload).to_bytes(8, "little"))
        value.update(payload)
    return value.hexdigest()


def _resolve_family(request: BuildRequest) -> str:
    """Return the explicit family dispatch key."""

    return _validate_id("family", request.family)


def _load_family(family: str) -> ModuleType:
    """Import only ``families.<family>.model`` for an exact family ID."""

    family = _validate_id("family", family)
    family_package = f"families.{family}"
    module_name = f"{family_package}.model"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name in {family_package, module_name}:
            raise ModuleNotFoundError(
                f"family {family!r} does not provide {module_name}"
            ) from error
        raise


def _select_backend(backend: str) -> None:
    """Bind the explicit build backend before importing a family builder."""

    loaded = sys.modules.get("tensorrt")
    if backend == "trt":
        if sys.modules.get("tensorrt_rtx") is not None:
            raise RuntimeError("TensorRT-RTX is already loaded in this process")
        return

    rtx = importlib.import_module("tensorrt_rtx")
    if loaded is not None and loaded is not rtx:
        raise RuntimeError("TensorRT is already loaded in this process")
    sys.modules["tensorrt"] = rtx


def build(request: BuildRequest) -> None:
    """Run one family builder and publish its bundle on success."""

    family = _resolve_family(request)
    _select_backend(request.backend)
    family_module = _load_family(family)
    writer = BundleWriter(request.output_path)
    try:
        with graph_transform(request.graph_transform):
            family_module.build(request, writer)
        writer.add_json("provenance.json", _build_provenance(request))
        writer.finish()
    except BaseException:
        writer.abort()
        raise


def resolve_source_revision(explicit: str = "") -> str:
    """Return the exact source revision that produced a bundle."""

    candidates = (
        ("source_revision", explicit),
        ("TRTMC_ENGINE_BUILD_REVISION", os.environ.get("TRTMC_ENGINE_BUILD_REVISION", "")),
        ("GITHUB_SHA", os.environ.get("GITHUB_SHA", "")),
    )
    for field, candidate in candidates:
        revision = candidate.strip().lower()
        if not revision:
            continue
        if _EXACT_REVISION.fullmatch(revision) is None:
            raise ValueError(f"{field} must be an exact 40-character Git SHA")
        return revision

    try:
        completed = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        completed = None
    revision = completed.stdout.strip().lower() if completed and completed.returncode == 0 else ""
    if _EXACT_REVISION.fullmatch(revision):
        return revision
    raise ValueError(
        "source revision is unavailable; set TRTMC_ENGINE_BUILD_REVISION to the exact Git SHA"
    )


def _build_provenance(request: BuildRequest) -> dict[str, object]:
    options: dict[str, object] = {
        "family": request.family,
        "task": request.task,
        "backend": request.backend,
        "precision": request.precision,
        "max_batch_size": request.max_batch_size,
        "tensor_parallel_size": request.tensor_parallel_size,
        "context_parallel_size": request.context_parallel_size,
        "dynamic_kv_cache": request.dynamic_kv_cache,
    }
    if request.max_sequence_length is not None:
        options["max_sequence_length"] = request.max_sequence_length
    if request.image_height is not None:
        options["image_height"] = request.image_height
    if request.image_width is not None:
        options["image_width"] = request.image_width
    if request.video_num_frames is not None:
        options["video_num_frames"] = request.video_num_frames
    if request.quantization is not None:
        options["quantization"] = request.quantization
    if request.fp32_layers:
        options["fp32_layers"] = list(request.fp32_layers)
    return {
        "format": 1,
        "checkpoint": {
            "id": request.checkpoint_id or str(request.model_dir.resolve()),
            "revision": request.checkpoint_revision or "unknown",
        },
        "build": {"source_revision": resolve_source_revision(request.source_revision)},
        "request": options,
    }
