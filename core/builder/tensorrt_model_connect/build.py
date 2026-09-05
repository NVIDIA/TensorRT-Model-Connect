# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dispatch a build to exactly one model family."""

from __future__ import annotations

import importlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from .bundle_writer import BundleWriter
from .graph_transform import GraphTransform, graph_transform


_ID = re.compile(r"[a-z][a-z0-9_]*\Z")


@dataclass(frozen=True)
class BuildRequest:
    """Inputs shared by the build core and one family-owned builder."""

    model_dir: Path
    output_path: Path
    family: str
    task: str
    precision: str
    backend: str = "trt"
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
        writer.finish()
    except BaseException:
        writer.abort()
        raise
