# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""nemotron_h build inputs and strict compatibility for existing Python callers."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
import re
from typing import ClassVar

from tensorrt_model_connect.graph_transform import GraphTransform


def _validate_id(field: str, value: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[a-z][a-z0-9_]*", value) is None:
        raise ValueError(f"{field} must be a lowercase identifier")


@dataclass(frozen=True)
class BuildRequest:
    """nemotron_h-owned inputs; unsupported legacy controls are read-only defaults."""

    model_dir: Path
    output_path: Path
    family: str
    task: str
    precision: str
    backend: str = "trt"
    max_sequence_length: int | None = None
    image_height: ClassVar[int | None] = None
    image_width: ClassVar[int | None] = None
    video_num_frames: ClassVar[int | None] = None
    max_batch_size: ClassVar[int] = 1
    tensor_parallel_size: int = 1
    context_parallel_size: ClassVar[int] = 1
    quantization: ClassVar[str | None] = None
    fp32_layers: ClassVar[tuple[int, ...]] = ()
    dynamic_kv_cache: ClassVar[bool] = False
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


def coerce_request(request: object) -> BuildRequest:
    """Reject unsupported/unknown legacy inputs before converting to owner fields."""
    if isinstance(request, BuildRequest):
        return request
    unsupported = {
        "image_height": None,
        "image_width": None,
        "video_num_frames": None,
        "max_batch_size": 1,
        "context_parallel_size": 1,
        "quantization": None,
        "fp32_layers": (),
        "dynamic_kv_cache": False,
    }
    for name, default in unsupported.items():
        value = getattr(request, name, default)
        if name == "quantization" and value == "none":
            continue
        if name == "fp32_layers" and isinstance(value, (list, tuple)) and not value:
            continue
        if value != default:
            raise NotImplementedError(f"nemotron_h does not support {name}")
    names = {field.name for field in fields(BuildRequest)}
    if unknown := set(vars(request)) - names - set(unsupported):
        raise ValueError(f"unknown nemotron_h build inputs: {sorted(unknown)}")
    return BuildRequest(**{name: getattr(request, name) for name in names})
