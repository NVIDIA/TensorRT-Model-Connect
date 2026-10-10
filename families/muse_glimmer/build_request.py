# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse-Glimmer-owned build inputs and compatibility for Python callers."""

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
    """Inputs for the one qualified Muse-Glimmer NVFP4 Edge profile."""

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
    tensor_parallel_size: ClassVar[int] = 1
    context_parallel_size: ClassVar[int] = 1
    quantization: str | None = "nvfp4"
    fp32_layers: ClassVar[tuple[int, ...]] = ()
    dynamic_kv_cache: ClassVar[bool] = False
    execution_variant: str = "autoregressive"
    companion: Path | None = None
    verbose: bool = False
    graph_transform: GraphTransform | None = None

    def __post_init__(self) -> None:
        _validate_id("family", self.family)
        _validate_id("task", self.task)
        if self.family != "muse_glimmer":
            raise ValueError("Muse-Glimmer request family must be muse_glimmer")
        if self.task != "text_generation":
            raise ValueError("Muse-Glimmer supports only task=text_generation")
        if self.precision != "fp16":
            raise ValueError("Muse-Glimmer supports only precision=fp16")
        if self.backend != "trt":
            raise ValueError("Muse-Glimmer supports only backend=trt")
        if self.max_sequence_length is not None and self.max_sequence_length < 1:
            raise ValueError("max_sequence_length must be positive")
        if self.quantization not in {None, "none", "nvfp4"}:
            raise ValueError("Muse-Glimmer supports only quantization=nvfp4")
        if self.execution_variant not in {"autoregressive", "dflash"}:
            raise ValueError("Muse-Glimmer supports autoregressive or dflash execution")
        if (self.execution_variant == "dflash") != (self.companion is not None):
            raise ValueError("Muse-Glimmer dflash requires exactly one companion")
        if self.graph_transform is not None:
            raise NotImplementedError("Muse-Glimmer Edge offload does not support graph_transform")


def coerce_request(request: object) -> BuildRequest:
    """Reject unsupported shared inputs before converting to owner fields."""
    if isinstance(request, BuildRequest):
        return request
    unsupported = {
        "image_height": None,
        "image_width": None,
        "video_num_frames": None,
        "max_batch_size": 1,
        "tensor_parallel_size": 1,
        "context_parallel_size": 1,
        "fp32_layers": (),
        "dynamic_kv_cache": False,
    }
    for name, default in unsupported.items():
        if getattr(request, name, default) != default:
            raise NotImplementedError(f"muse_glimmer does not support {name}")
    names = {field.name for field in fields(BuildRequest)}
    if unknown := set(vars(request)) - names - set(unsupported):
        raise ValueError(f"unknown muse_glimmer build inputs: {sorted(unknown)}")
    return BuildRequest(**{name: getattr(request, name) for name in names if hasattr(request, name)})
