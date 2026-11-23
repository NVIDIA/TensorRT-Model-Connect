# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-owned build inputs; weight stripping does not extend the shared request."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from tensorrt_model_connect.graph_transform import GraphTransform


def _validate_id(field: str, value: object) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[a-z][a-z0-9_]*", value) is None:
        raise ValueError(f"{field} must be a lowercase identifier")


@dataclass(frozen=True)
class BuildRequest:
    """Inputs accepted or explicitly rejected by the Qwen builder."""

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
    strip_weights: bool = False
    verbose: bool = False
    graph_transform: GraphTransform | None = None
    # Rejected by Qwen; retained as an explicit input check.
    mtp_seq_len: int | None = None

    def __post_init__(self) -> None:
        if type(self.strip_weights) is not bool:
            raise ValueError("strip_weights must be a bool")
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
        if self.mtp_seq_len is not None and self.mtp_seq_len < 2:
            raise ValueError("mtp_seq_len must be >= 2 when provided")
