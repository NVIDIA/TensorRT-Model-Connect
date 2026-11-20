# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phi4's public build command; the Edge interpreter is an explicit input."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from tensorrt_model_connect.build import build as publish_bundle
from tensorrt_model_connect.model_support import load_model_metadata, resolve_family, resolve_model


@dataclass(frozen=True)
class Phi4BuildRequest:
    """Family-owned vision-language build inputs, including the optional builder interpreter."""

    model_dir: Path
    output_path: Path
    task: str
    backend: str
    precision: str
    max_sequence_length: int | None
    quantization: str | None
    verbose: bool
    tensor_parallel_size: int
    edge_builder_python: Path | None

    family: ClassVar[str] = "phi4_multimodal"
    mtp_seq_len: ClassVar[None] = None
    dynamic_kv_cache: ClassVar[bool] = False
    image_height: ClassVar[None] = None
    image_width: ClassVar[None] = None
    video_num_frames: ClassVar[None] = None
    graph_transform: ClassVar[None] = None
    max_batch_size: int = 1
    context_parallel_size: ClassVar[int] = 1
    fp32_layers: ClassVar[tuple[int, ...]] = ()


def build(
    *, model: str, output: Path, revision: str | None = None, task: str = "vision_language_generation",
    precision: str = "fp32", backend: str = "trt", max_sequence_length: int | None = None,
    quantization: str | None = None, verbose: bool = False, tensor_parallel_size: int = 1,
    edge_builder_python: Path | None = None, max_batch_size: int = 1,
) -> int:
    """Resolve the checkpoint normally and pass the interpreter only to this owner."""
    if max_sequence_length is not None and max_sequence_length < 1:
        raise ValueError("max_sequence_length must be positive")
    if max_batch_size not in (1, 2):
        raise ValueError("max_batch_size must be 1 or 2")
    model_dir = resolve_model(model, revision)
    resolve_family(load_model_metadata(model_dir), "phi4_multimodal")
    publish_bundle(Phi4BuildRequest(
        model_dir=model_dir, output_path=output, task=task, backend=backend, precision=precision,
        max_sequence_length=max_sequence_length, quantization=quantization, verbose=verbose,
        tensor_parallel_size=tensor_parallel_size, edge_builder_python=edge_builder_python,
        max_batch_size=max_batch_size,
    ))
    return 0
