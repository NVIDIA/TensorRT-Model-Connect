# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""gpt_neo command handlers and model-owned build inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tensorrt_model_connect.build import select_backend
from tensorrt_model_connect.bundle_writer import BundleWriter
from tensorrt_model_connect.graph_transform import GraphTransform, graph_transform
from tensorrt_model_connect.model_support import resolve_model


@dataclass(frozen=True)
class BuildRequest:
    model_dir: Path
    task: str = "text_generation"
    precision: str = "fp32"
    backend: str = "trt"
    max_sequence_length: int | None = None
    tensor_parallel_size: int = 1
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.task not in {"text_generation"}:
            raise ValueError("unsupported gpt_neo task")
        if str(self.precision).lower() not in {"fp16", "bf16", "fp32"}:
            raise ValueError("unsupported gpt_neo precision")
        if self.backend not in {"trt", "trt_rtx"}:
            raise ValueError("backend must be trt or trt_rtx")
        if self.max_sequence_length is not None and (
            type(self.max_sequence_length) is not int or self.max_sequence_length < 1
        ):
            raise ValueError("max_sequence_length must be positive")
        if type(self.tensor_parallel_size) is not int or self.tensor_parallel_size not in {
            1,
            2,
            4,
            8,
        }:
            raise ValueError("tensor_parallel_size must be 1, 2, 4, or 8")


def _reject_legacy_options(request: object) -> None:
    if request.dynamic_kv_cache:
        raise NotImplementedError("gpt_neo does not support dynamic_kv_cache")
    if request.image_height is not None:
        raise NotImplementedError("gpt_neo does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("gpt_neo does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("gpt_neo does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("gpt_neo does not support max_batch_size")
    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("GPT-Neo has no qualified family-owned quantized build")
    if request.fp32_layers:
        raise NotImplementedError("GPT-Neo does not expose mixed-precision layer selection")


def coerce_request(request: object) -> BuildRequest:
    """Preserve the legacy Python API's unsupported-value rejection."""
    if isinstance(request, BuildRequest):
        return request
    _reject_legacy_options(request)
    fields = BuildRequest.__dataclass_fields__
    legacy = {
        "family",
        "dynamic_kv_cache",
        "quantization",
        "graph_transform",
        "fp32_layers",
        "video_num_frames",
        "context_parallel_size",
        "max_batch_size",
        "image_height",
        "output_path",
        "image_width",
    }
    if unknown := set(vars(request)) - set(fields) - legacy:
        raise ValueError(f"unknown gpt_neo build inputs: {sorted(unknown)}")
    return BuildRequest(**{name: getattr(request, name) for name in fields})


def build_bundle(
    request: BuildRequest, output: Path, *, transform: GraphTransform | None = None
) -> None:
    select_backend(request.backend)
    from .model import build as build_model

    writer = BundleWriter(output)
    try:
        with graph_transform(transform):
            build_model(request, writer)
        writer.finish()
    except BaseException:
        writer.abort()
        raise


def build(
    *,
    model: str,
    output: Path,
    revision: str | None = None,
    task: str = "text_generation",
    precision: str = "fp32",
    backend: str = "trt",
    max_sequence_length: int | None = None,
    tensor_parallel_size: int = 1,
    verbose: bool = False,
) -> int:
    request = BuildRequest(
        model_dir=resolve_model(model, revision),
        task=task,
        precision=precision,
        backend=backend,
        max_sequence_length=max_sequence_length,
        tensor_parallel_size=tensor_parallel_size,
        verbose=verbose,
    )
    build_bundle(request, output)
    return 0
