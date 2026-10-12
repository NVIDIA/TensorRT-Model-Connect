# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned timm_seresnet commands with lazy builder imports."""

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
    task: str = "classification"
    precision: str = "fp32"
    backend: str = "trt"
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.task != "classification":
            raise ValueError("timm_seresnet supports only task=classification")
        if str(self.precision).lower() not in {"fp16", "fp32"}:
            raise ValueError("timm_seresnet supports only fp16 and fp32 precision")
        if self.backend not in {"trt", "trt_rtx"}:
            raise ValueError("backend must be 'trt' or 'trt_rtx'")


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _validate_legacy_inputs(request: object) -> None:
    if request.dynamic_kv_cache:
        raise NotImplementedError("timm_seresnet does not support dynamic_kv_cache")
    if request.image_height is not None:
        raise NotImplementedError("timm_seresnet does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("timm_seresnet does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("timm_seresnet does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("timm_seresnet does not support max_batch_size")


def _validate_legacy_build(request: object) -> None:
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("timm_seresnet does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("timm_seresnet does not support context parallelism")
    if request.task != "classification":
        raise ValueError("timm_seresnet supports only task=classification")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("timm_seresnet does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("timm_seresnet does not support mixed-precision layers")
    _positive_int(request.max_sequence_length or 1, "max_sequence_length")


def coerce_request(request: object) -> BuildRequest:
    """Retain legacy rejection behavior without retaining its shared request union."""
    if isinstance(request, BuildRequest):
        return request
    _validate_legacy_inputs(request)
    _validate_legacy_build(request)
    allowed = {
        "model_dir",
        "task",
        "precision",
        "backend",
        "verbose",
        "family",
        "output_path",
        "graph_transform",
        "dynamic_kv_cache",
        "image_height",
        "image_width",
        "video_num_frames",
        "max_batch_size",
        "tensor_parallel_size",
        "context_parallel_size",
        "quantization",
        "fp32_layers",
        "max_sequence_length",
    }
    if unknown := set(vars(request)) - allowed:
        raise ValueError(f"unknown timm_seresnet build inputs: {sorted(unknown)}")
    return BuildRequest(
        request.model_dir, request.task, request.precision, request.backend, request.verbose
    )


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
    task: str = "classification",
    precision: str = "fp32",
    backend: str = "trt",
    verbose: bool = False,
) -> int:
    request = BuildRequest(resolve_model(model, revision), task, precision, backend, verbose)
    build_bundle(request, output)
    return 0
