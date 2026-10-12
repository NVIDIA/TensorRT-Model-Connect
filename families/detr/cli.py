# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""detr CLI handlers and build inputs; model imports stay lazy."""

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
    task: str = "object_detection"
    precision: str = "fp32"
    backend: str = "trt"
    verbose: bool = False
    image_height: int | None = None
    image_width: int | None = None

    def __post_init__(self) -> None:
        if self.task != "object_detection":
            raise ValueError("detr supports only task=object_detection")
        if str(self.precision).lower() not in {"fp16", "fp32"}:
            raise ValueError("detr supports only fp16 or fp32")
        if self.backend not in {"trt", "trt_rtx"}:
            raise ValueError("backend must be 'trt' or 'trt_rtx'")
        for name in ("image_height", "image_width"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer")


def _legacy_common_options(request: object) -> None:
    if request.task != "object_detection":
        raise ValueError("detr supports only task=object_detection")
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("backend must be 'trt' or 'trt_rtx'")
    if request.dynamic_kv_cache:
        raise NotImplementedError("detr does not support dynamic_kv_cache")
    if request.max_sequence_length not in {None, 1}:
        raise NotImplementedError("detr supports only max_sequence_length=1")
    if request.max_batch_size != 1:
        raise NotImplementedError("detr does not support max_batch_size")


def _legacy_model_options(request: object) -> None:
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("detr does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("detr does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("detr does not support video_num_frames")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("detr does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("detr does not support mixed-precision layers")


def coerce_request(request: object) -> BuildRequest:
    """Preserve supported legacy inputs and reject unsupported nondefaults."""
    if isinstance(request, BuildRequest):
        return request
    _legacy_common_options(request)
    _legacy_model_options(request)
    fields = BuildRequest.__dataclass_fields__
    legacy = {"family", "output_path", "graph_transform", "dynamic_kv_cache", "image_height",
              "image_width", "video_num_frames", "max_batch_size", "context_parallel_size",
              "quantization", "fp32_layers", "tensor_parallel_size", "max_sequence_length"}
    if unknown := set(vars(request)) - set(fields) - legacy:
        raise ValueError(f"unknown detr build inputs: {sorted(unknown)}")
    return BuildRequest(**{name: getattr(request, name) for name in fields})


def build_bundle(request: BuildRequest, output: Path, *, transform: GraphTransform | None = None) -> None:
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


def build(*, model: str, output: Path, revision: str | None = None,
          task: str = "object_detection", precision: str = "fp32", backend: str = "trt",
          verbose: bool = False, image_height: int | None = None, image_width: int | None = None) -> int:
    request = BuildRequest(resolve_model(model, revision), task=task, precision=precision,
                           backend=backend, verbose=verbose, image_height=image_height, image_width=image_width)
    build_bundle(request, output)
    return 0
