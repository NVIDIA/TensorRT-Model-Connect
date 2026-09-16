# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BERT command handlers and typed build inputs; importing this module is CPU-only."""

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
    task: str = "encoding"
    precision: str = "fp32"
    backend: str = "trt"
    max_sequence_length: int | None = None
    tensor_parallel_size: int = 1
    fp32_layers: tuple[int, ...] = ()
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.max_sequence_length is not None and (
            type(self.max_sequence_length) is not int or self.max_sequence_length < 1
        ):
            raise ValueError("max_sequence_length must be positive")
        if type(self.tensor_parallel_size) is not int or self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be positive")
        if any(type(layer) is not int or layer < 0 for layer in self.fp32_layers):
            raise ValueError("fp32_layers must contain non-negative indices")
        if self.backend not in {"trt", "trt_rtx"}:
            raise ValueError("backend must be 'trt' or 'trt_rtx'")


def coerce_request(request: object) -> BuildRequest:
    """Keep the old Python API strict while the shared request is retired."""
    if isinstance(request, BuildRequest):
        return request
    for name, default in (("dynamic_kv_cache", False), ("image_height", None),
                          ("image_width", None), ("video_num_frames", None),
                          ("max_batch_size", 1)):
        if getattr(request, name, default) != default:
            raise NotImplementedError(f"bert does not support {name}")
    if getattr(request, "context_parallel_size", 1) != 1:
        raise ValueError("this family does not support context parallelism")
    if getattr(request, "quantization", None) is not None:
        raise ValueError("BERT does not support quantization")
    fields = BuildRequest.__dataclass_fields__
    legacy = {"family", "output_path", "graph_transform", "dynamic_kv_cache", "image_height",
              "image_width", "video_num_frames", "max_batch_size", "context_parallel_size", "quantization"}
    if unknown := set(vars(request)) - set(fields) - legacy:
        raise ValueError(f"unknown BERT build inputs: {sorted(unknown)}")
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
          task: str = "encoding", precision: str = "fp32", backend: str = "trt",
          max_sequence_length: int | None = None, tensor_parallel_size: int = 1,
          fp32_layers: list[int] | tuple[int, ...] = (), verbose: bool = False) -> int:
    request = BuildRequest(resolve_model(model, revision), task, precision, backend,
                           max_sequence_length, tensor_parallel_size, tuple(fp32_layers), verbose)
    build_bundle(request, output)
    return 0
