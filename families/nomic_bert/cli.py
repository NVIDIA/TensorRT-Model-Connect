# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tensorrt_model_connect.build import select_backend
from tensorrt_model_connect.bundle_writer import BundleWriter
from tensorrt_model_connect.graph_transform import graph_transform
from tensorrt_model_connect.model_support import resolve_model


@dataclass(frozen=True)
class BuildRequest:
    model_dir: Path
    task: str = "text_to_embedding"
    precision: str = "fp32"
    backend: str = "trt"
    max_sequence_length: int = 512
    verbose: bool = False

    def __post_init__(self):
        if self.task != "text_to_embedding" or self.precision != "fp32":
            raise ValueError("Nomic implements only FP32 text_to_embedding")
        if self.backend != "trt":
            raise ValueError("Nomic implements only backend='trt'")
        if type(self.max_sequence_length) is not int or not 2 <= self.max_sequence_length <= 2048:
            raise ValueError("max_sequence_length must be an integer in [2, 2048]")


def build_bundle(request: BuildRequest, output: Path, *, transform=None):
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


def coerce_request(request):
    if isinstance(request, BuildRequest):
        return request
    if not hasattr(request, "__dict__"):
        raise TypeError("Nomic requires build inputs")
    for name, default in (
        ("tensor_parallel_size", 1),
        ("context_parallel_size", 1),
        ("max_batch_size", 1),
        ("quantization", None),
        ("dynamic_kv_cache", False),
        ("fp32_layers", ()),
        ("image_height", None),
        ("image_width", None),
        ("video_num_frames", None),
    ):
        if getattr(request, name, default) != default:
            raise ValueError(f"Nomic does not support {name}")
    fields = BuildRequest.__dataclass_fields__
    legacy = {
        "family",
        "output_path",
        "graph_transform",
        "tensor_parallel_size",
        "context_parallel_size",
        "max_batch_size",
        "quantization",
        "dynamic_kv_cache",
        "fp32_layers",
        "image_height",
        "image_width",
        "video_num_frames",
    }
    if unknown := set(vars(request)) - set(fields) - legacy:
        raise ValueError(f"unknown Nomic build inputs: {sorted(unknown)}")
    return BuildRequest(
        request.model_dir,
        request.task,
        request.precision,
        request.backend,
        512 if request.max_sequence_length is None else request.max_sequence_length,
        request.verbose,
    )


def build(
    *,
    model: str,
    output: Path,
    revision: str | None = None,
    task="text_to_embedding",
    precision="fp32",
    backend="trt",
    max_sequence_length=512,
    verbose=False,
):
    request = BuildRequest(
        resolve_model(model, revision), task, precision, backend, max_sequence_length, verbose
    )
    build_bundle(request, output)
    return 0
