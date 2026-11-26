# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family-owned command handler; importing help does not load TensorRT."""

from pathlib import Path
from dataclasses import dataclass
from tensorrt_model_connect.build import select_backend
from tensorrt_model_connect.bundle_writer import BundleWriter
from tensorrt_model_connect.graph_transform import graph_transform
from tensorrt_model_connect.model_support import load_model_metadata, resolve_family, resolve_model


@dataclass(frozen=True)
class BuildRequest:
    model_dir: Path
    task: str = "text_generation"
    precision: str = "bf16"
    backend: str = "trt"
    max_sequence_length: int = 1024
    tensor_parallel_size: int = 1
    verbose: bool = False

    def __post_init__(self):
        if self.task != "text_generation" or self.precision != "bf16":
            raise ValueError("Flint requires text_generation with bf16 precision")
        if self.backend != "trt" or self.tensor_parallel_size != 1:
            raise ValueError("Flint requires TensorRT on one GPU")
        if type(self.max_sequence_length) is not int or self.max_sequence_length <= 0:
            raise ValueError("Flint max_sequence_length must be positive")


def coerce_request(request):
    if isinstance(request, BuildRequest):
        return request
    unsupported = {
        "max_batch_size": 1,
        "context_parallel_size": 1,
        "quantization": None,
        "dynamic_kv_cache": False,
        "fp32_layers": (),
        "image_height": None,
        "image_width": None,
        "video_num_frames": None,
        "mtp_seq_len": None,
    }
    for field, expected in unsupported.items():
        value = getattr(request, field, expected)
        if field == "quantization" and value == "none":
            value = None
        if value != expected:
            raise ValueError(f"Flint does not support {field}={value!r}")
    ignored = {"family", "output_path", "graph_transform", *unsupported}
    fields = BuildRequest.__dataclass_fields__
    if unknown := vars(request).keys() - fields.keys() - ignored:
        raise ValueError(f"unknown Flint build inputs: {sorted(unknown)}")
    values = {field: getattr(request, field) for field in fields}
    if values["max_sequence_length"] is None:
        values["max_sequence_length"] = 1024
    return BuildRequest(**values)


def build_bundle(request, output, *, transform=None):
    from .model import build as build_model

    select_backend(request.backend)
    writer = BundleWriter(output)
    writer.record_build(request)
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
    max_sequence_length: int = 1024,
    verbose: bool = False,
) -> int:
    directory = resolve_model(model, revision)
    resolve_family(load_model_metadata(directory), "flint")
    build_bundle(
        BuildRequest(model_dir=directory, max_sequence_length=max_sequence_length, verbose=verbose),
        output,
    )
    return 0
