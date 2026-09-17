# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Boltz-2 owns its build and request-preparation command contracts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from tensorrt_model_connect.build import select_backend
from tensorrt_model_connect.bundle_writer import BundleWriter
from tensorrt_model_connect.graph_transform import GraphTransform, graph_transform
from tensorrt_model_connect.model_support import resolve_model


@dataclass(frozen=True)
class BuildRequest:
    model_dir: Path
    task: str = "structure_prediction"
    precision: str = "bf16"
    backend: str = "trt"
    max_sequence_length: int | None = 117
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.backend != "trt":
            raise NotImplementedError("boltz2 supports only the TensorRT backend")
        if self.task != "structure_prediction":
            raise ValueError("boltz2 supports only task=structure_prediction")
        if self.precision != "bf16":
            raise ValueError("boltz2 supports only bf16 precision")
        if self.max_sequence_length not in {None, 117}:
            raise NotImplementedError("boltz2 max_sequence_length must match the 117-token profile")


def coerce_request(request: object) -> BuildRequest:
    """Reject legacy options that cannot affect the bounded Boltz-2 build."""
    if isinstance(request, BuildRequest):
        return request
    for name, default in (("max_batch_size", 1), ("tensor_parallel_size", 1),
                          ("context_parallel_size", 1), ("dynamic_kv_cache", False),
                          ("image_height", None), ("image_width", None), ("video_num_frames", None)):
        if getattr(request, name, default) != default:
            raise NotImplementedError(f"boltz2 does not support {name}")
    if getattr(request, "quantization", None) not in {None, "none"} or getattr(request, "fp32_layers", ()):
        raise NotImplementedError("boltz2 does not support quantization or fp32 layer overrides")
    fields = BuildRequest.__dataclass_fields__
    legacy = {"family", "output_path", "graph_transform", "max_batch_size", "tensor_parallel_size",
              "context_parallel_size", "dynamic_kv_cache", "image_height", "image_width",
              "video_num_frames", "quantization", "fp32_layers"}
    if unknown := set(vars(request)) - set(fields) - legacy:
        raise ValueError(f"unknown Boltz-2 build inputs: {sorted(unknown)}")
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
          task: str = "structure_prediction", precision: str = "bf16", backend: str = "trt",
          max_sequence_length: int | None = 117, verbose: bool = False) -> int:
    request = BuildRequest(resolve_model(model, revision), task, precision, backend,
                           max_sequence_length, verbose)
    build_bundle(request, output)
    return 0


def prepare_structure(*, model: str, input: Path, output: Path,
                      revision: str | None = None, cache_dir: Path | None = None,
                      sampling_steps: int = 200, diffusion_samples: int = 1,
                      seed: int = 42, affinity_sampling_steps: int = 200,
                      affinity_diffusion_samples: int = 5) -> int:
    from .request_preparation import prepare_structure_request

    result = prepare_structure_request(
        resolve_model(model, revision), input, output, cache_dir=cache_dir,
        sampling_steps=sampling_steps, diffusion_samples=diffusion_samples, seed=seed,
        affinity_sampling_steps=affinity_sampling_steps,
        affinity_diffusion_samples=affinity_diffusion_samples,
    )
    print(json.dumps(result, sort_keys=True))
    return 0
