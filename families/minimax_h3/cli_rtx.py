# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build MiniMax-H3 bundles with the family-owned TensorRT or TensorRT-RTX profile."""

from __future__ import annotations

from pathlib import Path

from tensorrt_model_connect.build import BuildRequest, select_backend
from tensorrt_model_connect.bundle_writer import BundleWriter
from tensorrt_model_connect.model_support import resolve_model

from .runtime_config_schema import normalize_build_options


def _build(request: BuildRequest, options: dict) -> int:
    select_backend(request.backend)
    if request.backend == "trt_rtx":
        from .rtx_model import build as build_model
    else:
        from .model import build as build_model

    writer = BundleWriter(request.output_path)
    writer.record_build(request)
    try:
        build_model(request, writer, **options)
        writer.finish()
    except BaseException:
        writer.abort()
        raise
    return 0


def build_trt(
    *,
    model: str,
    output: Path,
    revision: str | None = None,
    first_block_cache: bool = False,
    quantization: str = "none",
    verbose: bool = False,
) -> int:
    if first_block_cache and quantization != "none":
        raise ValueError("MiniMax-H3 FP8 requires the monolithic denoiser")
    request = BuildRequest(
        model_dir=resolve_model(model, revision),
        output_path=output,
        family="minimax_h3",
        task="image_generation",
        precision="bf16",
        backend="trt",
        quantization=quantization,
        verbose=verbose,
    )
    return _build(request, {"first_block_cache": first_block_cache})


def build_rtx(
    *,
    model: str,
    output: Path,
    revision: str | None = None,
    image_height: int | None = None,
    image_width: int | None = None,
    video_num_frames: int | None = None,
    verbose: bool = False,
    **options: object,
) -> int:
    normalized = normalize_build_options(
        {key: str(value) if isinstance(value, Path) else value for key, value in options.items()}
    )
    if (image_width is None) != (image_height is None):
        raise ValueError("MiniMax-H3 requires both image height and width, or neither")
    request = BuildRequest(
        model_dir=resolve_model(model, revision),
        output_path=output,
        family="minimax_h3",
        task="image_generation",
        precision="bf16",
        backend="trt_rtx",
        image_height=image_height,
        image_width=image_width,
        video_num_frames=video_num_frames,
        verbose=verbose,
    )
    return _build(request, {"options": normalized})
