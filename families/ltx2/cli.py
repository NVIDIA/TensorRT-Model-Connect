# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LTX-2.5 build command (``trtmc ltx2 build``) and typed build inputs; importing this module is CPU-only."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from tensorrt_model_connect.build import select_backend
from tensorrt_model_connect.bundle_writer import BundleWriter
from tensorrt_model_connect.model_support import resolve_model

from .vae_tiling import TileConfig

TASK = "text_to_audio_video"


@dataclass(frozen=True)
class BuildRequest:
    model_dir: Path
    task: str = TASK
    precision: str = "bf16"
    backend: str = "trt"
    max_sequence_length: int | None = None
    image_height: int | None = None
    image_width: int | None = None
    video_num_frames: int | None = None
    context_parallel_size: int = 1
    vae_tiles: TileConfig = field(default_factory=TileConfig)
    two_stage: bool = False
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.backend not in {"trt", "trt_rtx"}:
            raise ValueError("backend must be 'trt' or 'trt_rtx'")
        self.vae_tiles.validate()


def coerce_request(request: object) -> BuildRequest:
    """Accept the shared ``trtmc build`` request; reject the options LTX-2.5 cannot honour."""
    if isinstance(request, BuildRequest):
        return request
    if getattr(request, "dynamic_kv_cache", False):
        raise NotImplementedError("ltx2 does not support dynamic_kv_cache")
    if getattr(request, "tensor_parallel_size", 1) != 1:
        raise NotImplementedError("ltx2 requires tensor_parallel_size=1 (it shards the video tokens)")
    if getattr(request, "max_batch_size", 1) != 1:
        raise NotImplementedError("ltx2 requires max_batch_size=1")
    if getattr(request, "quantization", None) not in (None, "none"):
        raise NotImplementedError("ltx2 does not support quantization")
    if getattr(request, "fp32_layers", ()):
        raise NotImplementedError("ltx2 does not support fp32_layers (its fp32 islands are fixed in the graph)")
    return BuildRequest(
        model_dir=Path(request.model_dir), task=request.task, precision=request.precision,
        backend=getattr(request, "backend", "trt"),
        max_sequence_length=getattr(request, "max_sequence_length", None),
        image_height=getattr(request, "image_height", None), image_width=getattr(request, "image_width", None),
        video_num_frames=getattr(request, "video_num_frames", None),
        context_parallel_size=int(getattr(request, "context_parallel_size", 1)),
        verbose=bool(getattr(request, "verbose", False)))


def build_bundle(request: BuildRequest, output: Path) -> None:
    select_backend(request.backend)
    from .model import build as build_model

    writer = BundleWriter(output)
    try:
        build_model(request, writer)
        writer.finish()
    except BaseException:
        writer.abort()
        raise


def build(*, model: str, output: Path, revision: str | None = None, precision: str = "bf16",
          backend: str = "trt", image_height: int | None = None, image_width: int | None = None,
          video_num_frames: int | None = None, max_sequence_length: int | None = None,
          context_parallel_size: int = 1, vae_tile_pixels: int = 512, vae_tile_overlap_pixels: int = 64,
          vae_tile_frames: int = 256, vae_tile_overlap_frames: int = 24, two_stage: bool = False,
          verbose: bool = False) -> int:
    tiles = TileConfig(tile_pixels=vae_tile_pixels, overlap_pixels=vae_tile_overlap_pixels,
                       tile_frames=vae_tile_frames, overlap_frames=vae_tile_overlap_frames)
    request = BuildRequest(model_dir=resolve_model(model, revision), precision=precision, backend=backend,
                           max_sequence_length=max_sequence_length, image_height=image_height,
                           image_width=image_width, video_num_frames=video_num_frames,
                           context_parallel_size=context_parallel_size, vae_tiles=tiles,
                           two_stage=two_stage, verbose=verbose)
    build_bundle(request, output)
    return 0
