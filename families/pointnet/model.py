# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the PointNet semantic-segmentation bundle."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .builder import build_pointnet_engine
from .config import ModelConfig


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one point-cloud segmentation bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("pointnet does not support dynamic_kv_cache")
    if request.task != "points_to_semantic_segmentation":
        raise ValueError("pointnet supports only task=points_to_semantic_segmentation")
    if request.precision not in {"fp16", "fp32"}:
        raise ValueError("pointnet supports only fp16 or fp32 precision")
    if request.max_sequence_length not in {None, 1}:
        raise NotImplementedError("pointnet supports only max_sequence_length=1")
    if request.image_height is not None:
        raise NotImplementedError("pointnet does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("pointnet does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("pointnet does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("pointnet does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("pointnet does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("pointnet does not support context parallelism")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("pointnet does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("pointnet does not support mixed-precision layers")

    config = ModelConfig.load(Path(request.model_dir))
    plan = build_pointnet_engine(
        Path(request.model_dir), config, request.precision, bool(request.verbose)
    )

    writer.set_header(family="pointnet", task=request.task, backend=request.backend)
    writer.add_json(
        "runtime.json",
        {
            "num_points": config.num_points,
            "num_classes": config.num_classes,
            "input_dim": config.input_dim,
        },
    )
    writer.add_bytes("engine.plan", plan)
