# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the Uni3D point-cloud embedding bundle."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .builder import build_uni3d_engine
from .config import ModelConfig


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one point-cloud embedding bundle."""
    if request.dynamic_kv_cache:
        raise NotImplementedError("uni3d does not support dynamic_kv_cache")
    if request.task != "points_to_embedding":
        raise ValueError("uni3d supports only task=points_to_embedding")
    if request.max_sequence_length not in {None, 1}:
        raise NotImplementedError("uni3d supports only max_sequence_length=1")
    if request.image_height is not None:
        raise NotImplementedError("uni3d does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("uni3d does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("uni3d does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("uni3d does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("uni3d does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("uni3d does not support context parallelism")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("uni3d does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("uni3d does not support mixed-precision layer lists")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("uni3d does not support MTP speculative decoding")

    config = ModelConfig.load(Path(request.model_dir))
    if request.precision != config.precision:
        raise ValueError(
            f"uni3d.onnx was exported for precision={config.precision}; "
            f"re-export it for precision={request.precision}"
        )
    plan = build_uni3d_engine(Path(request.model_dir), config, bool(request.verbose))

    writer.set_header(family="uni3d", task=request.task, backend=request.backend)
    writer.add_json(
        "runtime.json",
        {
            "num_points": config.num_points,
            "min_points": config.num_group,
            "input_dim": config.input_dim,
            "embed_dim": config.embed_dim,
        },
    )
    writer.add_bytes("engine.plan", plan)
