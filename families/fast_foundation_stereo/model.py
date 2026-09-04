# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fast Foundation Stereo family plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

_CHECKPOINT = Path("weights/23-36-37/model_best_bp2_serialize.pth")


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one Fast Foundation Stereo bundle."""
    if request.context_parallel_size != 1:
        raise ValueError("this family does not support context parallelism")

    if request.task != "stereo_disparity":
        raise ValueError("fast_foundation_stereo supports only task=stereo_disparity")
    if request.image_height is not None:
        raise NotImplementedError("fast_foundation_stereo does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("fast_foundation_stereo does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("fast_foundation_stereo does not support video_num_frames")
    if request.max_sequence_length is not None:
        raise NotImplementedError("fast_foundation_stereo does not support max_sequence_length")
    if request.max_batch_size != 1:
        raise NotImplementedError("fast_foundation_stereo requires max_batch_size=1")
    if (
        request.tensor_parallel_size != 1
        or request.quantization not in {None, "none"}
        or request.fp32_layers
    ):
        raise NotImplementedError(
            "Fast Foundation Stereo supports only single-device non-quantized builds"
        )
    model_dir = Path(request.model_dir).resolve()
    required = (
        model_dir / "core/foundation_stereo.py",
        model_dir / "core/submodule.py",
        model_dir / _CHECKPOINT,
    )
    missing = [path.relative_to(model_dir).as_posix() for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Fast Foundation Stereo model directory is incomplete: " + ", ".join(missing)
        )

    from .builder import build_feature_engine, build_post_engine

    plan = build_feature_engine(
        str(model_dir),
        precision=request.precision,
        max_disparity=192,
        valid_iters=8,
        verbose=request.verbose,
    )
    post = build_post_engine(
        str(model_dir),
        precision=request.precision,
        max_disparity=192,
        valid_iters=8,
        verbose=request.verbose,
    )
    writer.set_header(family="fast_foundation_stereo", task=request.task, backend="trt")
    writer.add_bytes("engine.plan", plan)
    writer.add_bytes("post.plan", post)
