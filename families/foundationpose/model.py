# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the FoundationPose refiner and scorer from their exact ONNX weight containers."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .builder import build_foundationpose_engine


if TYPE_CHECKING:
    from tensorrt_model_connect.build import BuildRequest
    from tensorrt_model_connect.bundle_writer import BundleWriter

_REFINER_FILE = "refine_model.onnx"
_SCORER_FILE = "score_model.onnx"


def _model_files(model_dir: Path) -> tuple[Path, Path]:
    refiner = model_dir / _REFINER_FILE
    scorer = model_dir / _SCORER_FILE
    missing = [path.name for path in (refiner, scorer) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"FoundationPose model directory is missing required files: {missing}"
        )
    return refiner, scorer


def build(request: "BuildRequest", writer: "BundleWriter") -> None:
    """Build one pose-hypothesis refinement bundle."""
    if request.task != "pose_hypothesis_refinement":
        raise ValueError("foundationpose supports only task=pose_hypothesis_refinement")
    if request.precision not in {"fp16", "fp32"}:
        raise ValueError("foundationpose supports only fp16 or fp32 precision")
    if request.max_sequence_length not in {None, 1}:
        raise NotImplementedError("foundationpose supports only max_sequence_length=1")
    if request.image_height is not None:
        raise NotImplementedError("foundationpose does not support image_height")
    if request.image_width is not None:
        raise NotImplementedError("foundationpose does not support image_width")
    if request.video_num_frames is not None:
        raise NotImplementedError("foundationpose does not support video_num_frames")
    if request.max_batch_size != 1:
        raise NotImplementedError("foundationpose does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("foundationpose does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("foundationpose does not support context parallelism")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("foundationpose does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("foundationpose does not support mixed-precision layers")

    refiner, scorer = _model_files(Path(request.model_dir))
    refiner_plan = build_foundationpose_engine(
        str(refiner),
        kind="refiner",
        max_batch=42,
        precision=request.precision,
        verbose=bool(request.verbose),
    )
    scorer_plan = build_foundationpose_engine(
        str(scorer),
        kind="scorer",
        max_batch=252,
        precision=request.precision,
        verbose=bool(request.verbose),
    )
    writer.set_header(family="foundationpose", task=request.task, backend="trt")
    writer.add_bytes("engine.plan", refiner_plan)
    writer.add_bytes("score.plan", scorer_plan)
