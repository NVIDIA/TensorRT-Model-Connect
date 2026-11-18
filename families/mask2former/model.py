# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a Mask2Former semantic-segmentation bundle.

One float32 engine maps the preprocessed image to the semantic scores the reference post-processor computes before
its final resize: Swin backbone, pixel decoder, masked-attention decoder, class probabilities and the einsum with the
sigmoid of the resized mask logits. The native runtime owns the image preprocessing, the final bilinear resize to
the original image size and the argmax.
"""

from __future__ import annotations

from pathlib import Path

import tensorrt as trt

from . import config as config_module
from . import weights as weights_module
from .config import SEMANTIC_SCORE_SIZE, Mask2FormerConfig
from .pixel_decoder import build_pixel_decoder
from .swin import build_backbone
from .transformer_decoder import build_semantic_scores

_TASK = "image_to_semantic_segmentation"


def _build_engine(weights, config: Mask2FormerConfig, *, verbose: bool) -> bytes:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    pixels = network.add_input(
        "pixel_values", trt.float32, (1, 3, config.image_height, config.image_width)
    )
    features = build_backbone(network, pixels, weights, config)
    multi_scale, mask_features = build_pixel_decoder(network, features, weights, config)
    scores = build_semantic_scores(network, multi_scale, mask_features, weights, config)
    scores.name = "semantic_scores"
    network.mark_output(scores)

    builder_config = builder.create_builder_config()
    builder_config.builder_optimization_level = 3
    # The checkpoint is float32 and so is the reference; TF32 would round every matrix product to 10 mantissa bits.
    builder_config.clear_flag(trt.BuilderFlag.TF32)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("mask2former engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one Mask2Former bundle for `image_to_semantic_segmentation`."""

    if request.task != _TASK:
        raise ValueError(f"mask2former supports only task={_TASK}")
    if request.backend != "trt":
        raise ValueError("mask2former supports only backend=trt")
    precision = str(request.precision).lower()
    if precision != "fp32":
        raise ValueError(
            f"mask2former builds the float32 checkpoint at fp32 only, got precision={precision}"
        )
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("mask2former does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("mask2former does not support mixed-precision layers")
    if request.max_batch_size != 1:
        raise NotImplementedError("mask2former builds one image per run (max_batch_size=1)")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("mask2former does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("mask2former does not support context parallelism")
    if request.dynamic_kv_cache:
        raise NotImplementedError("mask2former does not support dynamic_kv_cache")
    if request.video_num_frames is not None:
        raise NotImplementedError("mask2former does not support video_num_frames")
    if request.max_sequence_length not in {None, 1}:
        raise NotImplementedError("mask2former has no sequence length to configure")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("mask2former does not support MTP speculative decoding")

    model_dir = Path(request.model_dir)
    config = config_module.load(model_dir)
    for value, expected, name in (
        (request.image_height, config.image_height, "image_height"),
        (request.image_width, config.image_width, "image_width"),
    ):
        if value is not None and value != expected:
            raise NotImplementedError(
                f"mask2former builds at its processor size {config.image_height}x{config.image_width}; "
                f"{name}={value} is not supported"
            )
    plan = _build_engine(
        weights_module.load(model_dir, config), config, verbose=bool(request.verbose)
    )

    writer.set_header(family="mask2former", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_json(
        "runtime.json",
        {
            "input_height": config.image_height,
            "input_width": config.image_width,
            "score_height": SEMANTIC_SCORE_SIZE,
            "score_width": SEMANTIC_SCORE_SIZE,
            "num_classes": config.num_labels,
            "rescale_factor": config.rescale_factor,
            "image_mean": list(config.image_mean),
            "image_std": list(config.image_std),
            "labels": list(config.labels),
        },
    )
