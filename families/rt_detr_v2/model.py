# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one RT-DETR v2 object-detection bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorrt as trt
from safetensors import safe_open

from . import backbone_builder, config as config_module, decoder_builder, encoder_builder
from . import graph as g

_DEFAULT_IMAGE_SIZE = 640
_DEFAULT_THRESHOLD = 0.3


def _load_weights(model_dir: Path) -> dict:
    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"missing rt_detr_v2 weights: {path}")
    weights: dict = {}
    with safe_open(str(path), framework="numpy") as reader:
        for key in reader.keys():
            weights[key] = reader.get_tensor(key)
    return weights


def build_detector_engine(weights, cfg, *, precision, verbose=False) -> bytes:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    work_np, work_trt = ((np.float16, trt.float16) if precision == "fp16"
                         else (np.float32, trt.float32))

    size = cfg["image_size"]
    pixels = network.add_input("pixel_values", trt.float32, (1, 3, size, size))
    source = pixels if pixels.dtype == work_trt else network.add_cast(
        pixels, work_trt).get_output(0)

    levels = backbone_builder.build_backbone(network, source, weights, cfg, work_np)
    fused = encoder_builder.build_encoder(network, levels, weights, cfg, work_np)
    projected = encoder_builder.project_for_decoder(network, fused, weights, work_np)
    # The decoder reads one flat sequence, levels in order, highest resolution first.
    memory = g.concat(network, [g.spatial_to_tokens(network, level) for level in projected],
                      axis=1)
    logits, boxes = decoder_builder.build_decoder(
        network, memory, cfg["shapes"], weights, cfg, work_np)

    for tensor, name in ((logits, "logits"), (boxes, "boxes")):
        out = tensor if tensor.dtype == trt.float32 else network.add_cast(
            tensor, trt.float32).get_output(0)
        out.name = name
        network.mark_output(out)

    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("rt_detr_v2 engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one RT-DETR v2 bundle."""
    if request.task != "object_detection":
        raise ValueError("rt_detr_v2 supports only task=object_detection")
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("rt_detr_v2 supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("rt_detr_v2 does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("rt_detr_v2 does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("rt_detr_v2 does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("rt_detr_v2 does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("rt_detr_v2 does not support video_num_frames")
    if request.max_sequence_length is not None:
        raise NotImplementedError("rt_detr_v2 does not support max_sequence_length")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("rt_detr_v2 does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("rt_detr_v2 does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("rt_detr_v2 does not support MTP speculative decoding")

    precision = str(request.precision).lower()
    if precision not in {"fp16", "fp32"}:
        raise ValueError(f"Unsupported rt_detr_v2 precision: {precision}")

    height = int(request.image_height or 0) or _DEFAULT_IMAGE_SIZE
    width = int(request.image_width or 0) or _DEFAULT_IMAGE_SIZE
    if height != width:
        raise NotImplementedError("rt_detr_v2 builds a square input only")
    if height % 32:
        raise ValueError("rt_detr_v2 input size must be a multiple of 32")

    model_dir = Path(request.model_dir)
    cfg = config_module.resolve(model_dir, image_size=height)
    weights = _load_weights(model_dir)

    writer.set_header(family="rt_detr_v2", task=request.task, backend=request.backend)
    writer.add_bytes("detector.plan", build_detector_engine(
        weights, cfg, precision=precision, verbose=bool(request.verbose)))
    writer.add_json(
        "runtime.json",
        {
            "input_image_h": height,
            "input_image_w": width,
            "num_queries": cfg["num_queries"],
            "num_labels": cfg["num_labels"],
            "score_threshold": _DEFAULT_THRESHOLD,
            # Recorded so the runtime cannot reintroduce it: the checkpoint's
            # preprocessor lists ImageNet statistics but disables them.
            "do_normalize": False,
        },
    )
