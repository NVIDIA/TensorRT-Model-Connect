# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a Deformable DETR bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorrt as trt
from safetensors import safe_open

from . import backbone_builder, checkpoint, config as config_module, decoder_builder
from . import encoder_builder
from . import graph as g, position_builder, projection_builder

_DEFAULT_IMAGE_SIZE = 800
_DEFAULT_THRESHOLD = 0.3
# The reference post-processor keeps 100 detections, not one per query.
_DEFAULT_TOP_K = 100


def _load_weights(model_dir: Path) -> dict:
    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"missing deformable_detr weights: {path}")
    raw: dict = {}
    with safe_open(str(path), framework="numpy") as reader:
        for key in reader.keys():
            raw[key] = reader.get_tensor(key)
    weights = checkpoint.normalise(raw)
    # Fail here, naming the key, rather than deep inside a builder. The file's
    # names are not the names a loaded model reports, so this is a real trap.
    required = (
        "model.backbone.model.conv1.weight",
        "model.level_embed",
        "model.query_position_embeddings.weight",
        "class_embed.0.weight",
    )
    missing = [key for key in required if key not in weights]
    if missing:
        raise KeyError(
            "deformable_detr checkpoint is missing expected tensors: " + ", ".join(missing))
    return weights


def build_detector_engine(weights, cfg, *, precision, verbose=False) -> bytes:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    work_np, work_trt = ((np.float16, trt.float16) if precision == "fp16"
                         else (np.float32, trt.float32))

    size = cfg["image_size"]
    pixels = network.add_input("pixel_values", trt.float32, (1, 3, -1, -1))
    source = pixels if pixels.dtype == work_trt else network.add_cast(
        pixels, work_trt).get_output(0)

    features = backbone_builder.build_backbone(network, source, weights, work_np)
    levels = projection_builder.build_projections(network, features, weights, work_np)
    shapes = [g.shape_of(network, level, [2, 3]) for level in levels]
    # One flat sequence, levels in order, highest resolution first.
    hidden = g.concat(
        network, [g.spatial_to_tokens(network, level) for level in levels], axis=1)

    position = position_builder.build_dynamic_level_embeddings(
        network, shapes, weights["model.level_embed"], work_np)

    memory = encoder_builder.build_encoder(
        network, hidden, position, weights, shapes, cfg, dtype=work_np)
    hidden, reference = decoder_builder.build_decoder(
        network, memory, weights, shapes, cfg, dtype=work_np)
    logits, boxes = decoder_builder.build_heads(
        network, hidden, reference, weights, cfg, dtype=work_np)

    for tensor, name in ((logits, "logits"), (boxes, "boxes")):
        out = tensor if tensor.dtype == trt.float32 else network.add_cast(
            tensor, trt.float32).get_output(0)
        out.name = name
        network.mark_output(out)

    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    profile = builder.create_optimization_profile()
    maximum = cfg["resize_longest_edge"]
    profile.set_shape("pixel_values", (1, 3, 1, 1), (1, 3, size, size),
                      (1, 3, maximum, maximum))
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("deformable_detr engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one Deformable DETR bundle."""
    if request.task != "object_detection":
        raise ValueError("deformable_detr supports only task=object_detection")
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("deformable_detr supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("deformable_detr does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("deformable_detr does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("deformable_detr does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("deformable_detr does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("deformable_detr does not support video_num_frames")
    if request.max_sequence_length is not None:
        raise NotImplementedError("deformable_detr does not support max_sequence_length")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("deformable_detr does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("deformable_detr does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("deformable_detr does not support MTP speculative decoding")

    precision = str(request.precision).lower()
    if precision not in {"fp16", "fp32"}:
        raise ValueError(f"Unsupported deformable_detr precision: {precision}")

    height = int(request.image_height or 0) or _DEFAULT_IMAGE_SIZE
    width = int(request.image_width or 0) or _DEFAULT_IMAGE_SIZE
    if height != width:
        raise NotImplementedError("deformable_detr resize target needs equal height and width")
    if height <= 0 or height % 32:
        raise ValueError("deformable_detr resize target must be a positive multiple of 32")

    model_dir = Path(request.model_dir)
    cfg = config_module.resolve(model_dir, image_size=height)
    weights = _load_weights(model_dir)

    writer.set_header(family="deformable_detr", task=request.task, backend=request.backend)
    writer.add_bytes("detector.plan", build_detector_engine(
        weights, cfg, precision=precision, verbose=bool(request.verbose)))
    writer.add_json(
        "runtime.json",
        {
            "input_image_h": height,
            "input_image_w": width,
            "resize_shortest_edge": cfg["resize_shortest_edge"],
            "resize_longest_edge": cfg["resize_longest_edge"],
            "num_queries": cfg["num_queries"],
            "num_labels": cfg["num_labels"],
            "score_threshold": _DEFAULT_THRESHOLD,
            "top_k": _DEFAULT_TOP_K,
            # This checkpoint's preprocessor does normalise, unlike RT-DETR's.
            "do_normalize": cfg["do_normalize"],
            "image_mean": cfg["image_mean"],
            "image_std": cfg["image_std"],
        },
    )
