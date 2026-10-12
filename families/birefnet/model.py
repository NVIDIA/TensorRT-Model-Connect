# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build one BiRefNet segmentation bundle."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorrt as trt
from safetensors import safe_open

from . import config as config_module, decoder_builder, swin_builder

_DEFAULT_IMAGE_SIZE = 1024


def _load_weights(model_dir: Path) -> dict:
    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"missing birefnet weights: {path}")
    weights: dict = {}
    with safe_open(str(path), framework="numpy") as reader:
        for key in reader.keys():
            weights[key] = reader.get_tensor(key)
    return weights


def build_segmenter_engine(weights, cfg, *, precision, verbose=False) -> bytes:
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

    levels = swin_builder.build_dual_scale(network, source, weights, cfg, dtype=work_np)
    logits = decoder_builder.build_decoder(network, source, levels, weights, dtype=work_np)
    out = logits if logits.dtype == trt.float32 else network.add_cast(
        logits, trt.float32).get_output(0)
    out.name = "logits"
    network.mark_output(out)

    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("birefnet engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one BiRefNet bundle."""
    if request.task != "segmentation":
        raise ValueError("birefnet supports only task=segmentation")
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("birefnet supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("birefnet does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("birefnet does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("birefnet does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("birefnet does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("birefnet does not support video_num_frames")
    if request.max_sequence_length is not None:
        raise NotImplementedError("birefnet does not support max_sequence_length")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("birefnet does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("birefnet does not support mixed-precision layers")

    precision = str(request.precision).lower()
    if precision not in {"fp16", "fp32"}:
        raise ValueError(f"Unsupported birefnet precision: {precision}")

    height = int(request.image_height or 0) or _DEFAULT_IMAGE_SIZE
    width = int(request.image_width or 0) or _DEFAULT_IMAGE_SIZE
    if height != width:
        raise NotImplementedError("birefnet builds a square input only")

    model_dir = Path(request.model_dir)
    cfg = config_module.resolve(model_dir, image_size=height)
    weights = _load_weights(model_dir)
    config_module.check_weights(weights)

    writer.set_header(family="birefnet", task=request.task, backend=request.backend)
    writer.add_bytes("segmenter.plan", build_segmenter_engine(
        weights, cfg, precision=precision, verbose=bool(request.verbose)))
    writer.add_json(
        "runtime.json",
        {
            "input_image_h": height,
            "input_image_w": width,
            "image_mean": cfg["image_mean"],
            "image_std": cfg["image_std"],
            "mask_threshold": cfg["mask_threshold"],
        },
    )
