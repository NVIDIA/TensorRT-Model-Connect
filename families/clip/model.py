# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a CLIP bundle.

The bundle carries **two** engines, because CLIP is two towers that meet only
at a dot product. Fusing them into one engine would force every call to run
both towers, which is the wrong shape for the three tasks this family serves:
embedding an image, embedding a text, and scoring one against the other.

Neither engine normalises its output. The projected embedding is what the
checkpoint computes - its norm is around 8.8 for an image and 10.5 for a text,
not 1 - and the runtime is where L2 normalisation and the ``logit_scale``
belong, so that what the engine returns stays the thing the reference returns.

This family requires torch in the build environment: the published
``openai/clip-vit-base-patch32`` repository ships ``pytorch_model.bin`` and no
safetensors file.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import tensorrt as trt

from . import config as config_module, text_builder, vision_builder

_SUPPORTED_TASKS = ("image_to_embedding", "text_to_embedding", "text_image_to_relevance")


def _open_torch_checkpoint(model_dir: Path) -> dict:
    """Open the family's required pytorch_model.bin checkpoint."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("This family requires torch in the build environment") from error

    path = Path(model_dir) / "pytorch_model.bin"
    if not path.is_file():
        raise FileNotFoundError(f"Required PyTorch checkpoint is missing: {path}")
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    return {key: value.numpy() for key, value in state.items()}


def _require_weights(weights: dict) -> None:
    required = (
        "logit_scale",
        "text_model.embeddings.token_embedding.weight",
        "text_model.final_layer_norm.weight",
        "text_projection.weight",
        # The checkpoint spells this one without the 'e'; that is its name, not ours.
        "vision_model.pre_layrnorm.weight",
        "vision_model.embeddings.class_embedding",
        "vision_model.post_layernorm.weight",
        "visual_projection.weight",
    )
    missing = [key for key in required if key not in weights]
    if missing:
        raise KeyError("clip checkpoint is missing expected tensors: " + ", ".join(missing))


def _builder(precision, verbose):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    if precision == "fp32":
        # TF32 caps a conv-and-matmul graph near 0.99998 cosine by default,
        # which hides real graph error behind a plausible-looking number.
        config.clear_flag(trt.BuilderFlag.TF32)
    return builder, network, config


def _serialize(builder, network, config, what):
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"clip {what} engine build failed")
    return bytes(plan)


def build_vision_engine(weights, cfg, *, image_size, precision, verbose=False) -> bytes:
    builder, network, config = _builder(precision, verbose)
    work_np, work_trt = ((np.float16, trt.float16) if precision == "fp16"
                         else (np.float32, trt.float32))
    pixels = network.add_input("pixel_values", trt.float32, (1, 3, image_size, image_size))
    source = (pixels if pixels.dtype == work_trt
              else network.add_cast(pixels, work_trt).get_output(0))
    embeds = vision_builder.build_vision_tower(
        network, source, weights, cfg["vision"], weights["visual_projection.weight"],
        dtype=work_np)
    out = (embeds if embeds.dtype == trt.float32
           else network.add_cast(embeds, trt.float32).get_output(0))
    out.name = "image_embeds"
    network.mark_output(out)
    return _serialize(builder, network, config, "vision")


def build_text_engine(weights, cfg, *, max_text_len, precision, verbose=False) -> bytes:
    builder, network, config = _builder(precision, verbose)
    work_np = np.float16 if precision == "fp16" else np.float32
    input_ids = network.add_input("input_ids", trt.int32, (max_text_len,))
    eos_index = network.add_input("eos_index", trt.int32, (1,))
    embeds = text_builder.build_text_tower(
        network, input_ids, eos_index, weights, cfg["text"],
        weights["text_projection.weight"], dtype=work_np)
    out = (embeds if embeds.dtype == trt.float32
           else network.add_cast(embeds, trt.float32).get_output(0))
    out.name = "text_embeds"
    network.mark_output(out)
    return _serialize(builder, network, config, "text")


def build(request, writer) -> None:
    """Build one CLIP bundle."""
    if request.task not in _SUPPORTED_TASKS:
        raise ValueError("clip supports only task=" + ", ".join(_SUPPORTED_TASKS))
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("clip supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("clip does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("clip does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("clip does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("clip does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("clip does not support video_num_frames")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("clip does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("clip does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("clip does not support MTP speculative decoding")

    precision = str(request.precision).lower()
    if precision not in {"fp16", "fp32"}:
        raise ValueError(f"Unsupported clip precision: {precision}")

    model_dir = Path(request.model_dir)
    cfg = config_module.resolve(model_dir)
    vision = cfg["vision"]

    image_size = int(request.image_height or 0) or vision["image_size"]
    if request.image_width is not None and int(request.image_width) != image_size:
        raise NotImplementedError("clip needs a square input size")
    if image_size != vision["image_size"]:
        # The position table has one row per patch of the trained grid, so a
        # different input size needs the table interpolated, not just resized.
        raise NotImplementedError(
            f"clip input size is fixed at the checkpoint's {vision['image_size']}")

    max_text_len = cfg["text"]["max_position_embeddings"]
    if request.max_sequence_length is not None:
        if int(request.max_sequence_length) != max_text_len:
            raise NotImplementedError(
                "clip prompt length is fixed at the checkpoint's max_position_embeddings")

    weights = _open_torch_checkpoint(model_dir)
    _require_weights(weights)

    tokenizer = model_dir / "tokenizer.json"
    if not tokenizer.is_file():
        raise FileNotFoundError(f"missing clip tokenizer: {tokenizer}")

    # Stored exponentiated, the way the reference applies it: the checkpoint
    # keeps log scale (4.60517) and every use is exp() of it.
    logit_scale = float(math.exp(float(np.asarray(weights["logit_scale"]).reshape(()))))

    writer.set_header(family="clip", task=request.task, backend=request.backend)
    writer.add_bytes("vision.plan", build_vision_engine(
        weights, cfg, image_size=image_size, precision=precision,
        verbose=bool(request.verbose)))
    writer.add_bytes("text.plan", build_text_engine(
        weights, cfg, max_text_len=max_text_len, precision=precision,
        verbose=bool(request.verbose)))
    writer.add_bytes("tokenizer.json", tokenizer.read_bytes())
    writer.add_json(
        "runtime.json",
        {
            "input_image_h": image_size,
            "input_image_w": image_size,
            "image_mean": cfg["image_mean"],
            "image_std": cfg["image_std"],
            "max_text_len": max_text_len,
            "projection_dim": cfg["projection_dim"],
            "logit_scale": logit_scale,
        },
    )
