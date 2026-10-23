# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a ViT bundle.

One engine, one output: the class logits. The classifier reads the class row
of the final layer norm - ``logits = classifier(last_hidden[:, 0])`` - and
nothing here softmaxes them, because the Task reports a ScoreKind and the
honest kind for this head is a logit.

The class vocabulary rides in the bundle rather than being looked up at
runtime, so a bundle names its own classes or says it cannot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorrt as trt

from . import config as config_module, encoder_builder, graph as g

_SUPPORTED_TASKS = ("image_to_class_scores",)


def _open_checkpoint(model_dir: Path) -> dict:
    """Load the safetensors checkpoint this family publishes."""
    try:
        from safetensors.numpy import load_file
    except ImportError as error:
        raise RuntimeError("This family requires safetensors in the build environment") from error

    path = Path(model_dir) / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Required safetensors checkpoint is missing: {path}")
    return dict(load_file(str(path)))


def _require_weights(weights: dict, prefix: str) -> None:
    required = (
        f"{prefix}embeddings.cls_token",
        f"{prefix}embeddings.position_embeddings",
        f"{prefix}embeddings.patch_embeddings.projection.weight",
        f"{prefix}embeddings.patch_embeddings.projection.bias",
        f"{prefix}layernorm.weight",
        f"{prefix}layernorm.bias",
        "classifier.weight",
        "classifier.bias",
    )
    missing = [key for key in required if key not in weights]
    if missing:
        raise KeyError("vit checkpoint is missing expected tensors: " + ", ".join(missing))


def _builder(verbose):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    # TF32 caps a conv-and-matmul graph near 0.99998 cosine by default,
    # which hides real graph error behind a plausible-looking number.
    config.clear_flag(trt.BuilderFlag.TF32)
    return builder, network, config


def build_engine(weights, cfg, *, verbose=False) -> bytes:
    builder, network, config = _builder(verbose)
    # fp32 throughout. build() refuses every other precision and records the
    # measurement behind that refusal.
    work_np = np.float32
    prefix = cfg["prefix"]
    width = cfg["hidden_size"]
    patch = cfg["patch_size"]
    grid = cfg["grid"]
    size = cfg["image_size"]

    source = network.add_input("pixel_values", trt.float32, (1, 3, size, size))

    patches = g.add_conv2d(network, source,
                           weights[f"{prefix}embeddings.patch_embeddings.projection.weight"],
                           weights[f"{prefix}embeddings.patch_embeddings.projection.bias"],
                           stride=(patch, patch), dtype=work_np)
    flat = network.add_shuffle(patches)
    flat.reshape_dims = (width, grid * grid)
    flat.second_transpose = (1, 0)

    # The class token is prepended, so row 0 of the position table is its row.
    class_token = g.add_constant(
        network, (1, width),
        np.asarray(weights[f"{prefix}embeddings.cls_token"]).reshape(1, width), dtype=work_np)
    concat = network.add_concatenation([class_token, flat.get_output(0)])
    concat.axis = 0
    hidden = concat.get_output(0)

    # This table already covers the grid the engine runs at, so unlike
    # DINOv2's it is added as-is. config.py refuses a checkpoint whose
    # processed size would not match it rather than resizing silently.
    table = np.asarray(weights[f"{prefix}embeddings.position_embeddings"]).reshape(
        grid * grid + 1, width)
    hidden = g.add_sum(network, hidden,
                       g.add_constant(network, table.shape, table, dtype=work_np))

    hidden = encoder_builder.build_layers(network, hidden, weights, cfg, dtype=work_np)
    hidden = g.add_layer_norm(network, hidden, weights[f"{prefix}layernorm.weight"],
                              weights[f"{prefix}layernorm.bias"], cfg["eps"])

    # The head reads the class row of the normalised output. ViTModel also
    # has a tanh pooler, and the classification model does not use it.
    class_row = network.add_slice(hidden, start=(0, 0), shape=(1, width), stride=(1, 1))
    logits = g.add_linear(network, class_row.get_output(0), weights["classifier.weight"],
                          weights["classifier.bias"], dtype=work_np)

    logits.name = "logits"
    network.mark_output(logits)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("vit engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one ViT bundle."""
    if request.task not in _SUPPORTED_TASKS:
        raise ValueError("vit supports only task=" + ", ".join(_SUPPORTED_TASKS))
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("vit supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("vit does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("vit does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("vit does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("vit does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("vit does not support video_num_frames")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("vit does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("vit does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("vit does not support MTP speculative decoding")
    if request.max_sequence_length is not None and int(request.max_sequence_length) != 1:
        # The sequence is the patch grid plus the class token; it follows from
        # the image size and is not a free parameter.
        raise NotImplementedError("vit derives its sequence length from the image size")

    # fp32 only. fp16 is refused rather than built unqualified, because
    # measured on this checkpoint it cannot be gated. Against the reference a
    # correct fp32 engine is 1.0e-05 off, an all-fp16 one is 6.9e-03, and the
    # weakest defect the E2E gate catches - the tanh GELU - is 1.9e-02. An
    # fp16 tolerance would have to sit above its own noise and below that
    # break, which leaves under 3x of room. Taking the attention softmax in
    # fp32 does not recover it: measured, 6.9e-03 either way, so the loss is
    # in the matmuls themselves and not in a reduction done too narrowly.
    precision = str(request.precision).lower()
    if precision != "fp32":
        raise NotImplementedError(
            f"vit builds fp32 only; {precision} is not qualified for this family")

    model_dir = Path(request.model_dir)
    weights = _open_checkpoint(model_dir)
    prefix = "vit." if any(key.startswith("vit.") for key in weights) else ""
    _require_weights(weights, prefix)
    cfg = config_module.resolve(model_dir, weights)

    image_size = int(request.image_height or 0) or cfg["image_size"]
    if request.image_width is not None and int(request.image_width) != image_size:
        raise NotImplementedError("vit needs a square input size")
    if image_size != cfg["image_size"]:
        # The position table has one row per patch of the trained grid, so a
        # different input size needs the table interpolated, not just resized.
        raise NotImplementedError(
            f"vit input size is fixed at the checkpoint's {cfg['image_size']}")

    writer.set_header(family="vit", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", build_engine(
        weights, cfg, verbose=bool(request.verbose)))
    writer.add_json(
        "runtime.json",
        {
            "input_image_h": image_size,
            "input_image_w": image_size,
            "image_mean": cfg["image_mean"],
            "image_std": cfg["image_std"],
            "resample": cfg["resample"],
            "num_classes": cfg["num_classes"],
            "labels": cfg["labels"],
            "vocabulary_id": cfg["vocabulary_id"],
        },
    )
