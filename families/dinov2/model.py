# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a DINOv2 bundle.

One engine. DINOv2 is a plain ViT whose only outputs are the final hidden
rows, so a single forward pass serves both halves of the task: the patch
rows are the token features and the class row is the pooled vector. The
runtime splits them; it does not run the model twice.

The engine returns the rows unnormalised, because that is what the reference
returns. Nothing here applies an L2 norm, and the runtime says so rather
than quietly normalising and reporting a unit vector.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorrt as trt

from . import config as config_module, encoder_builder, graph as g, position

_SUPPORTED_TASKS = ("image_to_token_and_pooled_features",)


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


def _require_weights(weights: dict) -> None:
    required = (
        "embeddings.cls_token",
        "embeddings.position_embeddings",
        "embeddings.patch_embeddings.projection.weight",
        "embeddings.patch_embeddings.projection.bias",
        "layernorm.weight",
        # LayerScale is what makes this not a generic ViT.
        "encoder.layer.0.layer_scale1.lambda1",
        "encoder.layer.0.layer_scale2.lambda1",
    )
    missing = [key for key in required if key not in weights]
    if missing:
        raise KeyError("dinov2 checkpoint is missing expected tensors: " + ", ".join(missing))


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
    width = cfg["hidden_size"]
    patch = cfg["patch_size"]
    grid = cfg["grid"]
    size = cfg["image_size"]

    source = network.add_input("pixel_values", trt.float32, (1, 3, size, size))

    patches = g.add_conv2d(network, source,
                           weights["embeddings.patch_embeddings.projection.weight"],
                           weights["embeddings.patch_embeddings.projection.bias"],
                           stride=(patch, patch), dtype=work_np)
    flat = network.add_shuffle(patches)
    flat.reshape_dims = (width, grid * grid)
    flat.second_transpose = (1, 0)

    # The class token is prepended, so row 0 of the table belongs to it.
    class_token = g.add_constant(network, (1, width),
                                 np.asarray(weights["embeddings.cls_token"]).reshape(1, width),
                                 dtype=work_np)
    concat = network.add_concatenation([class_token, flat.get_output(0)])
    concat.axis = 0
    hidden = concat.get_output(0)

    # Resized once, here, rather than on every forward pass: the engine's
    # input size is fixed, so the reference's runtime interpolation is a
    # build-time weight transform. See position.py for what it is worth.
    table = position.interpolate_position_table(
        weights["embeddings.position_embeddings"], cfg["trained_grid"], grid)
    hidden = g.add_sum(network, hidden,
                       g.add_constant(network, (grid * grid + 1, width), table, dtype=work_np))

    hidden = encoder_builder.build_layers(network, hidden, weights, cfg, dtype=work_np)
    hidden = g.add_layer_norm(network, hidden, weights["layernorm.weight"],
                              weights["layernorm.bias"], cfg["eps"])

    hidden.name = "last_hidden_state"
    network.mark_output(hidden)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("dinov2 engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one DINOv2 bundle."""
    if request.task not in _SUPPORTED_TASKS:
        raise ValueError("dinov2 supports only task=" + ", ".join(_SUPPORTED_TASKS))
    if request.backend not in {"trt", "trt_rtx"}:
        raise ValueError("dinov2 supports only backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("dinov2 does not support dynamic_kv_cache")
    if request.max_batch_size != 1:
        raise NotImplementedError("dinov2 does not support max_batch_size")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("dinov2 does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("dinov2 does not support context parallelism")
    if request.video_num_frames is not None:
        raise NotImplementedError("dinov2 does not support video_num_frames")
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("dinov2 does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("dinov2 does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("dinov2 does not support MTP speculative decoding")
    if request.max_sequence_length is not None:
        # The sequence is the patch grid plus the class token; it follows from
        # the image size and is not a free parameter.
        raise NotImplementedError(
            "dinov2 derives its sequence length from the image size")

    # fp32 only, and this family is the clearest case for refusing fp16.
    # Against the reference a correct fp32 engine is 4.5e-04 off; an all-fp16
    # one is about 1.0, on features whose own peak is 25.6. The weakest
    # defect the E2E gate catches - the tanh GELU - is 6.2e-02, so fp16's
    # error is sixteen times larger than the break it would have to catch: an
    # fp16 gate would pass a broken build. Cosine stays above 0.9999 through
    # all of it, which is exactly why cosine is not the gate. Taking the
    # attention softmax in fp32 was measured and moves nothing.
    precision = str(request.precision).lower()
    if precision != "fp32":
        raise NotImplementedError(
            f"dinov2 builds fp32 only; {precision} is not qualified for this family")

    model_dir = Path(request.model_dir)
    weights = _open_checkpoint(model_dir)
    _require_weights(weights)
    cfg = config_module.resolve(model_dir, weights)

    image_size = int(request.image_height or 0) or cfg["image_size"]
    if request.image_width is not None and int(request.image_width) != image_size:
        raise NotImplementedError("dinov2 needs a square input size")
    if image_size != cfg["image_size"]:
        # Another size is buildable - the table would simply be resized to a
        # different grid - but no other size has been measured, and the
        # preprocessing that feeds it is sized for this one.
        raise NotImplementedError(
            f"dinov2 input size is fixed at the preprocessor's {cfg['image_size']}")

    writer.set_header(family="dinov2", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", build_engine(
        weights, cfg, verbose=bool(request.verbose)))
    writer.add_json(
        "runtime.json",
        {
            "input_image_h": image_size,
            "input_image_w": image_size,
            "shortest_edge": cfg["shortest_edge"],
            "image_mean": cfg["image_mean"],
            "image_std": cfg["image_std"],
            "resample": cfg["resample"],
            "patch_size": cfg["patch_size"],
            "grid": cfg["grid"],
            "hidden_size": cfg["hidden_size"],
        },
    )
