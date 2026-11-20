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
from .quantization import calibrate_dinov2

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


_DUAL_GEMM_KNOB = "peep:match_dual_gemm"

# Work precision per build precision. fp16 (the default) keeps float32 engine
# I/O and float32 layer-norm statistics. Measured on the fp32 reference over
# 200 ImageNet validation images and 16 noise and flat images, the largest
# activation (the residual stream) is 520, 126x below the fp16 limit, and the
# layer norm squares it only after the float32 cast.
_WORK_DTYPES = {"fp32": np.float32, "fp16": np.float16}


def _builder(verbose, quant_ctx=None):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    config.builder_optimization_level = 3
    # TF32 caps a conv-and-matmul graph near 0.99998 cosine by default,
    # which hides real graph error behind a plausible-looking number.
    config.clear_flag(trt.BuilderFlag.TF32)
    if quant_ctx is not None:
        # Keep layer names so the plan inspector shows which kernels run.
        config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        if quant_ctx.has_fp8_weights and _DUAL_GEMM_KNOB in getattr(config, "all_build_routes", ""):
            # TensorRT's fused dual-GEMM kernel fails to compile with scalar-scale FP8.
            # The route is set only when this TensorRT build lists the knob.
            config.build_route = f"-{_DUAL_GEMM_KNOB}=off"
    return builder, network, config


def build_engine(weights, cfg, *, verbose=False, quant_ctx=None, dtype=np.float32) -> bytes:
    builder, network, config = _builder(verbose, quant_ctx)
    # The work precision of every layer: float32, or float16 for precision=fp16.
    # Layer-norm statistics are taken in float32 either way (graph.add_layer_norm).
    work_np = np.dtype(dtype).type
    width = cfg["hidden_size"]
    patch = cfg["patch_size"]
    grid = cfg["grid"]
    size = cfg["image_size"]

    # The engine I/O stays float32 at every precision, so the runtime and the
    # bundle contract do not change; only the network between them narrows.
    source = network.add_input("pixel_values", trt.float32, (1, 3, size, size))
    if work_np != np.float32:
        source = network.add_cast(source, trt.float16).get_output(0)

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

    hidden = encoder_builder.build_layers(network, hidden, weights, cfg, dtype=work_np,
                                          quant=quant_ctx)
    hidden = g.add_layer_norm(network, hidden, weights["layernorm.weight"],
                              weights["layernorm.bias"], cfg["eps"])
    if hidden.dtype != trt.float32:
        hidden = network.add_cast(hidden, trt.float32).get_output(0)

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
    if request.quantization not in {None, "none", "fp8"}:
        raise NotImplementedError(
            f"dinov2 supports quantization fp8 or none; got {request.quantization!r}")
    if request.fp32_layers:
        raise NotImplementedError("dinov2 does not support mixed-precision layers")
    if request.mtp_seq_len is not None:
        raise NotImplementedError("dinov2 does not support MTP speculative decoding")
    if request.max_sequence_length is not None:
        # The sequence is the patch grid plus the class token; it follows from
        # the image size and is not a free parameter.
        raise NotImplementedError(
            "dinov2 derives its sequence length from the image size")

    # fp16 (the default) or fp32. On dinov2-base the fp16 engine is 5.2x
    # (B200) and 4.6x (B300) faster than fp32, and kNN top-1 stays within 0.2
    # points of the reference. Its features are 0.35 to 1.16 off the
    # reference on the E2E image, where the weakest defect - the tanh GELU -
    # is 6.2e-02, so no feature tolerance can catch that defect in fp16; the
    # fp16 E2E case adds a check that does not depend on that margin
    # (test_e2e.py). A float32 residual stream or a float32 softmax was
    # measured and recovers nothing: the error is in the fp16 matmuls.
    precision = str(request.precision).lower()
    if precision not in _WORK_DTYPES:
        raise NotImplementedError(
            f"dinov2 builds fp16 or fp32; {precision} is not qualified for this family")
    if request.quantization == "fp8" and precision != "fp32":
        # The FP8 scales, the graph and its measurements are float32 (fp32 Dequantize).
        raise NotImplementedError(
            f"dinov2 fp8 builds fp32 only; pass --precision fp32 (got {precision})")

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

    # fp8: the projections are quantized from the original checkpoint, one at a time, with
    # the stored activation scales. FP8 weights are never stored, and the engine still
    # works in float32 around the quantized projections.
    quant_ctx = None
    if request.quantization == "fp8":
        quant_ctx = calibrate_dinov2(weights, cfg, "fp8")

    writer.set_header(family="dinov2", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", build_engine(
        weights, cfg, verbose=bool(request.verbose), quant_ctx=quant_ctx,
        dtype=_WORK_DTYPES[precision]))
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
