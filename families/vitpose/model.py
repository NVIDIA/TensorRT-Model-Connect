# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a ViTPose bundle: one float32 engine from the crop's pixel values to the keypoint heatmaps.

`VitPoseForPoseEstimation` with the simple decoder: the patch convolution (padding 2) plus the position table without
its class row (there is no class token, and row 0 is added to every patch), the pre-norm ViT layers, the final layer
norm, then ReLU, bilinear upsampling and a 3 x 3 convolution to one heatmap per keypoint. Cropping, normalization
and the keypoint decoding run in the native runtime (runtime/image_ops.cpp).
"""

from __future__ import annotations

from pathlib import Path

from . import config as config_module
from . import weights as weights_module
from .config import VitPoseConfig

_TASK = "image_text_to_points"
INPUT = "pixel_values"
OUTPUT = "heatmaps"


def _layers(network, tokens, weights, cfg: VitPoseConfig):
    from . import graph as g

    for index in range(cfg.num_layers):
        prefix = f"backbone.encoder.layer.{index}"

        def tensor(name: str):
            return weights[f"{prefix}.{name}"]

        normed = g.layer_norm(
            network,
            tokens,
            tensor("layernorm_before.weight"),
            tensor("layernorm_before.bias"),
            cfg.layer_norm_eps,
        )
        query, key, value = (
            g.linear(
                network,
                normed,
                tensor(f"attention.attention.{name}.weight"),
                tensor(f"attention.attention.{name}.bias"),
            )
            for name in ("query", "key", "value")
        )
        context = g.attention(
            network, query, key, value, heads=cfg.num_heads, head_dim=cfg.head_dim
        )
        attended = g.linear(
            network,
            context,
            tensor("attention.output.dense.weight"),
            tensor("attention.output.dense.bias"),
        )
        tokens = g.add(network, attended, tokens)
        normed = g.layer_norm(
            network,
            tokens,
            tensor("layernorm_after.weight"),
            tensor("layernorm_after.bias"),
            cfg.layer_norm_eps,
        )
        inner = g.gelu(
            network, g.linear(network, normed, tensor("mlp.fc1.weight"), tensor("mlp.fc1.bias"))
        )
        tokens = g.add(
            network,
            g.linear(network, inner, tensor("mlp.fc2.weight"), tensor("mlp.fc2.bias")),
            tokens,
        )
    return tokens


def build_engine(weights, cfg: VitPoseConfig, *, verbose: bool = False) -> bytes:
    import tensorrt as trt

    from . import graph as g

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    pixels = network.add_input(INPUT, trt.float32, (1, 3, cfg.image_height, cfg.image_width))

    rows, columns = cfg.grid
    width = cfg.hidden_size
    patches = g.conv2d(
        network,
        pixels,
        weights["backbone.embeddings.patch_embeddings.projection.weight"],
        weights["backbone.embeddings.patch_embeddings.projection.bias"],
        stride=(cfg.patch_height, cfg.patch_width),
        padding=(config_module.PATCH_PADDING, config_module.PATCH_PADDING),
    )
    flat = network.add_shuffle(patches)
    flat.reshape_dims = (width, rows * columns)
    flat.second_transpose = (1, 0)
    # embeddings + position_embeddings[:, 1:] + position_embeddings[:, :1], in that order.
    table = weights["backbone.embeddings.position_embeddings"].reshape(rows * columns + 1, width)
    tokens = g.add(network, flat.get_output(0), g.constant(network, table[1:]))
    tokens = g.add(network, tokens, g.constant(network, table[:1]))

    tokens = _layers(network, tokens, weights, cfg)
    tokens = g.layer_norm(
        network,
        tokens,
        weights["backbone.layernorm.weight"],
        weights["backbone.layernorm.bias"],
        cfg.layer_norm_eps,
    )
    grid = network.add_shuffle(tokens)
    grid.first_transpose = (1, 0)
    grid.reshape_dims = (1, width, rows, columns)
    features = network.add_activation(grid.get_output(0), trt.ActivationType.RELU).get_output(0)
    features = g.upsample_bilinear(network, features, cfg.scale_factor)
    heatmaps = g.conv2d(
        network,
        features,
        weights["head.conv.weight"],
        weights["head.conv.bias"],
        stride=(1, 1),
        padding=(1, 1),
    )
    heatmaps.name = OUTPUT
    network.mark_output(heatmaps)
    expected = (1, cfg.num_keypoints, *cfg.heatmap_size)
    if tuple(heatmaps.shape) != expected or heatmaps.dtype != trt.float32:
        raise RuntimeError(
            f"vitpose heatmaps are {tuple(heatmaps.shape)} {heatmaps.dtype}, expected {expected} float32"
        )

    builder_config = builder.create_builder_config()
    builder_config.builder_optimization_level = 3
    # The checkpoint and the reference are float32; TF32 would round every product to 10 mantissa bits.
    builder_config.clear_flag(trt.BuilderFlag.TF32)
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("vitpose engine build failed")
    return bytes(plan)


def build(request, writer) -> None:
    """Build one ViTPose bundle for `image_text_to_points`; every build option it does not consume is rejected."""

    if request.task != _TASK:
        raise ValueError(f"vitpose supports only task={_TASK}")
    if request.backend != "trt":
        raise ValueError("vitpose supports only backend=trt")
    precision = str(request.precision).lower()
    if precision != "fp32":
        raise NotImplementedError(
            f"vitpose builds the float32 checkpoint at fp32 only, got precision={precision}"
        )
    if request.quantization not in {None, "none"}:
        raise NotImplementedError("vitpose does not support quantization")
    if request.fp32_layers:
        raise NotImplementedError("vitpose does not support mixed-precision layers")
    if request.max_batch_size != 1:
        raise NotImplementedError("vitpose crops one person per run (max_batch_size=1)")
    if request.tensor_parallel_size != 1:
        raise NotImplementedError("vitpose does not support tensor parallelism")
    if request.context_parallel_size != 1:
        raise NotImplementedError("vitpose does not support context parallelism")
    if request.dynamic_kv_cache:
        raise NotImplementedError("vitpose does not support dynamic_kv_cache")
    if request.video_num_frames is not None:
        raise NotImplementedError("vitpose does not support video_num_frames")
    if request.max_sequence_length is not None:
        raise NotImplementedError(
            "vitpose takes its token count from the crop size; max_sequence_length is not used"
        )
    if request.mtp_seq_len is not None:
        raise NotImplementedError("vitpose does not support MTP speculative decoding")

    model_dir = Path(request.model_dir)
    cfg = config_module.resolve(model_dir, weights_module.read_header(model_dir))
    for value, expected, name in (
        (request.image_height, cfg.image_height, "image_height"),
        (request.image_width, cfg.image_width, "image_width"),
    ):
        if value is not None and value != expected:
            raise NotImplementedError(
                f"vitpose crops at its processor size {cfg.image_height}x{cfg.image_width}; {name}={value} is not supported"
            )
    plan = build_engine(weights_module.load(model_dir, cfg), cfg, verbose=bool(request.verbose))
    writer.set_header(family="vitpose", task=request.task, backend=request.backend)
    writer.add_bytes("engine.plan", plan)
    writer.add_json("runtime.json", config_module.runtime_metadata(cfg))
