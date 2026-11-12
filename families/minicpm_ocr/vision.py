# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniCPM-OCR vision engine: SigLIP-style tower plus the 2x2 merger, one image piece per call.

The checkpoint runs in the "4x" token mode, where the reference skips the tower's inserted window merger: all
encoder layers attend over the whole piece, then ``post_layernorm`` and ``merger.mlp[0]`` turn every 2x2 patch window
into one decoder-width row. The host supplies the patches grouped by merge window (and the matching position ids),
so the window merge is a reshape here.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .checkpoint import Checkpoint, ModelConfig
from .graph import Graph

PATCH_INPUT = "patches"
POSITION_INPUT = "position_ids"
FEATURE_OUTPUT = "features"


def _linear(checkpoint: Checkpoint, name: str, out_width: int, in_width: int):
    weight = checkpoint.tensor(f"{name}.weight", (out_width, in_width))
    bias = checkpoint.tensor(f"{name}.bias", (out_width,))
    return np.ascontiguousarray(weight.T), bias


def build_vision_engine(checkpoint: Checkpoint, config: ModelConfig, precision: str, max_patches: int,
                        verbose: bool) -> bytes:
    vision = config.vision
    hidden, heads, head_dim = vision.hidden_size, vision.num_heads, vision.head_dim
    patch_values = 3 * vision.patch_size * vision.patch_size
    merge = config.merge_kernel[0] * config.merge_kernel[1]
    text_hidden = config.text.hidden_size
    if max_patches < merge or max_patches % merge:
        raise ValueError("minicpm_ocr: the vision patch capacity must be a positive multiple of the merge window")

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    build_config = builder.create_builder_config()
    build_config.builder_optimization_level = 3
    if precision == "fp32":
        build_config.clear_flag(trt.BuilderFlag.TF32)
    else:
        # TF32 multiplies the linear layers' FP16 operands exactly; it may round the attention's FP32 scaled queries
        # and probabilities to a 10-bit mantissa, as FP16 attention would.
        build_config.set_flag(trt.BuilderFlag.TF32)
    g = Graph(network, trt.float16 if precision == "fp16" else trt.float32)

    patches = network.add_input(PATCH_INPUT, trt.float32, (-1, patch_values))
    positions = network.add_input(POSITION_INPUT, trt.int32, (-1,))
    profile = builder.create_optimization_profile()
    optimum = min(max_patches, 1024)
    profile.set_shape(PATCH_INPUT, (merge, patch_values), (optimum, patch_values), (max_patches, patch_values))
    profile.set_shape(POSITION_INPUT, (merge,), (optimum,), (max_patches,))
    build_config.add_optimization_profile(profile)

    prefix = "model.vision_tower"
    conv = checkpoint.tensor(f"{prefix}.embeddings.patch_embedding.weight",
                             (hidden, 3, vision.patch_size, vision.patch_size))
    x = g.linear(g.to_work(patches), np.ascontiguousarray(conv.reshape(hidden, patch_values).T),
                 checkpoint.tensor(f"{prefix}.embeddings.patch_embedding.bias", (hidden,)))
    table = checkpoint.tensor(f"{prefix}.embeddings.position_embedding.weight",
                              (vision.position_grid ** 2, hidden))
    x = g.add(x, network.add_gather(g.constant(table, g.np_work), positions, 0).get_output(0))

    eps = vision.layer_norm_eps
    for index in range(vision.num_layers):
        layer = f"{prefix}.encoder.layers.{index}"
        normed = g.layer_norm(x, checkpoint.tensor(f"{layer}.layer_norm1.weight", (hidden,)),
                              checkpoint.tensor(f"{layer}.layer_norm1.bias", (hidden,)), eps)
        projected = []
        for name in ("q_proj", "k_proj", "v_proj"):
            rows = g.linear(normed, *_linear(checkpoint, f"{layer}.self_attn.{name}", hidden, hidden))
            # [N, heads * dim] -> [1, heads, N, dim]
            projected.append(g.reshape(g.transpose(g.reshape(rows, (-1, heads, head_dim)), (1, 0, 2)),
                                       (1, heads, -1, head_dim)))
        context = g.attention(*projected, scale=head_dim ** -0.5)
        context = g.reshape(g.transpose(g.reshape(context, (heads, -1, head_dim)), (1, 0, 2)), (-1, hidden))
        x = g.add(x, g.linear(context, *_linear(checkpoint, f"{layer}.self_attn.out_proj", hidden, hidden)))
        normed = g.layer_norm(x, checkpoint.tensor(f"{layer}.layer_norm2.weight", (hidden,)),
                              checkpoint.tensor(f"{layer}.layer_norm2.bias", (hidden,)), eps)
        mlp = g.linear(normed, *_linear(checkpoint, f"{layer}.mlp.fc1", vision.intermediate_size, hidden))
        mlp = g.gelu(mlp, approximate=True)
        x = g.add(x, g.linear(mlp, *_linear(checkpoint, f"{layer}.mlp.fc2", hidden, vision.intermediate_size)))

    x = g.layer_norm(x, checkpoint.tensor(f"{prefix}.post_layernorm.weight", (hidden,)),
                     checkpoint.tensor(f"{prefix}.post_layernorm.bias", (hidden,)), eps)
    merged_width = hidden * merge
    x = g.reshape(x, (-1, merged_width))
    mlp = "model.merger.mlp.0"
    x = g.layer_norm(x, checkpoint.tensor(f"{mlp}.pre_norm.weight", (merged_width,)),
                     checkpoint.tensor(f"{mlp}.pre_norm.bias", (merged_width,)), 1e-6)
    x = g.linear(x, *_linear(checkpoint, f"{mlp}.linear_1", merged_width, merged_width))
    x = g.gelu(x, approximate=False)
    x = g.linear(x, *_linear(checkpoint, f"{mlp}.linear_2", text_hidden, merged_width))
    x.name = FEATURE_OUTPUT
    network.mark_output(x)

    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("minicpm_ocr: TensorRT failed to build the vision engine")
    return bytes(plan)
