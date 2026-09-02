# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Qwen3-VL vision tower shared by MiniMax-H3 FL2VA and Ref2VA.

The engine evaluates one image or one temporal video-patch block.  Ref2VA video
conditioning invokes it once per temporal block and concatenates outputs, which
is exactly Qwen3-VL's packed ``cu_seqlens`` attention boundary.  Interpolation
indices/weights and 2-D rotary positions are runtime inputs because reference
images have dynamic aspect and Ref2VA images use a larger 2048-short-edge grid.
The learned position table and every vision weight remain inside this one plan.
"""

from __future__ import annotations

import gc
import math
import sys
from pathlib import Path

import numpy as np

from tensorrt_model_connect import trt_compat

from . import graph_ops as op
from .fl2va_contract import (
    QWEN_VISION_HIDDEN_SIZE,
    QWEN_VISION_MERGE_SIZE,
    QWEN_VISION_PATCH_WIDTH,
    VisionEncoderProfile,
    vision_encoder_abi,
)


trt = trt_compat.get_trt()

DEPTH = 27
NUM_HEADS = 16
HEAD_DIM = QWEN_VISION_HIDDEN_SIZE // NUM_HEADS
INTERMEDIATE_SIZE = 4304
POSITION_TABLE_SIDE = 48
POSITION_TABLE_ROWS = POSITION_TABLE_SIDE**2
MERGE_UNIT = QWEN_VISION_MERGE_SIZE**2
MERGED_HIDDEN_SIZE = QWEN_VISION_HIDDEN_SIZE * MERGE_UNIT
DEEPSTACK_VISUAL_INDEXES = (8, 16, 24)
NORM_EPS = 1.0e-6
VISION_ENCODER_DEFAULT_WORKSPACE_BYTES = 32 << 30
ROPE_INV_FREQ_BITS = (
    0x3F800000,
    0x3F1977CC,
    0x3EB800D6,
    0x3E5C9D35,
    0x3E044133,
    0x3D9E91B6,
    0x3D3E1E95,
    0x3CE3F280,
    0x3C88A69B,
    0x3C23D70A,
    0x3BC47060,
    0x3B6B8631,
    0x3B0D3169,
    0x3AA94938,
    0x3A4AF7F3,
    0x39F35A5C,
    0x3991E2E1,
    0x392EE9BF,
)


def checkpoint_keys() -> tuple[str, ...]:
    names = ["model.visual.patch_embed.proj.weight", "model.visual.patch_embed.proj.bias"]
    names.append("model.visual.pos_embed.weight")
    for index in range(DEPTH):
        prefix = f"model.visual.blocks.{index}"
        names.extend(
            [
                f"{prefix}.norm1.weight",
                f"{prefix}.norm1.bias",
                f"{prefix}.attn.qkv.weight",
                f"{prefix}.attn.qkv.bias",
                f"{prefix}.attn.proj.weight",
                f"{prefix}.attn.proj.bias",
                f"{prefix}.norm2.weight",
                f"{prefix}.norm2.bias",
                f"{prefix}.mlp.linear_fc1.weight",
                f"{prefix}.mlp.linear_fc1.bias",
                f"{prefix}.mlp.linear_fc2.weight",
                f"{prefix}.mlp.linear_fc2.bias",
            ]
        )
    for prefix in [
        "model.visual.merger",
        *(f"model.visual.deepstack_merger_list.{index}" for index in range(3)),
    ]:
        names.extend(
            [
                f"{prefix}.norm.weight",
                f"{prefix}.norm.bias",
                f"{prefix}.linear_fc1.weight",
                f"{prefix}.linear_fc1.bias",
                f"{prefix}.linear_fc2.weight",
                f"{prefix}.linear_fc2.bias",
            ]
        )
    return tuple(names)


def _layer_norm(network, hidden, weights, prefix: str, width: int):
    rank = len(tuple(hidden.shape))
    shape = (1,) * (rank - 1) + (width,)
    gamma = op.weight_constant(network, np.asarray(weights[f"{prefix}.weight"]).reshape(shape))
    beta = op.weight_constant(network, np.asarray(weights[f"{prefix}.bias"]).reshape(shape))
    gamma = op.cast(network, gamma, hidden.dtype)
    beta = op.cast(network, beta, hidden.dtype)
    layer = network.add_normalization_v2(hidden, gamma, beta, 1 << (rank - 1))
    if layer is None:
        raise RuntimeError(f"TensorRT rejected MiniMax-H3 Qwen vision LayerNorm {prefix}")
    layer.name = prefix
    layer.epsilon = NORM_EPS
    return layer.get_output(0)


def _gelu(network, hidden, activation_type):
    layer = network.add_activation(hidden, activation_type)
    if layer is None:
        raise RuntimeError("TensorRT rejected MiniMax-H3 Qwen vision GELU")
    return layer.get_output(0)


def _interpolated_position_embeddings(network, weights, indices, interp_weights):
    table = op.weight_constant(network, weights["model.visual.pos_embed.weight"])
    table = op.cast(network, table, trt.bfloat16)
    gathered = network.add_gather(table, indices, 0)
    if gathered is None:
        raise RuntimeError("TensorRT rejected MiniMax-H3 Qwen position-table gather")
    # The learned table is BF16, but Transformers multiplies it by FP32
    # bilinear weights and performs the four-way sum in FP32 before rounding
    # the completed positional embedding back to the patch-embedding dtype.
    gathered_value = op.cast(network, gathered.get_output(0), trt.float32)
    coefficients = network.add_shuffle(interp_weights)
    coefficients.reshape_dims = (-1, 4, 1)
    coefficients_value = op.cast(network, coefficients.get_output(0), trt.float32)
    weighted = network.add_elementwise(
        gathered_value, coefficients_value, trt.ElementWiseOperation.PROD
    )
    if weighted is None:
        raise RuntimeError("TensorRT rejected MiniMax-H3 Qwen position interpolation")
    reduced = network.add_reduce(weighted.get_output(0), trt.ReduceOperation.SUM, 1 << 1, False)
    if reduced is None:
        raise RuntimeError("TensorRT rejected MiniMax-H3 Qwen position interpolation reduce")
    return reduced.get_output(0)


def _vision_rope_cache(network, position_ids):
    positions = op.cast(network, position_ids, trt.float32)
    # Preserve Qwen3VLVisionRotaryEmbedding's Torch-created FP32 buffer
    # without making Torch a builder dependency.
    inverse = np.asarray(ROPE_INV_FREQ_BITS, dtype=np.uint32).view(np.float32)
    inverse = op.constant(network, inverse.reshape(1, -1))
    axes = []
    for axis in range(2):
        coordinate = op.dynamic_slice(network, positions, (0, axis), (None, 1))
        axes.append(
            network.add_elementwise(coordinate, inverse, trt.ElementWiseOperation.PROD).get_output(
                0
            )
        )
    frequency = network.add_concatenation(axes)
    frequency.axis = 1
    cos = network.add_unary(frequency.get_output(0), trt.UnaryOperation.COS).get_output(0)
    sin = network.add_unary(frequency.get_output(0), trt.UnaryOperation.SIN).get_output(0)
    # Qwen3-VL explicitly performs vision rotary embedding in FP32 and rounds
    # q/k back to their source dtype only after both products are added.
    return cos, sin


def _vision_partial_rope(network, tensor, cos_half, sin_half):
    """Qwen3-VL's FP32 vision rotate-half operation."""

    value = op.rows_to_heads(network, tensor, 1, NUM_HEADS, HEAD_DIM)
    source_dtype = value.dtype
    value = op.cast(network, value, trt.float32)
    first = op.dynamic_slice(network, value, (0, 0, 0, 0), (1, NUM_HEADS, None, HEAD_DIM // 2))
    second = op.dynamic_slice(
        network,
        value,
        (0, 0, 0, HEAD_DIM // 2),
        (1, NUM_HEADS, None, HEAD_DIM // 2),
    )
    negative_second = network.add_unary(second, trt.UnaryOperation.NEG).get_output(0)
    rotated_half = network.add_concatenation((negative_second, first))
    rotated_half.axis = 3

    def duplicate(table):
        reshape = network.add_shuffle(op.cast(network, table, trt.float32))
        reshape.reshape_dims = (1, 1, -1, HEAD_DIM // 2)
        result = network.add_concatenation((reshape.get_output(0), reshape.get_output(0)))
        result.axis = 3
        return result.get_output(0)

    left = network.add_elementwise(
        value, duplicate(cos_half), trt.ElementWiseOperation.PROD
    ).get_output(0)
    right = network.add_elementwise(
        rotated_half.get_output(0), duplicate(sin_half), trt.ElementWiseOperation.PROD
    ).get_output(0)
    result = network.add_elementwise(left, right, trt.ElementWiseOperation.SUM).get_output(0)
    result = op.cast(network, result, source_dtype)
    return op.heads_to_rows(network, result, 1, QWEN_VISION_HIDDEN_SIZE)


def _attention(network, hidden, weights, prefix: str, cos, sin):
    projected = op.linear(
        network,
        hidden,
        weights[f"{prefix}.qkv.weight"],
        weights[f"{prefix}.qkv.bias"],
    )
    query, key, value = tuple(
        op.dynamic_slice(
            network,
            projected,
            (0, index * QWEN_VISION_HIDDEN_SIZE),
            (None, QWEN_VISION_HIDDEN_SIZE),
        )
        for index in range(3)
    )
    query = _vision_partial_rope(network, query, cos, sin)
    key = _vision_partial_rope(network, key, cos, sin)
    query_heads = op.rows_to_heads(network, query, 1, NUM_HEADS, HEAD_DIM)
    key_heads = op.rows_to_heads(network, key, 1, NUM_HEADS, HEAD_DIM)
    value_heads = op.rows_to_heads(network, value, 1, NUM_HEADS, HEAD_DIM)
    scale = op.cast(
        network,
        op.constant(network, np.full((1, 1, 1, 1), 1.0 / math.sqrt(HEAD_DIM), np.float32)),
        query_heads.dtype,
    )
    query_heads = network.add_elementwise(
        query_heads, scale, trt.ElementWiseOperation.PROD
    ).get_output(0)
    layer = network.add_attention(
        query_heads,
        key_heads,
        value_heads,
        trt.AttentionNormalizationOp.SOFTMAX,
        False,
    )
    if layer is None:
        raise RuntimeError(f"TensorRT rejected MiniMax-H3 Qwen vision attention {prefix}")
    layer.name = f"{prefix}.native_attention"
    layer.metadata = f"trtmc.native_op=IAttention;source={layer.name}"
    layer.get_output(0).name = f"{layer.name}.output"
    layer.decomposable = False
    rows = op.heads_to_rows(network, layer.get_output(0), 1, QWEN_VISION_HIDDEN_SIZE)
    return op.linear(
        network,
        rows,
        weights[f"{prefix}.proj.weight"],
        weights[f"{prefix}.proj.bias"],
    )


def _mlp(network, hidden, weights, prefix: str):
    hidden = op.linear(
        network,
        hidden,
        weights[f"{prefix}.linear_fc1.weight"],
        weights[f"{prefix}.linear_fc1.bias"],
    )
    hidden = _gelu(network, hidden, trt.ActivationType.GELU_TANH)
    return op.linear(
        network,
        hidden,
        weights[f"{prefix}.linear_fc2.weight"],
        weights[f"{prefix}.linear_fc2.bias"],
    )


def _merge(network, hidden, weights, prefix: str, *, postshuffle_norm: bool):
    if postshuffle_norm:
        reshape = network.add_shuffle(hidden)
        reshape.reshape_dims = (-1, MERGED_HIDDEN_SIZE)
        merged = _layer_norm(
            network, reshape.get_output(0), weights, f"{prefix}.norm", MERGED_HIDDEN_SIZE
        )
    else:
        normalized = _layer_norm(
            network, hidden, weights, f"{prefix}.norm", QWEN_VISION_HIDDEN_SIZE
        )
        reshape = network.add_shuffle(normalized)
        reshape.reshape_dims = (-1, MERGED_HIDDEN_SIZE)
        merged = reshape.get_output(0)
    merged = op.linear(
        network,
        merged,
        weights[f"{prefix}.linear_fc1.weight"],
        weights[f"{prefix}.linear_fc1.bias"],
    )
    # Qwen3-VL patch mergers use nn.GELU(approximate="none"), unlike the
    # tanh-approximate GELU in each vision MLP.
    merged = _gelu(network, merged, trt.ActivationType.GELU_ERF)
    return op.linear(
        network,
        merged,
        weights[f"{prefix}.linear_fc2.weight"],
        weights[f"{prefix}.linear_fc2.bias"],
    )


@op.cleanup_failed_build
def build_multimodal_vision_encoder_engine(
    weights: dict[str, np.ndarray],
    profile: VisionEncoderProfile = VisionEncoderProfile(),
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    weight_streaming: bool = False,
    output_path: str | Path | None = None,
) -> bytes | dict[str, int | str]:
    """Build one shared dynamic-aspect Qwen3-VL vision tower."""

    profile.validate()
    expected_keys = set(checkpoint_keys())
    missing = sorted(expected_keys - set(weights))
    unexpected = sorted(set(weights) - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "MiniMax-H3 multimodal vision checkpoint partition mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    op.configure_builder(config, weight_streaming=weight_streaming)
    op.configure_workspace(
        config,
        workspace_bytes,
        default_bytes=VISION_ENCODER_DEFAULT_WORKSPACE_BYTES,
    )

    abi = vision_encoder_abi(profile)
    inputs = {
        binding.name: network.add_input(
            binding.name,
            {
                "float32": trt.float32,
                "int32": trt.int32,
            }[binding.dtype],
            (-1, binding.max_shape[1]),
        )
        for binding in abi.inputs
    }
    optimization = builder.create_optimization_profile()
    for binding in abi.inputs:
        optimization.set_shape(
            binding.name, binding.min_shape, binding.opt_shape, binding.max_shape
        )
    config.add_optimization_profile(optimization)

    patch_weight = np.asarray(weights["model.visual.patch_embed.proj.weight"]).reshape(
        QWEN_VISION_HIDDEN_SIZE, QWEN_VISION_PATCH_WIDTH
    )
    hidden = op.linear(
        network,
        inputs["pixel_values"],
        patch_weight,
        weights["model.visual.patch_embed.proj.bias"],
    )
    position = _interpolated_position_embeddings(
        network, weights, inputs["interp_indices"], inputs["interp_weights"]
    )
    position = op.cast(network, position, hidden.dtype)
    hidden = network.add_elementwise(hidden, position, trt.ElementWiseOperation.SUM).get_output(0)
    cos, sin = _vision_rope_cache(network, inputs["vision_position_ids"])

    deepstack = []
    for index in range(DEPTH):
        prefix = f"model.visual.blocks.{index}"
        normalized = _layer_norm(
            network, hidden, weights, f"{prefix}.norm1", QWEN_VISION_HIDDEN_SIZE
        )
        update = _attention(network, normalized, weights, f"{prefix}.attn", cos, sin)
        update = op.cast(network, update, hidden.dtype)
        hidden = network.add_elementwise(hidden, update, trt.ElementWiseOperation.SUM).get_output(0)

        normalized = _layer_norm(
            network, hidden, weights, f"{prefix}.norm2", QWEN_VISION_HIDDEN_SIZE
        )
        update = _mlp(network, normalized, weights, f"{prefix}.mlp")
        update = op.cast(network, update, hidden.dtype)
        hidden = network.add_elementwise(hidden, update, trt.ElementWiseOperation.SUM).get_output(0)

        if index in DEEPSTACK_VISUAL_INDEXES:
            merger_index = DEEPSTACK_VISUAL_INDEXES.index(index)
            deepstack.append(
                _merge(
                    network,
                    hidden,
                    weights,
                    f"model.visual.deepstack_merger_list.{merger_index}",
                    postshuffle_norm=True,
                )
            )

    main = _merge(
        network,
        hidden,
        weights,
        "model.visual.merger",
        postshuffle_norm=False,
    )
    for tensor, binding in zip((main, *deepstack), abi.outputs):
        output = op.cast(network, tensor, trt.float32)
        output.name = binding.name
        network.mark_output(output)

    op.validate_native_network(
        network, expected_attentions=DEPTH, label="multimodal vision encoder"
    )
    print(
        "[minimax-h3] building shared Qwen3-VL vision tower: "
        f"layers={DEPTH}, patches={profile.min_patches}..{profile.max_patches}, "
        f"merged={profile.min_patches // MERGE_UNIT}..{profile.max_patches // MERGE_UNIT}",
        file=sys.stderr,
    )
    plan = None
    record = None
    try:
        if output_path is None:
            plan = builder.build_serialized_network(network, config)
        else:
            record = trt_compat.build_serialized_network_to_file(
                builder, network, config, output_path
            )
    finally:
        op.release_weight_buffers(network)
        if consume_weights:
            weights.clear()
    if output_path is None and plan is None:
        raise RuntimeError("TensorRT failed to build MiniMax-H3 multimodal vision encoder")
    del network, config, builder
    gc.collect()
    return record if record is not None else bytes(plan)
