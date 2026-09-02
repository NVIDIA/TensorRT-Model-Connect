# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TensorRT batched tile decoder for the MiniMax-H3 video VAE."""

from __future__ import annotations

import gc
import math
import sys
from pathlib import Path

import numpy as np

from tensorrt_model_connect import trt_compat

from . import graph_ops as op
from .config import VAE_TILE_DECODER_DEFAULT_WORKSPACE_BYTES


trt = trt_compat.get_trt()

MIN_BATCH = 16
OPT_BATCH = 28
MAX_BATCH = 33
CHANNELS = 24
FRAMES = 7
HEIGHT = 16
WIDTH = 16
TOKENS = FRAMES * HEIGHT * WIDTH
REGISTER_TOKENS = 4
SEQUENCE = TOKENS + REGISTER_TOKENS + 1
DIM = 2048
LAYERS = 36
HEADS = 32
HEAD_DIM = 64
FFN_DIM = 8192
ROTARY_DIM = 48
NORM_EPS = 1.0e-5
PATCH_T = 4
PATCH = 16
OUT_CHANNELS = 3


def checkpoint_keys() -> tuple[str, ...]:
    names = [
        "post_quant_conv.weight",
        "post_quant_conv.bias",
        "decoder.proj_in.weight",
        "decoder.proj_in.bias",
        "decoder.register_tokens",
        "decoder.norm_out.weight",
        "decoder.norm_out.bias",
        "decoder.proj_out.weight",
        "decoder.proj_out.bias",
    ]
    for index in range(LAYERS):
        prefix = f"decoder.transformer_blocks.{index}"
        names.extend(
            [
                f"{prefix}.norm1.weight",
                f"{prefix}.norm2.weight",
                f"{prefix}.scale1",
                f"{prefix}.scale2",
                *(
                    f"{prefix}.attn.to_{name}.{kind}"
                    for name in ("q", "k", "v")
                    for kind in ("weight", "bias")
                ),
                f"{prefix}.attn.to_out.0.weight",
                f"{prefix}.attn.to_out.0.bias",
                f"{prefix}.ff.net.0.proj.weight",
                f"{prefix}.ff.net.0.proj.bias",
                f"{prefix}.ff.net.2.weight",
                f"{prefix}.ff.net.2.bias",
            ]
        )
    return tuple(names)


def _heads(network, tensor):
    reshape = network.add_shuffle(tensor)
    reshape.reshape_dims = (-1, SEQUENCE, HEADS, HEAD_DIM)
    reshape.second_transpose = trt.Permutation([0, 2, 1, 3])
    return reshape.get_output(0)


def _rows(network, tensor):
    reshape = network.add_shuffle(tensor)
    reshape.first_transpose = trt.Permutation([0, 2, 1, 3])
    reshape.reshape_dims = (-1, SEQUENCE, DIM)
    return reshape.get_output(0)


def _per_head_norm(network, tensor):
    reshape = network.add_shuffle(tensor)
    reshape.reshape_dims = (-1, SEQUENCE, HEADS, HEAD_DIM)
    normalized = op.rms_norm(
        network, reshape.get_output(0), np.ones(HEAD_DIM, np.float32), HEAD_DIM, NORM_EPS
    )
    flatten = network.add_shuffle(normalized)
    flatten.reshape_dims = (-1, SEQUENCE, DIM)
    return flatten.get_output(0)


def _broadcast_rows(network, reference, value):
    """Broadcast a ``[1, rows, width]`` constant over runtime tile batch ``B``."""

    array = np.asarray(value)
    if array.ndim != 3 or array.shape[0] != 1:
        raise ValueError("MiniMax-H3 VAE batch constants must have shape [1, rows, width]")
    rows, width = array.shape[1:]
    batch_reference = op.dynamic_slice(
        network,
        reference,
        (0, 0, 0),
        (None, rows, width),
    )
    zero = op.cast(network, op.constant(network, np.zeros((1, 1, 1), np.float32)), reference.dtype)
    batch_zeros = network.add_elementwise(
        batch_reference, zero, trt.ElementWiseOperation.PROD
    ).get_output(0)
    constant = op.cast(network, op.weight_constant(network, array), reference.dtype)
    return network.add_elementwise(
        batch_zeros, constant, trt.ElementWiseOperation.SUM
    ).get_output(0)


def _rope_cache(network, reference):
    axes = [
        2.0 * (np.arange(size, dtype=np.float32) + 0.5) / size - 1.0
        for size in (FRAMES, HEIGHT, WIDTH)
    ]
    positions = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(TOKENS, 3)
    positions = np.concatenate(
        [positions, np.zeros((REGISTER_TOKENS + 1, 3), dtype=np.float32)], axis=0
    )
    inverse = 1.0 / (100.0 ** np.arange(0, 1, 6.0 / ROTARY_DIM, dtype=np.float32))
    frequency = (2.0 * math.pi * positions[:, :, None] * inverse[None, None, :]).reshape(
        SEQUENCE, ROTARY_DIM // 2
    )
    cos = np.broadcast_to(
        np.cos(frequency).reshape(1, SEQUENCE, ROTARY_DIM // 2),
        (1, SEQUENCE, ROTARY_DIM // 2),
    ).copy()
    sin = np.broadcast_to(
        np.sin(frequency).reshape(1, SEQUENCE, ROTARY_DIM // 2),
        (1, SEQUENCE, ROTARY_DIM // 2),
    ).copy()
    cos = _broadcast_rows(network, reference, cos)
    sin = _broadcast_rows(network, reference, sin)
    return cos, sin


def _rope(network, tensor, cos, sin):
    value = _heads(network, tensor)
    layer = network.add_rotary_embedding(value, cos, sin, False, ROTARY_DIM)
    if layer is None:
        raise RuntimeError("TensorRT failed to add MiniMax-H3 VAE rotary embedding")
    return _rows(network, layer.get_output(0))


def _fused_qkv(
    network,
    hidden,
    weights,
    prefix: str,
    *,
    consume_weights: bool = False,
):
    weight_keys = tuple(f"{prefix}.to_{name}.weight" for name in ("q", "k", "v"))
    bias_keys = tuple(f"{prefix}.to_{name}.bias" for name in ("q", "k", "v"))
    packed = op.linear(
        network,
        hidden,
        np.concatenate([weights[key] for key in weight_keys], axis=0),
        np.concatenate([weights[key] for key in bias_keys], axis=0),
        compute_dtype=trt.float16,
    )
    if consume_weights:
        for key in (*weight_keys, *bias_keys):
            weights.pop(key)
    return tuple(
        op.dynamic_slice(
            network,
            packed,
            (0, 0, part * DIM),
            (None, SEQUENCE, DIM),
        )
        for part in range(3)
    )


def _swiglu(network, hidden, weights, prefix: str):
    projected = op.linear(
        network,
        hidden,
        weights[f"{prefix}.net.0.proj.weight"],
        weights[f"{prefix}.net.0.proj.bias"],
        compute_dtype=trt.float16,
    )
    value = op.dynamic_slice(
        network, projected, (0, 0, 0), (None, SEQUENCE, FFN_DIM)
    )
    gate = op.dynamic_slice(
        network, projected, (0, 0, FFN_DIM), (None, SEQUENCE, FFN_DIM)
    )
    gate = op.silu(network, gate)
    hidden = network.add_elementwise(value, gate, trt.ElementWiseOperation.PROD).get_output(0)
    return op.linear(
        network,
        hidden,
        weights[f"{prefix}.net.2.weight"],
        weights[f"{prefix}.net.2.bias"],
        compute_dtype=trt.float16,
    )


@op.cleanup_failed_build
def build_vae_tile_decoder_engine(
    weights: dict,
    *,
    verbose: bool = False,
    consume_weights: bool = False,
    workspace_bytes: int | None = None,
    weight_streaming: bool = False,
    output_path: str | Path | None = None,
) -> bytes | dict[str, int | str]:
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    config = builder.create_builder_config()
    op.configure_builder(config, weight_streaming=weight_streaming)
    op.configure_workspace(
        config,
        workspace_bytes,
        default_bytes=VAE_TILE_DECODER_DEFAULT_WORKSPACE_BYTES,
    )
    latent = network.add_input(
        "latent_tiles", trt.float32, (-1, CHANNELS, FRAMES, HEIGHT, WIDTH)
    )
    shape_profile = builder.create_optimization_profile()
    shape_profile.set_shape(
        "latent_tiles",
        (MIN_BATCH, CHANNELS, FRAMES, HEIGHT, WIDTH),
        (OPT_BATCH, CHANNELS, FRAMES, HEIGHT, WIDTH),
        (MAX_BATCH, CHANNELS, FRAMES, HEIGHT, WIDTH),
    )
    profile_index = config.add_optimization_profile(shape_profile)
    if profile_index != 0:
        raise RuntimeError("TensorRT rejected MiniMax-H3 VideoVAE dynamic tile profile")
    rows = network.add_shuffle(latent)
    rows.first_transpose = trt.Permutation([0, 2, 3, 4, 1])
    rows.reshape_dims = (-1, TOKENS, CHANNELS)
    post_weight = weights["post_quant_conv.weight"].reshape(CHANNELS, CHANNELS)
    hidden = op.linear(
        network,
        rows.get_output(0),
        post_weight,
        weights["post_quant_conv.bias"],
        compute_dtype=trt.float16,
    )
    hidden = op.linear(
        network,
        hidden,
        weights["decoder.proj_in.weight"],
        weights["decoder.proj_in.bias"],
        compute_dtype=trt.float16,
    )
    hidden = op.cast(network, hidden, trt.float16)
    registers = _broadcast_rows(
        network,
        hidden,
        np.asarray(weights["decoder.register_tokens"]).reshape(1, REGISTER_TOKENS, DIM),
    )
    cls = _broadcast_rows(network, hidden, np.zeros((1, 1, DIM), np.float32))
    packed = network.add_concatenation([hidden, registers, cls])
    packed.axis = 1
    hidden = packed.get_output(0)
    cos, sin = _rope_cache(network, hidden)

    for index in range(LAYERS):
        prefix = f"decoder.transformer_blocks.{index}"
        normalized = op.rms_norm(network, hidden, weights[f"{prefix}.norm1.weight"], DIM, NORM_EPS)
        q, k, v = _fused_qkv(
            network,
            normalized,
            weights,
            f"{prefix}.attn",
            consume_weights=consume_weights,
        )
        q, k = _per_head_norm(network, q), _per_head_norm(network, k)
        q, k = _rope(network, q, cos, sin), _rope(network, k, cos, sin)
        q4, k4, v4 = _heads(network, q), _heads(network, k), _heads(network, v)
        scale = op.cast(
            network,
            op.constant(
                network,
                np.full((1, 1, 1, 1), 1.0 / math.sqrt(HEAD_DIM), np.float32),
            ),
            trt.float16,
        )
        q4 = network.add_elementwise(q4, scale, trt.ElementWiseOperation.PROD).get_output(0)
        attention = network.add_attention(q4, k4, v4, trt.AttentionNormalizationOp.SOFTMAX, False)
        if attention is None:
            raise RuntimeError(f"TensorRT failed to add MiniMax-H3 VAE attention layer {index}")
        attention.name = f"{prefix}.attn.native_attention"
        attention.metadata = f"trtmc.native_op=IAttention;source={attention.name}"
        attention.get_output(0).name = f"{attention.name}.output"
        attention.decomposable = False
        update = op.linear(
            network,
            _rows(network, attention.get_output(0)),
            weights[f"{prefix}.attn.to_out.0.weight"],
            weights[f"{prefix}.attn.to_out.0.bias"],
            compute_dtype=trt.float16,
        )
        update = op.cast(network, update, trt.float32)
        scale1 = op.weight_constant(network, weights[f"{prefix}.scale1"].reshape(1, 1, DIM))
        scale1 = op.cast(network, scale1, update.dtype)
        update = network.add_elementwise(update, scale1, trt.ElementWiseOperation.PROD).get_output(
            0
        )
        hidden = op.cast(network, hidden, trt.float32)
        hidden = network.add_elementwise(hidden, update, trt.ElementWiseOperation.SUM).get_output(0)

        normalized = op.rms_norm(network, hidden, weights[f"{prefix}.norm2.weight"], DIM, NORM_EPS)
        update = _swiglu(network, normalized, weights, f"{prefix}.ff")
        update = op.cast(network, update, trt.float32)
        scale2 = op.weight_constant(network, weights[f"{prefix}.scale2"].reshape(1, 1, DIM))
        scale2 = op.cast(network, scale2, update.dtype)
        update = network.add_elementwise(update, scale2, trt.ElementWiseOperation.PROD).get_output(
            0
        )
        hidden = network.add_elementwise(hidden, update, trt.ElementWiseOperation.SUM).get_output(0)

    gamma = op.weight_constant(network, weights["decoder.norm_out.weight"].reshape(1, 1, DIM))
    beta = op.weight_constant(network, weights["decoder.norm_out.bias"].reshape(1, 1, DIM))
    hidden_fp32 = op.cast(network, hidden, trt.float32)
    gamma = op.cast(network, gamma, hidden_fp32.dtype)
    beta = op.cast(network, beta, hidden_fp32.dtype)
    norm = network.add_normalization_v2(hidden_fp32, gamma, beta, 1 << 2)
    norm.epsilon = NORM_EPS
    pixels = op.linear(
        network,
        norm.get_output(0),
        weights["decoder.proj_out.weight"],
        weights["decoder.proj_out.bias"],
        compute_dtype=trt.float16,
    )
    pixels = op.dynamic_slice(
        network,
        pixels,
        (0, 0, 0),
        (None, TOKENS, OUT_CHANNELS * PATCH_T * PATCH * PATCH),
    )
    output = network.add_shuffle(pixels)
    output.reshape_dims = (-1, FRAMES, HEIGHT, WIDTH, OUT_CHANNELS, PATCH_T, PATCH, PATCH)
    output.second_transpose = trt.Permutation([0, 4, 1, 5, 2, 6, 3, 7])
    final = network.add_shuffle(output.get_output(0))
    final.reshape_dims = (-1, OUT_CHANNELS, FRAMES * PATCH_T, HEIGHT * PATCH, WIDTH * PATCH)
    # Keep the decoder math in FP16 and expose FP32 for the runtime's tile
    # assembly, ImageNet denormalization, and clamp. The resulting blend
    # delta against autocast assembly is covered by the decoded-video gate.
    result = op.cast(network, final.get_output(0), trt.float32)
    result.name = "decoded_tiles"
    network.mark_output(result)
    op.validate_native_network(network, expected_attentions=LAYERS, label="VAE tile decoder")
    print(
        "[minimax-h3] building native VAE tile decoder: "
        f"batch_profile={MIN_BATCH}/{OPT_BATCH}/{MAX_BATCH}, "
        f"sequence={SEQUENCE}, layers={LAYERS}",
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
        raise RuntimeError("TensorRT failed to build MiniMax-H3 VAE tile decoder")
    del network, config, builder
    gc.collect()
    return record if record is not None else bytes(plan)
