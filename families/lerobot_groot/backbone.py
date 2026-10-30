# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The kept Qwen3-VL text layers of the GR00T N1.7 backbone.

GR00T keeps the first ``select_layer`` decoder layers and reads the output of the
last one before the final norm. The image tokens replace the image placeholders,
and the DeepStack features are added to those positions after the first layers.
Rotary positions are the three-row (temporal, height, width) multimodal ids,
interleaved across the frequencies as ``apply_interleaved_mrope`` does.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import graph
from .checkpoint import embedding_key, require

WIDTH = 2048
HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128
MLP = 6144
RMS_EPS = 1e-6
ROPE_THETA = 5000000.0
MROPE_SECTION = (24, 20, 20)
VOCAB = 151936

_ROOT = "backbone.model.model.language_model."


def kept_layers(weights: dict[str, np.ndarray]) -> int:
    """How many decoder layers the checkpoint keeps (GR00T truncates at ``select_layer``)."""
    count = 0
    while f"{_ROOT}layers.{count}.input_layernorm.weight" in weights:
        count += 1
    if count == 0:
        raise ValueError("LeRobot GR00T checkpoint has no backbone decoder layers")
    return count


def mrope_frequencies() -> np.ndarray:
    """``M[3, HEAD_DIM/2]``: angle ``= position[row] * M[row, i]`` with one non-zero row per frequency."""
    inverse = np.float32(1.0) / (np.float32(ROPE_THETA) ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / np.float32(HEAD_DIM)))
    rows = np.zeros(HEAD_DIM // 2, dtype=np.int64)  # temporal by default
    for row, offset in ((1, 1), (2, 2)):  # height, width: every third frequency up to 3 * section
        rows[offset:MROPE_SECTION[row] * 3:3] = row
    matrix = np.zeros((3, HEAD_DIM // 2), dtype=np.float32)
    matrix[rows, np.arange(HEAD_DIM // 2)] = inverse.astype(np.float32)
    return matrix


def rope_tables(network, position_ids):
    """``cos``/``sin`` ``[S, 1, HEAD_DIM]`` in float32 from the int32 ``position_ids[3, S]``."""
    positions = graph.shuffle(network, graph.f32(network, position_ids), first=(1, 0))
    angles = network.add_matrix_multiply(
        positions, trt.MatrixOperation.NONE, graph.const(network, mrope_frequencies()),
        trt.MatrixOperation.NONE).get_output(0)
    angles = graph.concat(network, [angles, angles], 1)
    cos = network.add_unary(angles, trt.UnaryOperation.COS).get_output(0)
    sin = network.add_unary(angles, trt.UnaryOperation.SIN).get_output(0)
    return (graph.shuffle(network, t, reshape=(0, 1, HEAD_DIM)) for t in (cos, sin))


def _heads(network, x, weight, norm_weight, heads, cos, sin, dtype):
    """Project, RMS-normalise each head (QK-norm) and rotate -> float32 ``[S, heads*D]``."""
    rows = graph.shuffle(network, graph.linear(network, x, weight, dtype=dtype), reshape=(0, heads, HEAD_DIM))
    rows = graph.rms_norm(network, rows, norm_weight, RMS_EPS, dtype=dtype)
    return graph.shuffle(network, graph.rotate(network, rows, cos, sin, HEAD_DIM), reshape=(0, heads * HEAD_DIM))


def _scatter(network, rows, image_index, is_image, base=None):
    """Rows of ``rows`` gathered at ``image_index`` where ``is_image``; else ``base`` (or zero)."""
    gathered = graph.f32(network, network.add_gather(rows, image_index, 0).get_output(0))
    if base is None:
        base = graph.mul(network, gathered, graph.scalar(network, 0.0, 2))
    return network.add_select(is_image, gathered, base).get_output(0)


def build_backbone(network, weights, *, token_ids, image_index, image_mask, position_ids, causal_bias,
                   image_tokens, deepstack, dtype):
    """Float32 hidden states ``[S, WIDTH]`` of the last kept layer, before the final norm.

    ``token_ids``/``image_index``/``image_mask`` are ``[S]`` int32 (``image_index`` points into the
    image tokens where ``image_mask`` is 1), ``position_ids[3, S]``, ``causal_bias[1, 1, S, S]``.
    """
    table = require(weights, embedding_key(weights), (VOCAB, WIDTH))
    text = graph.f32(network, network.add_gather(graph.const(network, table, dtype), token_ids, 0).get_output(0))
    is_image = network.add_elementwise(
        image_mask, graph.const_int32(network, [1]), trt.ElementWiseOperation.EQUAL).get_output(0)
    is_image = graph.shuffle(network, is_image, reshape=(-1, 1))
    h = _scatter(network, image_tokens, image_index, is_image, base=text)
    cos, sin = rope_tables(network, position_ids)

    for index in range(kept_layers(weights)):
        p = f"{_ROOT}layers.{index}."
        norm = graph.rms_norm(network, h, weights[p + "input_layernorm.weight"], RMS_EPS, dtype=dtype)
        q = _heads(network, norm, weights[p + "self_attn.q_proj.weight"], weights[p + "self_attn.q_norm.weight"],
                   HEADS, cos, sin, dtype)
        k = _heads(network, norm, weights[p + "self_attn.k_proj.weight"], weights[p + "self_attn.k_norm.weight"],
                   KV_HEADS, cos, sin, dtype)
        v = graph.linear(network, norm, weights[p + "self_attn.v_proj.weight"], dtype=dtype)
        context = graph.attention(network, q, k, v, heads=HEADS, kv_heads=KV_HEADS, head_dim=HEAD_DIM,
                                  dtype=dtype, bias=causal_bias)
        h = graph.add(network, h, graph.f32(network, graph.linear(
            network, context, weights[p + "self_attn.o_proj.weight"], dtype=dtype)))
        norm = graph.rms_norm(network, h, weights[p + "post_attention_layernorm.weight"], RMS_EPS, dtype=dtype)
        gate = graph.silu(network, graph.linear(network, norm, weights[p + "mlp.gate_proj.weight"], dtype=dtype))
        hidden = graph.mul(network, gate, graph.linear(network, norm, weights[p + "mlp.up_proj.weight"], dtype=dtype))
        h = graph.add(network, h, graph.f32(network, graph.linear(
            network, hidden, weights[p + "mlp.down_proj.weight"], dtype=dtype)))
        if index < len(deepstack):
            h = graph.add(network, h, _scatter(network, deepstack[index], image_index, is_image))
    return h
