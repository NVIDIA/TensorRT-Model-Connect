# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-VL language layers as the flux3 text encoder.

Reproduces the reference text conditioning for text-only prompts: token
embeddings, decoder layers ``0 .. max(text_layers) - 1`` (no final norm, no LM
head, no vision tower), a causal mask that also hides padding keys, padding
tokens at position 1, and the hidden states after layers ``4, 8, ..., 32``
concatenated per token into one ``8 * hidden`` vector. The window length is a
dynamic dimension because the transformer attends to the padded rows.
"""

from __future__ import annotations

import ml_dtypes
import numpy as np
import tensorrt as trt

from . import rope
from .trt_ops import Graph, new_builder_config, new_network, serialize
from .weights import ComponentWeights

PREFIX = "model.language_model."


def _rope_theta(config: dict) -> float:
    """RoPE base from a Transformers 4.x (``rope_theta``) or 5.x (``rope_parameters``) text config."""

    theta = config.get("rope_theta")
    if theta is None:
        theta = (config.get("rope_parameters") or {}).get("rope_theta")
    if theta is None:
        raise ValueError("flux3 text encoder config has no rope_theta")
    return float(theta)


def _rms_norm(g: Graph, x: trt.ITensor, weight: np.ndarray, eps: float) -> trt.ITensor:
    """Transformers Qwen3 RMSNorm: float32 statistics, cast, then the weight in the compute dtype."""

    normed = g.cast(g.rms_norm32(x, np.ones(weight.shape, dtype=np.float32), eps), g.compute)
    shape = (1,) * (len(x.shape) - 1) + (weight.shape[-1],)
    return g.mul(normed, g.constant(weight.reshape(shape), g.compute))


def _rotate(g: Graph, x: trt.ITensor, cos: trt.ITensor, sin: trt.ITensor, head_dim: int):
    index, sign = rope.rotate_half_rotation(head_dim)
    rotated = g.mul(g.take_last(x, index), g.constant(sign.reshape(1, 1, -1), g.compute))
    return g.add(g.mul(x, cos), g.mul(rotated, sin))


def _repeat_kv(g: Graph, x: trt.ITensor, kv_heads: int, groups: int, head_dim: int):
    if groups == 1:
        return x
    grouped = g.reshape(x, (1, kv_heads, 1, -1, head_dim))
    repeated = g.concat([grouped] * groups, axis=2)
    return g.reshape(repeated, (1, kv_heads * groups, -1, head_dim))


def _mask(g: Graph, positions: trt.ITensor, valid_length: trt.ITensor) -> trt.ITensor:
    """Additive ``[1, 1, L, L]`` mask: query ``i`` sees keys ``j <= i`` with ``j < valid``."""

    rows = g.reshape(positions, (1, 1, -1, 1))
    cols = g.reshape(positions, (1, 1, 1, -1))
    future = g.net.add_elementwise(cols, rows, trt.ElementWiseOperation.GREATER).get_output(0)
    causal = g.net.add_unary(future, trt.UnaryOperation.NOT).get_output(0)
    valid = g.net.add_elementwise(
        cols, g.reshape(valid_length, (1, 1, 1, 1)), trt.ElementWiseOperation.LESS
    ).get_output(0)
    allowed = g.net.add_elementwise(causal, valid, trt.ElementWiseOperation.AND).get_output(0)
    blocked = float(ml_dtypes.finfo(ml_dtypes.bfloat16).min)
    zero = g.scalar(0.0, g.compute, 4)
    neg = g.scalar(blocked, g.compute, 4)
    return g.net.add_select(allowed, zero, neg).get_output(0)


def _layer(g: Graph, w: ComponentWeights, index: int, h: trt.ITensor, ctx: dict) -> trt.ITensor:
    cfg = ctx["config"]
    hidden, heads, kv_heads = (
        cfg["hidden_size"],
        cfg["num_attention_heads"],
        cfg["num_key_value_heads"],
    )
    head_dim, inter, eps = cfg["head_dim"], cfg["intermediate_size"], cfg["rms_norm_eps"]
    p = f"layers.{index}."

    x = _rms_norm(g, h, w.get(p + "input_layernorm.weight", (hidden,)), eps)
    q = g.heads(
        g.linear(x, w.get(p + "self_attn.q_proj.weight", (heads * head_dim, hidden))),
        heads,
        head_dim,
    )
    k = g.heads(
        g.linear(x, w.get(p + "self_attn.k_proj.weight", (kv_heads * head_dim, hidden))),
        kv_heads,
        head_dim,
    )
    v = g.heads(
        g.linear(x, w.get(p + "self_attn.v_proj.weight", (kv_heads * head_dim, hidden))),
        kv_heads,
        head_dim,
    )
    q = _rms_norm(g, q, w.get(p + "self_attn.q_norm.weight", (head_dim,)), eps)
    k = _rms_norm(g, k, w.get(p + "self_attn.k_norm.weight", (head_dim,)), eps)
    q = _rotate(g, q, ctx["cos"], ctx["sin"], head_dim)
    k = _rotate(g, k, ctx["cos"], ctx["sin"], head_dim)
    q = g.to_attention_layout(g.scale_query(q, head_dim, g.compute))
    k = _repeat_kv(g, g.to_attention_layout(k), kv_heads, heads // kv_heads, head_dim)
    v = _repeat_kv(g, g.to_attention_layout(v), kv_heads, heads // kv_heads, head_dim)
    attn = g.from_attention_layout(g.attention(q, k, v, ctx["mask"]), heads * head_dim)
    h = g.add(h, g.linear(attn, w.get(p + "self_attn.o_proj.weight", (hidden, heads * head_dim))))

    x = _rms_norm(g, h, w.get(p + "post_attention_layernorm.weight", (hidden,)), eps)
    gate = g.silu(g.linear(x, w.get(p + "mlp.gate_proj.weight", (inter, hidden))))
    up = g.linear(x, w.get(p + "mlp.up_proj.weight", (inter, hidden)))
    return g.add(h, g.linear(g.mul(gate, up), w.get(p + "mlp.down_proj.weight", (hidden, inter))))


def build_text_encoder(
    weights: ComponentWeights,
    config: dict,
    *,
    output_layers: tuple[int, ...],
    max_tokens: int,
    opt_tokens: int,
    min_tokens: int,
    compute: trt.DataType,
    optimization_level: int,
    verbose: bool = False,
) -> bytes:
    """Serialized engine: ``input_ids [1, L]``, ``valid_length [1]`` → ``text_embeds [1, L, 8 * hidden]``."""

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = new_network(builder)
    g = Graph(network, compute)
    hidden, head_dim = config["hidden_size"], config["head_dim"]

    input_ids = network.add_input("input_ids", trt.int32, (1, -1))
    valid_length = network.add_input("valid_length", trt.int32, (1,))
    ids = g.reshape(input_ids, (-1,))
    h = g.gather(
        g.constant(weights.get("embed_tokens.weight", (config["vocab_size"], hidden)), compute),
        ids,
        0,
    )

    positions = g.iota(g.dim(input_ids, 1))
    is_valid = network.add_elementwise(
        positions, valid_length, trt.ElementWiseOperation.LESS
    ).get_output(0)
    pad_position = g.constant(np.ones((1,), dtype=np.int32))
    rope_positions = network.add_select(is_valid, positions, pad_position).get_output(0)
    cos_table, sin_table = rope.text_encoder_tables(max_tokens, head_dim, _rope_theta(config))
    ctx = {
        "config": config,
        "cos": g.reshape(
            g.gather(g.constant(cos_table, compute), rope_positions, 0), (-1, 1, head_dim)
        ),
        "sin": g.reshape(
            g.gather(g.constant(sin_table, compute), rope_positions, 0), (-1, 1, head_dim)
        ),
        "mask": _mask(g, positions, valid_length),
    }

    stacked = []
    for index in range(max(output_layers)):
        h = _layer(g, weights, index, h, ctx)
        if index + 1 in output_layers:
            stacked.append(h)
    embeds = g.reshape(g.concat(stacked, axis=1), (1, -1, hidden * len(output_layers)))
    embeds.name = "text_embeds"
    network.mark_output(embeds)

    config_trt = new_builder_config(builder, optimization_level)
    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids", (1, min_tokens), (1, opt_tokens), (1, max_tokens))
    config_trt.add_optimization_profile(profile)
    return serialize(builder, network, config_trt)
