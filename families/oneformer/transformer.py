# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OneFormer task token, query transformer, masked-attention decoder and prediction heads.

Read from ``OneFormerTransformerModule`` / ``OneFormerTransformerDecoder``:

* the task token is the task MLP of the prompt's token ids taken as float
  values, then ``decoder_norm`` (``use_task_norm``);
* the query transformer starts every object query from that token and attends,
  over the stride-4 grid, to keys ``sine + proj(mask_features)`` and values
  ``sine`` (the positions are the memory); it is post-norm with a final norm;
* the decoder appends the normalised task token as the last query and cycles
  the stride-32/16/8 levels; each layer is masked cross-attention, then
  self-attention, then a ReLU FFN, all post-norm;
* every prediction (initial and after each layer) is ``decoder_norm`` ->
  class head and 3-layer mask MLP, mask logits = embedding . mask_features; its
  bilinear resize to the next level, ``sigmoid < 0.5``, is the next layer's
  mask, and a query whose mask covers every key is unmasked;
* ``nn.MultiheadAttention`` scales queries by ``sqrt(1 / head_dim)`` and turns
  the boolean mask into -inf; the decoder self-attention scales its projected
  queries by ``head_dim ** -0.5``.

The semantic map is ``softmax(class)[..., :-1]`` times ``sigmoid(mask)``,
summed over queries, at stride 4.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from .config import NUM_FEATURE_LEVELS
from .position import sine_tokens

E = trt.ElementWiseOperation
PREFIX = "model.transformer_module"
DECODER = f"{PREFIX}.decoder"


def _heads_split(g, tensor, heads):
    """(1, N, C) -> (1, heads, N, C / heads)."""
    return g.reshape(tensor, (1, -1, heads, int(tensor.shape[2]) // heads), second=(0, 2, 1, 3))


def _heads_merge(g, tensor):
    """(1, heads, N, d) -> (1, N, heads * d)."""
    return g.reshape(
        tensor, (1, -1, int(tensor.shape[1]) * int(tensor.shape[3])), first=(0, 2, 1, 3)
    )


def _attend(g, query, key, value, heads, blocked=None):
    """softmax(query . key^T [blocked = -inf]) . value on already projected, scaled tensors."""
    scores = g.matmul(
        _heads_split(g, query, heads), _heads_split(g, key, heads), transpose_right=True
    )
    if blocked is not None:
        scores = g.network.add_select(blocked, g.scalar(scores, -np.inf), scores).get_output(0)
    return _heads_merge(g, g.matmul(g.softmax(scores, 3), _heads_split(g, value, heads)))


def _packed_mha(g, weights, prefix, query, key, value, heads, blocked=None):
    """torch.nn.MultiheadAttention (need_weights path) with packed in_proj weights."""
    weight = weights.get(f"{prefix}.in_proj_weight")
    bias = weights.get(f"{prefix}.in_proj_bias")
    width = weight.shape[1]
    q = g.linear(query, weight[:width], bias[:width])
    k = g.linear(key, weight[width : 2 * width], bias[width : 2 * width])
    v = g.linear(value, weight[2 * width :], bias[2 * width :])
    q = g.ew(q, g.scalar(q, math.sqrt(1.0 / float(width // heads))), E.PROD)
    context = _attend(g, q, k, v, heads, blocked)
    return g.linear(
        context, weights.get(f"{prefix}.out_proj.weight"), weights.get(f"{prefix}.out_proj.bias")
    )


def _norm(g, tensor, weights, prefix, eps):
    return g.layer_norm(tensor, weights.get(f"{prefix}.weight"), weights.get(f"{prefix}.bias"), eps)


def _mlp(g, tensor, weights, prefix, layers):
    for index in range(layers):
        tensor = g.linear(
            tensor,
            weights.get(f"{prefix}.layers.{index}.0.weight"),
            weights.get(f"{prefix}.layers.{index}.0.bias"),
        )
        if index + 1 < layers:
            tensor = g.relu(tensor)
    return tensor


def _query_transformer(g, task_token, memory, memory_position, weights, config):
    eps = config.layer_norm_eps
    heads = config.num_attention_heads
    count = config.num_queries - 1
    query_position = g.constant(weights.get(f"{PREFIX}.queries_embedder.weight")[:count][None])
    output = g.add(
        task_token, g.constant(np.zeros((1, count, config.hidden_dim), dtype=np.float32))
    )
    keys = g.add(memory, memory_position)
    prefix = f"{DECODER}.query_transformer.decoder"
    for layer in range(config.query_dec_layers):
        base = f"{prefix}.layers.{layer}"
        posed = g.add(output, query_position)
        attended = _packed_mha(g, weights, f"{base}.self_attn", posed, posed, output, heads)
        output = _norm(g, g.add(output, attended), weights, f"{base}.norm1", eps)
        attended = _packed_mha(
            g, weights, f"{base}.multihead_attn", g.add(output, query_position), keys, memory, heads
        )
        output = _norm(g, g.add(output, attended), weights, f"{base}.norm2", eps)
        ffn = g.relu(
            g.linear(
                output, weights.get(f"{base}.linear1.weight"), weights.get(f"{base}.linear1.bias")
            )
        )
        ffn = g.linear(
            ffn, weights.get(f"{base}.linear2.weight"), weights.get(f"{base}.linear2.bias")
        )
        output = _norm(g, g.add(output, ffn), weights, f"{base}.norm3", eps)
    return _norm(g, output, weights, f"{prefix}.norm", eps)


def _predict(g, output, mask_features_flat, mask_shape, target_hw, weights, config):
    """Class logits, mask logits (1, Q, H/4, W/4) and the blocked-key mask for ``target_hw``."""
    decoded = _norm(g, output, weights, f"{DECODER}.decoder_norm", config.layer_norm_eps)
    logits = g.linear(
        decoded,
        weights.get(f"{DECODER}.class_embed.weight"),
        weights.get(f"{DECODER}.class_embed.bias"),
    )
    embedding = _mlp(g, decoded, weights, f"{DECODER}.mask_embed", 3)
    masks = g.reshape(g.matmul(embedding, mask_features_flat), mask_shape)
    blocked = None
    if target_hw is not None:
        height, width = target_hw
        resized = g.resize_bilinear(masks, g.vec([1, config.num_queries, height, width]))
        below = g.ew(g.sigmoid(resized), g.scalar(resized, 0.5), E.LESS)
        blocked = g.reshape(below, (1, 1, config.num_queries, -1))
        # A query whose mask blocks every key attends everywhere instead.
        coverage = g.network.add_reduce(
            g.cast(blocked, trt.float32), trt.ReduceOperation.MIN, 1 << 3, True
        ).get_output(0)
        everywhere = g.ew(coverage, g.f32([0.5], (1, 1, 1, 1)), E.GREATER)
        blocked = g.ew(blocked, g.unary(everywhere, trt.UnaryOperation.NOT), E.AND)
    return logits, masks, blocked


def _decoder_layer(g, output, feature, position, query_position, blocked, weights, prefix, config):
    eps = config.layer_norm_eps
    heads = config.num_attention_heads
    attended = _packed_mha(
        g,
        weights,
        f"{prefix}.cross_attn.multihead_attn",
        g.add(output, query_position),
        g.add(feature, position),
        feature,
        heads,
        blocked,
    )
    output = _norm(g, g.add(output, attended), weights, f"{prefix}.cross_attn.norm", eps)

    base = f"{prefix}.self_attn.self_attn"
    posed = g.add(output, query_position)
    q = g.linear(posed, weights.get(f"{base}.q_proj.weight"), weights.get(f"{base}.q_proj.bias"))
    q = g.ew(q, g.scalar(q, (config.hidden_dim // heads) ** -0.5), E.PROD)
    k = g.linear(posed, weights.get(f"{base}.k_proj.weight"), weights.get(f"{base}.k_proj.bias"))
    v = g.linear(output, weights.get(f"{base}.v_proj.weight"), weights.get(f"{base}.v_proj.bias"))
    attended = g.linear(
        _attend(g, q, k, v, heads),
        weights.get(f"{base}.out_proj.weight"),
        weights.get(f"{base}.out_proj.bias"),
    )
    output = _norm(g, g.add(output, attended), weights, f"{prefix}.self_attn.norm", eps)

    ffn = g.relu(
        g.linear(
            output,
            weights.get(f"{prefix}.ffn.linear1.weight"),
            weights.get(f"{prefix}.ffn.linear1.bias"),
        )
    )
    ffn = g.linear(
        ffn, weights.get(f"{prefix}.ffn.linear2.weight"), weights.get(f"{prefix}.ffn.linear2.bias")
    )
    return _norm(g, g.add(output, ffn), weights, f"{prefix}.ffn.norm", eps)


def build_transformer(g, levels, mask_features, task_ids, weights, config, dim_t):
    """Return (class logits (1, Q, classes + 1), mask logits (1, Q, H/4, W/4))."""
    eps = config.layer_norm_eps
    task = g.constant(np.asarray(task_ids, dtype=np.float32)[None])
    task = _mlp(g, task, weights, "model.task_encoder.task_mlp", 2)
    task_token = _norm(
        g, g.reshape(task, (1, 1, config.hidden_dim)), weights, f"{DECODER}.decoder_norm", eps
    )

    mask_height, mask_width = g.dim(mask_features, 2), g.dim(mask_features, 3)
    memory = sine_tokens(g, mask_height, mask_width, dim_t)
    projected = g.conv(
        mask_features,
        weights.get(f"{DECODER}.query_input_projection.weight"),
        weights.get(f"{DECODER}.query_input_projection.bias"),
    )
    memory_position = g.reshape(projected, (1, config.hidden_dim, -1), second=(0, 2, 1))
    objects = _query_transformer(g, task_token, memory, memory_position, weights, config)
    output = g.concat([objects, task_token], 1)

    level_embed = weights.get(f"{PREFIX}.level_embed.weight")
    features, positions, sizes = [], [], []
    for level, feature in enumerate(levels):
        height, width = g.dim(feature, 2), g.dim(feature, 3)
        sizes.append((height, width))
        tokens = g.reshape(feature, (1, config.hidden_dim, -1), second=(0, 2, 1))
        features.append(g.add(tokens, g.constant(level_embed[level], shape=(1, 1, -1))))
        positions.append(sine_tokens(g, height, width, dim_t))
    query_position = g.constant(weights.get(f"{PREFIX}.queries_embedder.weight")[None])

    mask_flat = g.reshape(mask_features, (1, config.mask_dim, -1))
    mask_shape = [1, config.num_queries, mask_height, mask_width]
    logits, masks, blocked = _predict(g, output, mask_flat, mask_shape, sizes[0], weights, config)
    for index in range(config.decoder_layers - 1):
        level = index % NUM_FEATURE_LEVELS
        output = _decoder_layer(
            g,
            output,
            features[level],
            positions[level],
            query_position,
            blocked,
            weights,
            f"{DECODER}.layers.{index}",
            config,
        )
        last = index == config.decoder_layers - 2
        target = None if last else sizes[(index + 1) % NUM_FEATURE_LEVELS]
        logits, masks, blocked = _predict(g, output, mask_flat, mask_shape, target, weights, config)
    return logits, masks


def semantic_scores(g, logits, masks, config):
    """einsum("qc,qhw->chw", softmax(logits)[:, :-1], sigmoid(masks)) as (1, classes, H/4, W/4)."""
    classes = config.num_labels
    probabilities = g.softmax(logits, 2)
    probabilities = g.slice(probabilities, [0, 0, 0], [1, config.num_queries, classes])
    flat = g.sigmoid(g.reshape(masks, (1, config.num_queries, -1)))
    scores = g.matmul(g.transpose(probabilities, (0, 2, 1)), flat)
    return g.reshape(scores, [1, classes, g.dim(masks, 2), g.dim(masks, 3)])
