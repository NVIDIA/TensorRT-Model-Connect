# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Mask2Former transformer module and the semantic-segmentation head.

Read from `Mask2FormerTransformerModule`, `Mask2FormerMaskedAttentionDecoder` and
`Mask2FormerImageProcessor.post_process_semantic_segmentation`:

* the decoder runs `decoder_layers - 1` post-norm layers; layer `i` cross-attends to pixel-decoder level `i % 3`,
  then self-attends, then applies its feed-forward block;
* before every layer the mask predictor turns the layer-normed queries into mask logits at the mask-feature
  resolution; resized bilinearly to the next level, `sigmoid < 0.5` marks the positions a query may not attend to,
  and a query whose positions are all marked attends to every position instead;
* the semantic map is the einsum of the class probabilities (no-object class removed) with the sigmoid of the final
  mask logits after bilinear resizing to the reference post-processor's fixed size.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from . import graph as g
from .config import FEATURE_LEVELS, NORM_EPS, SEMANTIC_SCORE_SIZE, Mask2FormerConfig
from .pixel_decoder import sine_position_embedding

_PREFIX = "model.transformer_module"


def _layer_norm(network, tensor, weights, name):
    return g.layer_norm(
        network, tensor, weights[f"{name}.weight"], weights[f"{name}.bias"], NORM_EPS
    )


def _mask_embeddings(network, queries, weights):
    hidden = queries
    for index in range(3):
        base = f"{_PREFIX}.decoder.mask_predictor.mask_embedder.{index}.0"
        hidden = g.linear(network, hidden, weights[f"{base}.weight"], weights[f"{base}.bias"])
        if index < 2:
            hidden = g.relu(network, hidden)
    return hidden


def _mask_logits(network, queries, mask_features, weights):
    """(1, queries, width) and (1, width, H*W) -> (1, queries, H*W) mask logits."""

    return g.matmul(network, _mask_embeddings(network, queries, weights), mask_features)


def _attention_mask(network, logits, mask_shape, target):
    """Positions a query may not attend to at the target level, as `Mask2FormerMaskPredictor` marks them.

    A query whose positions are all marked attends everywhere (`Mask2FormerMaskedAttentionDecoder.forward`).
    Returns a (1, 1, queries, h*w) boolean tensor; true means the position is excluded.
    """

    _, queries, _ = g.shape(logits)
    height, width = target
    resized = g.resize_bilinear(
        network, g.reshape(network, logits, (1, queries) + tuple(mask_shape)), height, width
    )
    excluded = g.elementwise(
        network,
        g.sigmoid(network, resized),
        g.scalar_like(network, 0.5, 4),
        trt.ElementWiseOperation.LESS,
    )
    excluded = g.reshape(network, excluded, (1, 1, queries, height * width))
    count = network.add_reduce(
        network.add_cast(excluded, trt.float32).get_output(0), trt.ReduceOperation.SUM, 1 << 3, True
    )
    if count is None:
        raise RuntimeError("TensorRT rejected the Mask2Former attention-mask count")
    complete = g.elementwise(
        network,
        count.get_output(0),
        g.scalar_like(network, float(height * width), 4),
        trt.ElementWiseOperation.EQUAL,
    )
    keep = network.add_unary(complete, trt.UnaryOperation.NOT)
    if keep is None:
        raise RuntimeError("TensorRT rejected the Mask2Former attention-mask negation")
    return g.elementwise(network, excluded, keep.get_output(0), trt.ElementWiseOperation.AND)


def _attend(network, query, key, value, excluded=None):
    """Softmax attention over (1, heads, N, D) tensors; `excluded` positions get no weight."""

    scores = g.matmul(network, query, key, transpose_right=True)
    if excluded is not None:
        blocked = g.constant(network, np.full((1, 1, 1, 1), -np.inf, dtype=np.float32))
        select = network.add_select(excluded, blocked, scores)
        if select is None:
            raise RuntimeError("TensorRT rejected the Mask2Former masked attention")
        scores = select.get_output(0)
    return g.merge_heads(network, g.matmul(network, g.softmax(network, scores, 3), value))


def _cross_attention(
    network, hidden, query_position, memory, key_input, excluded, weights, prefix, heads
):
    """`nn.MultiheadAttention` with a boolean mask: q is scaled by head_dim ** -0.5 after its projection."""

    width = g.shape(hidden)[-1]
    in_weight = np.asarray(weights[f"{prefix}.in_proj_weight"], dtype=np.float32)
    in_bias = np.asarray(weights[f"{prefix}.in_proj_bias"], dtype=np.float32)
    if in_weight.shape != (3 * width, width) or in_bias.shape != (3 * width,):
        raise ValueError(f"mask2former {prefix} in-projection has shape {in_weight.shape}")
    query = g.linear(
        network, g.add(network, hidden, query_position), in_weight[:width], in_bias[:width]
    )
    query = g.mul(
        network,
        g.split_heads(network, query, heads),
        g.scalar_like(network, math.sqrt(1.0 / (width // heads)), 4),
    )
    key = g.split_heads(
        network,
        g.linear(network, key_input, in_weight[width : 2 * width], in_bias[width : 2 * width]),
        heads,
    )
    value = g.split_heads(
        network, g.linear(network, memory, in_weight[2 * width :], in_bias[2 * width :]), heads
    )
    attended = _attend(network, query, key, value, excluded)
    return g.linear(
        network, attended, weights[f"{prefix}.out_proj.weight"], weights[f"{prefix}.out_proj.bias"]
    )


def _self_attention(network, hidden, query_position, weights, prefix, heads):
    """`Mask2FormerAttention`: queries and keys carry the position embeddings, values do not."""

    width = g.shape(hidden)[-1]
    posed = g.add(network, hidden, query_position)
    query = g.linear(
        network, posed, weights[f"{prefix}.q_proj.weight"], weights[f"{prefix}.q_proj.bias"]
    )
    query = g.mul(network, query, g.scalar_like(network, (width // heads) ** -0.5, 3))
    key = g.linear(
        network, posed, weights[f"{prefix}.k_proj.weight"], weights[f"{prefix}.k_proj.bias"]
    )
    value = g.linear(
        network, hidden, weights[f"{prefix}.v_proj.weight"], weights[f"{prefix}.v_proj.bias"]
    )
    attended = _attend(
        network,
        g.split_heads(network, query, heads),
        g.split_heads(network, key, heads),
        g.split_heads(network, value, heads),
    )
    return g.linear(
        network, attended, weights[f"{prefix}.out_proj.weight"], weights[f"{prefix}.out_proj.bias"]
    )


def build_semantic_scores(network, multi_scale, mask_features, weights, config: Mask2FormerConfig):
    """Return (1, classes, 384, 384) semantic scores, the tensor the reference post-processor resizes and argmaxes."""

    heads = config.num_attention_heads
    width = config.hidden_dim
    level_embed = np.asarray(weights[f"{_PREFIX}.level_embed.weight"], dtype=np.float32)
    if level_embed.shape != (FEATURE_LEVELS, width):
        raise ValueError(f"mask2former transformer level_embed has shape {level_embed.shape}")
    memories, keys, sizes = [], [], []
    for index, level in enumerate(multi_scale):
        height, width_l = g.shape(level)[2:]
        sizes.append((height, width_l))
        memory = g.add(
            network,
            g.tokens_from_nchw(network, level),
            g.constant(network, level_embed[index].reshape(1, 1, width)),
        )
        memories.append(memory)
        keys.append(
            g.add(
                network,
                memory,
                g.constant(network, sine_position_embedding(height, width_l, width // 2)[None]),
            )
        )

    queries_weight = np.asarray(weights[f"{_PREFIX}.queries_embedder.weight"], dtype=np.float32)
    features_weight = np.asarray(weights[f"{_PREFIX}.queries_features.weight"], dtype=np.float32)
    expected = (config.num_queries, width)
    if queries_weight.shape != expected or features_weight.shape != expected:
        raise ValueError("mask2former query embeddings do not match num_queries and hidden_dim")
    query_position = g.constant(network, queries_weight[None])
    hidden = g.constant(network, features_weight[None])

    mask_shape = g.shape(mask_features)[2:]
    flat_masks = g.reshape(
        network, mask_features, (1, g.shape(mask_features)[1], mask_shape[0] * mask_shape[1])
    )
    normed = _layer_norm(network, hidden, weights, f"{_PREFIX}.decoder.layernorm")
    logits = _mask_logits(network, normed, flat_masks, weights)
    layers = config.decoder_layers - 1
    for index in range(layers):
        level = index % FEATURE_LEVELS
        excluded = _attention_mask(network, logits, mask_shape, sizes[level])
        prefix = f"{_PREFIX}.decoder.layers.{index}"
        attended = _cross_attention(
            network,
            hidden,
            query_position,
            memories[level],
            keys[level],
            excluded,
            weights,
            f"{prefix}.cross_attn",
            heads,
        )
        hidden = _layer_norm(
            network, g.add(network, hidden, attended), weights, f"{prefix}.cross_attn_layer_norm"
        )
        attended = _self_attention(
            network, hidden, query_position, weights, f"{prefix}.self_attn", heads
        )
        hidden = _layer_norm(
            network, g.add(network, hidden, attended), weights, f"{prefix}.self_attn_layer_norm"
        )
        inner = g.relu(
            network,
            g.linear(
                network, hidden, weights[f"{prefix}.fc1.weight"], weights[f"{prefix}.fc1.bias"]
            ),
        )
        inner = g.linear(
            network, inner, weights[f"{prefix}.fc2.weight"], weights[f"{prefix}.fc2.bias"]
        )
        hidden = _layer_norm(
            network, g.add(network, hidden, inner), weights, f"{prefix}.final_layer_norm"
        )
        normed = _layer_norm(network, hidden, weights, f"{_PREFIX}.decoder.layernorm")
        logits = _mask_logits(network, normed, flat_masks, weights)

    classes = config.num_labels
    class_logits = g.linear(
        network, normed, weights["class_predictor.weight"], weights["class_predictor.bias"]
    )
    if g.shape(class_logits)[-1] != classes + 1:
        raise ValueError(
            "mask2former class_predictor does not match id2label plus the no-object class"
        )
    probabilities = g.slice_static(
        network, g.softmax(network, class_logits, 2), (0, 0, 0), (1, config.num_queries, classes)
    )
    size = SEMANTIC_SCORE_SIZE
    masks = g.resize_bilinear(
        network, g.reshape(network, logits, (1, config.num_queries) + tuple(mask_shape)), size, size
    )
    masks = g.reshape(network, g.sigmoid(network, masks), (1, config.num_queries, size * size))
    scores = g.matmul(network, probabilities, masks, transpose_left=True)
    return g.reshape(network, scores, (1, classes, size, size))
