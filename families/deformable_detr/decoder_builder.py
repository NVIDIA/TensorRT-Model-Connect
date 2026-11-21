# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deformable DETR decoder and prediction heads.

Six post-norm layers, each self-attention then deformable cross-attention then
an MLP. The rules worth stating, all read from the reference:

* **Queries are posed, values are not** - in both attentions. Self-attention
  takes q and k from ``hidden + query_embed`` and v from the raw hidden states;
  cross-attention takes its offsets and weights from the posed tensor and its
  values from the encoder output.
* **The query embedding is one tensor split in half.**
  ``query_position_embeddings.weight`` is ``(300, 512)``: the first 256 columns
  are the positional half, the second 256 are the initial content. Swapping the
  halves is shape-identical and wrong.
* **Reference points are fixed across layers.** This checkpoint has
  ``with_box_refine=False``, so ``decoder.bbox_embed`` is ``None`` and nothing
  updates the references between layers. The iterative-refinement path belongs
  to the other variants.
* **The box head adds the reference to the first two coordinates only**, in
  inverse-sigmoid space, before the final sigmoid.

Because the query embedding is a constant, the initial content, the reference
points and their inverse sigmoid are all computed once in numpy at build time.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import graph as g
from .deformable_attention import build_deformable_attention

_NORM_EPS = 1e-5
_SIGMOID_EPS = 1e-5


def _inverse_sigmoid(x, eps=_SIGMOID_EPS):
    x = np.clip(x, 0.0, 1.0)
    return np.log(np.clip(x, eps, None) / np.clip(1.0 - x, eps, None))


def initial_queries(weights, prefix="model"):
    """Return ``(query_embed, target, reference_points)`` as numpy constants."""
    table = weights[f"{prefix}.query_position_embeddings.weight"]
    width = table.shape[1] // 2
    query_embed = table[:, :width]
    target = table[:, width:]
    logits = query_embed @ weights[f"{prefix}.reference_points.weight"].T \
        + weights[f"{prefix}.reference_points.bias"]
    reference = 1.0 / (1.0 + np.exp(-logits))
    return query_embed[None], target[None], reference[None]


def _self_attention(network, hidden, posed, weights, prefix, heads, dtype):
    """Standard multi-head self-attention; q and k posed, v not."""
    queries = int(hidden.shape[1])
    width = int(hidden.shape[2])
    head_dim = width // heads
    scaling = float(head_dim) ** -0.5

    def project(source, name):
        out = g.add_linear(network, source, weights[f"{prefix}.{name}.weight"],
                           weights[f"{prefix}.{name}.bias"], dtype)
        shuffle = network.add_shuffle(out)
        shuffle.reshape_dims = (queries, heads, head_dim)
        shuffle.second_transpose = (1, 0, 2)
        return shuffle.get_output(0)

    query = project(posed, "q_proj")
    key = project(posed, "k_proj")
    value = project(hidden, "v_proj")

    scores = network.add_matrix_multiply(
        query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE).get_output(0)
    scale = g.add_constant(network, (1, 1, 1), np.array([scaling], dtype=dtype), dtype)
    scores = network.add_elementwise(
        scores, scale, trt.ElementWiseOperation.PROD).get_output(0)
    softmax = network.add_softmax(scores)
    softmax.axes = 1 << 2
    attended = network.add_matrix_multiply(
        softmax.get_output(0), trt.MatrixOperation.NONE,
        value, trt.MatrixOperation.NONE).get_output(0)

    shuffle = network.add_shuffle(attended)
    shuffle.first_transpose = (1, 0, 2)
    shuffle.reshape_dims = (1, queries, width)
    return g.add_linear(network, shuffle.get_output(0),
                        weights[f"{prefix}.o_proj.weight"],
                        weights[f"{prefix}.o_proj.bias"], dtype)


def build_decoder(network, memory, weights, shapes, config,
                  prefix="model.decoder", dtype=np.float32):
    """Return the final decoder hidden states and the baked reference points."""
    query_embed, target, reference = initial_queries(weights)
    levels = len(shapes)

    hidden = g.add_constant(network, target.shape, target.astype(dtype), dtype)
    position = g.add_constant(network, query_embed.shape, query_embed.astype(dtype), dtype)
    # Every level shares the query's reference point, which is what valid
    # ratios of 1 reduce to.
    tiled = np.tile(reference[:, :, None, :], (1, 1, levels, 1)).astype(dtype)
    reference_tensor = g.add_constant(network, tiled.shape, tiled, dtype)

    for layer in range(config["decoder_layers"]):
        base = f"{prefix}.layers.{layer}"
        posed = network.add_elementwise(
            hidden, position, trt.ElementWiseOperation.SUM).get_output(0)
        attended = _self_attention(
            network, hidden, posed, weights, f"{base}.self_attn",
            config["decoder_attention_heads"], dtype)
        hidden = g.add_sum(network, hidden, attended)
        hidden = g.add_layer_norm(
            network, hidden, weights[f"{base}.self_attn_layer_norm.weight"],
            weights[f"{base}.self_attn_layer_norm.bias"], _NORM_EPS, dtype)

        posed = network.add_elementwise(
            hidden, position, trt.ElementWiseOperation.SUM).get_output(0)
        crossed = build_deformable_attention(
            network, hidden, posed, reference_tensor, weights, f"{base}.encoder_attn",
            shapes, config["decoder_attention_heads"], config["decoder_n_points"],
            dtype, value_source=memory)
        hidden = g.add_sum(network, hidden, crossed)
        hidden = g.add_layer_norm(
            network, hidden, weights[f"{base}.encoder_attn_layer_norm.weight"],
            weights[f"{base}.encoder_attn_layer_norm.bias"], _NORM_EPS, dtype)

        inner = g.add_linear(network, hidden, weights[f"{base}.mlp.fc1.weight"],
                             weights[f"{base}.mlp.fc1.bias"], dtype)
        inner = g.add_relu(network, inner)
        inner = g.add_linear(network, inner, weights[f"{base}.mlp.fc2.weight"],
                             weights[f"{base}.mlp.fc2.bias"], dtype)
        hidden = g.add_sum(network, hidden, inner)
        hidden = g.add_layer_norm(
            network, hidden, weights[f"{base}.final_layer_norm.weight"],
            weights[f"{base}.final_layer_norm.bias"], _NORM_EPS, dtype)

    return hidden, reference


def build_heads(network, hidden, reference, weights, config, dtype=np.float32):
    """Class logits and boxes from the last decoder layer.

    ``with_box_refine`` is false for this checkpoint, so every layer shares one
    head; index 0 and index 5 name the same weights.
    """
    logits = g.add_linear(network, hidden, weights["class_embed.0.weight"],
                          weights["class_embed.0.bias"], dtype)

    delta = hidden
    layers = config["box_head_layers"]
    for index in range(layers):
        delta = g.add_linear(network, delta,
                             weights[f"bbox_embed.0.layers.{index}.weight"],
                             weights[f"bbox_embed.0.layers.{index}.bias"], dtype)
        if index + 1 < layers:
            delta = g.add_relu(network, delta)

    # The reference enters in inverse-sigmoid space, and only on x and y.
    pad = np.zeros((1, reference.shape[1], 4), dtype=dtype)
    pad[:, :, :2] = _inverse_sigmoid(reference)
    offset = g.add_constant(network, pad.shape, pad, dtype)
    boxes = network.add_elementwise(delta, offset, trt.ElementWiseOperation.SUM).get_output(0)
    boxes = network.add_activation(boxes, trt.ActivationType.SIGMOID).get_output(0)
    return logits, boxes
