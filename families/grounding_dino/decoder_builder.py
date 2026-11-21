# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grounding DINO decoder: six layers over 900 queries, refining boxes.

Each layer is four post-norm blocks:

    self-attention  ->  text cross-attention  ->  deformable cross-attention
    ->  feed-forward

Four rules, all read from the reference and all shape-invariant:

* **the query position embedding is derived, not learned.** Every layer takes
  the *current* reference boxes, sine-embeds them at 128 features per
  coordinate - 512 for a 4-vector - and pushes that through a two-layer
  ``reference_points_head`` to the model width. It therefore changes as the
  boxes refine. ``families/deformable_detr`` uses a fixed learned table
  instead, and borrowing that is the natural mistake here;
* **references are four-dimensional, so deformable attention takes the box
  branch**: ``ref_xy + offsets / n_points * ref_wh * 0.5``. The centre-only
  branch belongs to Deformable DETR; cross-applying the two scores about 0.45;
* **boxes refine between layers.** After each layer the shared box head emits a
  delta that is added to ``logit(reference)`` and squashed back through a
  sigmoid. ``decoder_bbox_embed_share`` is true, so every layer uses
  ``bbox_embed.0``;
* **the logit uses an epsilon of 1e-5**, matching ``torch.special.logit``;
  clamping differently moves every refined box.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from . import graph as g
from .deformable_attention import build_deformable_attention

_LAYER_NORM_EPS = 1e-5
_LOGIT_EPS = 1e-5


def sine_box_embedding(network, boxes, features=128, temperature=10000.0,
                       scale=2.0 * np.pi, exchange_xy=True, dtype=np.float32):
    """Sine-embed ``(1, queries, 4)`` boxes into ``(1, queries, 4 * features)``.

    Built as graph ops rather than folded to a constant, because the boxes are
    a tensor that refines between layers.
    """
    queries = int(boxes.shape[1])
    coords = int(boxes.shape[2])

    scaled = network.add_elementwise(
        boxes, g.add_constant(network, (1, 1, 1), np.full((1,), scale), dtype=dtype),
        trt.ElementWiseOperation.PROD).get_output(0)
    widened = network.add_shuffle(scaled)
    widened.reshape_dims = (1, queries, coords, 1)

    index = np.arange(features, dtype=np.float64)
    dim_t = (temperature ** (2.0 * np.floor(index / 2.0) / features)).reshape(1, 1, 1, features)
    raw = network.add_elementwise(
        widened.get_output(0), g.add_constant(network, dim_t.shape, dim_t, dtype=dtype),
        trt.ElementWiseOperation.DIV).get_output(0)

    half = features // 2
    def half_slice(start, op):
        sliced = network.add_slice(raw, (0, 0, 0, start), (1, queries, coords, half),
                                   (1, 1, 1, 2)).get_output(0)
        out = network.add_unary(sliced, op).get_output(0)
        shuffle = network.add_shuffle(out)
        shuffle.reshape_dims = (1, queries, coords, half, 1)
        return shuffle.get_output(0)

    # sin on the even indices, cos on the odd ones, interleaved.
    interleaved = g.concat(
        network, [half_slice(0, trt.UnaryOperation.SIN),
                  half_slice(1, trt.UnaryOperation.COS)], axis=4)
    packed = network.add_shuffle(interleaved)
    packed.reshape_dims = (1, queries, coords, features)
    blocks = packed.get_output(0)

    # ``exchange_xy`` swaps the first two coordinate blocks, so the y embedding
    # precedes the x one. It defaults to True in the reference and the decoder
    # does not override it.
    order = list(range(coords))
    if exchange_xy and coords >= 2:
        order[0], order[1] = order[1], order[0]
    pieces = [network.add_slice(blocks, (0, 0, i, 0), (1, queries, 1, features),
                                (1, 1, 1, 1)).get_output(0) for i in order]
    flat = network.add_shuffle(g.concat(network, pieces, axis=2))
    flat.reshape_dims = (1, queries, coords * features)
    return flat.get_output(0)


def _logit(network, x, dtype):
    """``torch.special.logit`` with the reference's 1e-5 epsilon.

    Always evaluated in fp32, whatever the engine's working type. In fp16
    ``1 - 1e-5`` rounds to exactly 1.0, so the upper clamp leaves a complement
    of zero and the division returns ``+inf`` - which then propagates through
    iterative box refinement into the reference points and the final head.
    Measured, not assumed: an fp16 engine built with the clamp in fp16 scored
    a logits cosine of -0.95 against the reference and put every one of the
    900 queries above the detection threshold.
    """
    source = x if x.dtype == trt.float32 else network.add_cast(x, trt.float32).get_output(0)
    low = g.add_constant(network, (1, 1, 1), np.full((1,), _LOGIT_EPS), dtype=np.float32)
    high = g.add_constant(network, (1, 1, 1), np.full((1,), 1.0 - _LOGIT_EPS), dtype=np.float32)
    one = g.add_constant(network, (1, 1, 1), np.full((1,), 1.0), dtype=np.float32)
    clamped = network.add_elementwise(source, low, trt.ElementWiseOperation.MAX).get_output(0)
    clamped = network.add_elementwise(clamped, high, trt.ElementWiseOperation.MIN).get_output(0)
    complement = network.add_elementwise(
        one, clamped, trt.ElementWiseOperation.SUB).get_output(0)
    ratio = network.add_elementwise(
        clamped, complement, trt.ElementWiseOperation.DIV).get_output(0)
    result = network.add_unary(ratio, trt.UnaryOperation.LOG).get_output(0)
    target = trt.float16 if dtype == np.float16 else trt.float32
    return result if target == trt.float32 else network.add_cast(result, target).get_output(0)


def _attention(network, query_source, key_source, weights, prefix, heads, dtype,
               value_source=None, mask=None):
    """Multi-head attention; q and k from ``query_source``/``key_source``."""
    q_len = int(query_source.shape[1])
    k_len = int(key_source.shape[1])
    width = int(query_source.shape[2])
    head_dim = width // heads
    values = key_source if value_source is None else value_source

    def project(source, name, length):
        out = g.add_linear(network, source, weights[f"{prefix}.{name}.weight"],
                           weights[f"{prefix}.{name}.bias"], dtype=dtype)
        shuffle = network.add_shuffle(out)
        shuffle.reshape_dims = (length, heads, head_dim)
        shuffle.second_transpose = (1, 0, 2)
        return shuffle.get_output(0)

    query = project(query_source, "query", q_len)
    key = project(key_source, "key", k_len)
    value = project(values, "value", k_len)

    scores = network.add_matrix_multiply(
        query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE).get_output(0)
    scores = network.add_elementwise(
        scores, g.add_constant(network, (1, 1, 1), np.full((1,), 1.0 / math.sqrt(head_dim)),
                               dtype=dtype),
        trt.ElementWiseOperation.PROD).get_output(0)
    if mask is not None:
        scores = network.add_elementwise(
            scores, mask, trt.ElementWiseOperation.SUM).get_output(0)
    softmax = network.add_softmax(scores)
    softmax.axes = 1 << 2
    context = network.add_matrix_multiply(
        softmax.get_output(0), trt.MatrixOperation.NONE, value,
        trt.MatrixOperation.NONE).get_output(0)
    merged = network.add_shuffle(context)
    merged.first_transpose = (1, 0, 2)
    merged.reshape_dims = (1, q_len, width)
    return g.add_linear(network, merged.get_output(0),
                        weights[f"{prefix}.out_proj.weight"],
                        weights[f"{prefix}.out_proj.bias"], dtype=dtype)


def _norm(network, residual, delta, weights, prefix, dtype):
    return g.add_layer_norm(
        network, g.add_sum(network, residual, delta),
        weights[f"{prefix}.weight"], weights[f"{prefix}.bias"],
        _LAYER_NORM_EPS, dtype=dtype)


def build_decoder(network, target, reference, memory, text, weights, shapes, config,
                  prefix="model.decoder", dtype=np.float32, text_mask=None, debug=None):
    """Run the six decoder layers, refining the boxes as it goes.

    ``reference`` is the ``(1, queries, 4)`` initial box tensor. It is carried
    as a graph tensor because every layer refines it and the query position
    embedding is derived from the current value.
    """
    hidden = target
    boxes = reference
    heads = config["decoder_attention_heads"]
    levels = len(shapes)

    entering = boxes
    for layer in range(config["decoder_layers"]):
        base = f"{prefix}.layers.{layer}"
        # The final head scores against the reference ENTERING the last layer,
        # not the one the last layer produces.
        entering = boxes

        # Query position comes from the CURRENT boxes, through a small MLP.
        position = sine_box_embedding(network, boxes, config["d_model"] // 2, dtype=dtype)
        for index in (0, 1):
            position = g.add_linear(
                network, position,
                weights[f"{prefix}.reference_points_head.layers.{index}.weight"],
                weights[f"{prefix}.reference_points_head.layers.{index}.bias"], dtype=dtype)
            if index == 0:
                position = g.add_relu(network, position)

        posed = network.add_elementwise(
            hidden, position, trt.ElementWiseOperation.SUM).get_output(0)
        attended = _attention(network, posed, posed, weights, f"{base}.self_attn",
                              heads, dtype, value_source=hidden)
        hidden = _norm(network, hidden, attended, weights,
                       f"{base}.self_attn_layer_norm", dtype)

        posed = network.add_elementwise(
            hidden, position, trt.ElementWiseOperation.SUM).get_output(0)
        crossed = _attention(network, posed, text, weights, f"{base}.encoder_attn_text",
                             heads, dtype, mask=text_mask)
        hidden = _norm(network, hidden, crossed, weights,
                       f"{base}.encoder_attn_text_layer_norm", dtype)

        posed = network.add_elementwise(
            hidden, position, trt.ElementWiseOperation.SUM).get_output(0)
        # Four-dimensional references: the box branch, not the centre-only one.
        widen = network.add_shuffle(boxes)
        widen.reshape_dims = (1, int(boxes.shape[1]), 1, 4)
        reference_tensor = g.concat(network, [widen.get_output(0)] * levels, axis=2)
        deformed = build_deformable_attention(
            network, hidden, posed, reference_tensor, weights, f"{base}.encoder_attn",
            shapes, heads, config["decoder_n_points"], dtype, value_source=memory)
        hidden = _norm(network, hidden, deformed, weights,
                       f"{base}.encoder_attn_layer_norm", dtype)

        inner = g.add_relu(network, g.add_linear(
            network, hidden, weights[f"{base}.fc1.weight"],
            weights[f"{base}.fc1.bias"], dtype=dtype))
        inner = g.add_linear(network, inner, weights[f"{base}.fc2.weight"],
                             weights[f"{base}.fc2.bias"], dtype=dtype)
        hidden = _norm(network, hidden, inner, weights,
                       f"{base}.final_layer_norm", dtype)

        # Iterative refinement: the shared head emits a delta in logit space.
        delta = hidden
        for index in range(config["box_head_layers"]):
            delta = g.add_linear(network, delta,
                                 weights[f"bbox_embed.0.layers.{index}.weight"],
                                 weights[f"bbox_embed.0.layers.{index}.bias"], dtype=dtype)
            if index + 1 < config["box_head_layers"]:
                delta = g.add_relu(network, delta)
        refined = network.add_elementwise(
            delta, _logit(network, boxes, dtype), trt.ElementWiseOperation.SUM).get_output(0)
        boxes = network.add_activation(refined, trt.ActivationType.SIGMOID).get_output(0)

        if debug is not None:
            debug[f"hidden_{layer}"] = hidden
            debug[f"boxes_{layer}"] = boxes
    return hidden, boxes, entering


def build_heads(network, hidden, entering, text, weights, config,
                prefix="model.decoder", dtype=np.float32, text_mask=None):
    """Final contrastive logits and boxes from the last decoder layer.

    Two inputs that look interchangeable and are not:

    * the head reads the decoder's **layer-normed** output
      (``intermediate_hidden_states`` are ``decoder.layer_norm(hidden)``),
      while the in-loop refinement reads the raw layer output. The same box
      head therefore sees different tensors in the two places;
    * the reference it adds is the box **entering** the last layer, not the
      one that layer refined into.
    """
    normed = g.add_layer_norm(
        network, hidden, weights[f"{prefix}.layer_norm.weight"],
        weights[f"{prefix}.layer_norm.bias"], _LAYER_NORM_EPS, dtype=dtype)

    queries = int(normed.shape[1])
    logits = network.add_matrix_multiply(
        normed, trt.MatrixOperation.NONE, text, trt.MatrixOperation.TRANSPOSE).get_output(0)
    # The reference fills every position outside the prompt with -inf, both
    # the padded tail and, when the prompt is shorter, the concatenated rest.
    if text_mask is not None:
        logits = network.add_elementwise(
            logits, text_mask, trt.ElementWiseOperation.SUM).get_output(0)
    pad = config["max_text_len"] - int(text.shape[1])
    if pad > 0:
        filler = np.full((1, queries, pad), -np.inf, dtype=dtype)
        logits = g.concat(
            network, [logits, g.add_constant(network, filler.shape, filler, dtype)], axis=2)

    delta = normed
    for index in range(config["box_head_layers"]):
        delta = g.add_linear(network, delta,
                             weights[f"bbox_embed.0.layers.{index}.weight"],
                             weights[f"bbox_embed.0.layers.{index}.bias"], dtype=dtype)
        if index + 1 < config["box_head_layers"]:
            delta = g.add_relu(network, delta)
    coords = network.add_elementwise(
        delta, _logit(network, entering, dtype), trt.ElementWiseOperation.SUM).get_output(0)
    boxes = network.add_activation(coords, trt.ActivationType.SIGMOID).get_output(0)
    return logits, boxes
