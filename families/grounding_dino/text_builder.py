# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BERT text tower for Grounding DINO.

A stock BERT-base encoder - 12 post-norm layers, hidden 768, FFN 3072 - then a
single linear projection down to the model width of 256.

It happens to share the tensor vocabulary the Swin backbone uses
(``attention.self.{query,key,value}``, ``intermediate.dense``,
``output.dense``), so the same helpers apply. Two details are worth stating
because they are easy to get wrong and produce a same-shaped answer:

* the embedding is the **sum of three** tables - word, position and token type
  - normalised together afterwards, not the word table alone;
* BERT is **post-norm**: the layer norm follows the residual add, where a
  pre-norm transformer would put it before the sublayer. Swapping the two
  still builds and still trains-looking output.

The attention mask arrives as 1 for real tokens and 0 for padding, and enters
the scores additively as ``(1 - mask) * -10000`` - the value the reference
uses, not negative infinity.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from . import graph as g

_LAYER_NORM_EPS = 1e-12
TEXT_PREFIX = "model.text_backbone"
_MASK_FILL = -10000.0


def _embed(network, weights, input_ids, token_type_ids, prefix, dtype):
    """Word + position + token-type, summed, then normalised."""
    tokens = np.asarray(input_ids, dtype=np.int64).reshape(-1)
    length = tokens.shape[0]

    word = np.asarray(weights[f"{prefix}.embeddings.word_embeddings.weight"])[tokens]
    # Positions are the plain 0..n-1 range; this model has no offset.
    position = np.asarray(
        weights[f"{prefix}.embeddings.position_embeddings.weight"])[:length]
    types = np.asarray(token_type_ids, dtype=np.int64).reshape(-1)
    token_type = np.asarray(
        weights[f"{prefix}.embeddings.token_type_embeddings.weight"])[types]

    summed = (word + position + token_type).astype(dtype).reshape(1, length, -1)
    embedded = g.add_constant(network, summed.shape, summed, dtype)
    return g.add_layer_norm(
        network, embedded,
        weights[f"{prefix}.embeddings.LayerNorm.weight"],
        weights[f"{prefix}.embeddings.LayerNorm.bias"], _LAYER_NORM_EPS, dtype=dtype)


def _self_attention(network, hidden, mask, weights, prefix, heads, dtype):
    shape = tuple(int(v) for v in hidden.shape)
    length, width = shape[1], shape[2]
    head_dim = width // heads

    def project(name):
        out = g.add_linear(network, hidden,
                           weights[f"{prefix}.attention.self.{name}.weight"],
                           weights[f"{prefix}.attention.self.{name}.bias"], dtype=dtype)
        shuffle = network.add_shuffle(out)
        shuffle.reshape_dims = (length, heads, head_dim)
        shuffle.second_transpose = (1, 0, 2)
        return shuffle.get_output(0)

    query, key, value = project("query"), project("key"), project("value")
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
    merged.reshape_dims = (1, length, width)
    return g.add_linear(network, merged.get_output(0),
                        weights[f"{prefix}.attention.output.dense.weight"],
                        weights[f"{prefix}.attention.output.dense.bias"], dtype=dtype)


def _layer(network, hidden, mask, weights, prefix, heads, dtype):
    attended = _self_attention(network, hidden, mask, weights, prefix, heads, dtype)
    # Post-norm: add first, normalise second.
    hidden = g.add_layer_norm(
        network, g.add_sum(network, hidden, attended),
        weights[f"{prefix}.attention.output.LayerNorm.weight"],
        weights[f"{prefix}.attention.output.LayerNorm.bias"], _LAYER_NORM_EPS, dtype=dtype)

    inner = g.add_gelu(network, g.add_linear(
        network, hidden, weights[f"{prefix}.intermediate.dense.weight"],
        weights[f"{prefix}.intermediate.dense.bias"], dtype=dtype))
    inner = g.add_linear(network, inner, weights[f"{prefix}.output.dense.weight"],
                         weights[f"{prefix}.output.dense.bias"], dtype=dtype)
    return g.add_layer_norm(
        network, g.add_sum(network, hidden, inner),
        weights[f"{prefix}.output.LayerNorm.weight"],
        weights[f"{prefix}.output.LayerNorm.bias"], _LAYER_NORM_EPS, dtype=dtype)


def build_text_encoder(network, weights, cfg, input_ids, token_type_ids, attention_mask,
                       prefix=TEXT_PREFIX, dtype=np.float32, debug=None):
    """Return the BERT hidden states, before the projection to the model width."""
    hidden = _embed(network, weights, input_ids, token_type_ids, prefix, dtype)

    mask = None
    if attention_mask is not None:
        keep = np.asarray(attention_mask, dtype=np.float32).reshape(1, 1, -1)
        additive = ((1.0 - keep) * _MASK_FILL).astype(dtype)
        mask = g.add_constant(network, additive.shape, additive, dtype)

    for layer in range(cfg["num_hidden_layers"]):
        hidden = _layer(network, hidden, mask, weights,
                        f"{prefix}.encoder.layer.{layer}",
                        cfg["num_attention_heads"], dtype)
        if debug is not None:
            debug[f"text_layer_{layer}"] = hidden
    return hidden


def project_text(network, hidden, weights, dtype=np.float32):
    """BERT width (768) down to the detection model's width (256)."""
    return g.add_linear(network, hidden, weights["model.text_projection.weight"],
                        weights["model.text_projection.bias"], dtype=dtype)
