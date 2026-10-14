# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BERT text tower for Grounding DINO.

A stock BERT-base encoder - 12 post-norm layers, hidden 768, FFN 3072 - then a
single linear projection down to the model width of 256.

It happens to share the tensor vocabulary the Swin backbone uses
(``attention.self.{query,key,value}``, ``intermediate.dense``,
``output.dense``), so the same helpers apply.

**It is not fed like a stock BERT, and that is the whole trick here.** Before
the text tower runs, the model derives two things from the token ids alone
(``generate_masks_with_special_tokens_and_transfer_map``):

* a **block-diagonal self-attention mask**: the prompt is a list of phrases
  separated by special tokens (``[CLS]``, ``[SEP]``, ``.``, ``?``), and a token
  may attend only inside its own phrase. Special tokens attend to themselves
  only. Letting the whole prompt attend to itself - the obvious reading - is
  shape-identical and scores 0.84 at the first layer;
* **position ids that restart at every phrase**, measured as
  ``index - previous_special - 1`` clamped at zero. For
  ``"a cat. a remote control."`` that is ``[0, 0, 1, 0, 0, 1, 2, 0, 0]``, not
  ``0..8``.

Two ordinary BERT details also matter because each produces a same-shaped
wrong answer:

* the embedding is the **sum of three** tables - word, position and token type
  - normalised together afterwards, not the word table alone;
* BERT is **post-norm**: the layer norm follows the residual add, where a
  pre-norm transformer would put it before the sublayer.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from . import graph as g

_LAYER_NORM_EPS = 1e-12
TEXT_PREFIX = "model.text_backbone"
# Saturating but fp16-safe; the reference uses the dtype minimum, and any value
# this far below the row maximum drives the softmax term to zero either way.
_MASK_FILL = -10000.0
# [CLS], [SEP], "." and "?" - the phrase delimiters.
SPECIAL_TOKENS = (101, 102, 1012, 1029)


def block_masks_and_positions(input_ids):
    """Return ``(block_mask, position_ids)`` derived from the token ids alone.

    Ported from ``generate_masks_with_special_tokens_and_transfer_map``: tokens
    sharing the same *next* special token form a phrase and attend to each
    other; every token also attends to itself. Phrases whose delimiter sits at
    the very start or end are excluded, which is what isolates ``[CLS]`` and
    ``[SEP]``.
    """
    tokens = np.asarray(input_ids, dtype=np.int64).reshape(-1)
    length = tokens.shape[0]
    special = np.isin(tokens, np.asarray(SPECIAL_TOKENS, dtype=np.int64))
    indices = np.arange(length, dtype=np.int64)

    previous = np.maximum.accumulate(np.where(special, indices, -1))
    following = np.flip(np.minimum.accumulate(
        np.flip(np.where(special, indices, length))))

    valid = (following != 0) & (following != length - 1) & (following != length)
    mask = (following[:, None] == following[None, :]) & valid[None, :]
    mask = mask | np.eye(length, dtype=bool)

    positions = np.where(valid, indices - previous - 1, 0)
    return mask, np.clip(positions, 0, None).astype(np.int64)


def _embed(network, weights, input_ids, token_type_ids, position_ids, prefix, dtype):
    """Word + position + token-type, summed, then normalised."""
    tokens = np.asarray(input_ids, dtype=np.int64).reshape(-1)

    word = np.asarray(weights[f"{prefix}.embeddings.word_embeddings.weight"])[tokens]
    # NOT the plain 0..n-1 range: positions restart at each phrase.
    position = np.asarray(
        weights[f"{prefix}.embeddings.position_embeddings.weight"])[
            np.asarray(position_ids, dtype=np.int64).reshape(-1)]
    types = np.asarray(token_type_ids, dtype=np.int64).reshape(-1)
    token_type = np.asarray(
        weights[f"{prefix}.embeddings.token_type_embeddings.weight"])[types]

    summed = (word + position + token_type).astype(dtype).reshape(1, tokens.shape[0], -1)
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
    block, position_ids = block_masks_and_positions(input_ids)
    hidden = _embed(network, weights, input_ids, token_type_ids, position_ids, prefix, dtype)

    # The phrase block mask and the padding mask combine into one additive
    # (1, length, length) term, broadcast over heads.
    keep = block.astype(np.float32)
    if attention_mask is not None:
        padding = np.asarray(attention_mask, dtype=np.float32).reshape(1, -1)
        keep = keep * padding
    additive = ((1.0 - keep) * _MASK_FILL).astype(dtype).reshape(1, *keep.shape)
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
