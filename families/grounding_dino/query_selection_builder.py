# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two-stage query selection and the contrastive classification head.

``families/deformable_detr`` rejects the two-stage path outright; this
checkpoint requires it, so the whole block is new.

Every encoder token proposes a box. The proposals are scored against the text
by a contrastive head - there is no class layer - and the 900 best become the
decoder's reference points.

The rules worth stating, each read from the reference:

* **proposal boxes are a fixed grid, not learned.** Centre ``(i + 0.5) / size``
  per level, with a width and height of ``0.05 * 2 ** level``. They enter in
  inverse-sigmoid space;
* **border proposals are invalidated.** A proposal is valid only if every
  coordinate lies strictly inside ``(0.01, 0.99)``, tested on the raw box
  *before* the inverse sigmoid. The first and last row and column of the
  finest level fall outside - ``0.5 / 100 = 0.005`` - so they are dropped.
  Invalid entries are filled with **positive** infinity in the proposals and
  zero in the query embedding, which is easy to get backwards;
* **the classifier is a dot product against the text tokens**, padded out to
  ``max_text_len`` with negative infinity. The 900 queries are picked by the
  per-token maximum over that padded axis;
* **the decoder's content queries are the learned embedding table**, not the
  gathered proposals. ``embedding_init_target`` is true for this checkpoint;
  the gather path belongs to the variants where it is false.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import graph as g

_LAYER_NORM_EPS = 1e-5
_VALID_LOW = 0.01
_VALID_HIGH = 0.99


def encoder_proposals(shapes, dtype=np.float32):
    """Return ``(proposals, valid)`` for every encoder token.

    ``proposals`` is ``(1, tokens, 4)`` in inverse-sigmoid space with invalid
    entries set to ``+inf``; ``valid`` is the ``(1, tokens, 1)`` mask.
    """
    boxes = []
    for level, (height, width) in enumerate(shapes):
        grid_y, grid_x = np.meshgrid(
            np.arange(height, dtype=np.float64), np.arange(width, dtype=np.float64),
            indexing="ij")
        centre_x = (grid_x + 0.5) / width
        centre_y = (grid_y + 0.5) / height
        side = np.full_like(centre_x, 0.05 * (2.0 ** level))
        boxes.append(np.stack((centre_x, centre_y, side, side), axis=-1).reshape(-1, 4))
    raw = np.concatenate(boxes, axis=0)[None]

    valid = ((raw > _VALID_LOW) & (raw < _VALID_HIGH)).all(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        logits = np.log(raw / (1.0 - raw))
    logits = np.where(valid, logits, np.inf)
    return logits.astype(dtype), valid


def build_query_selection(network, vision, text, weights, config, shapes,
                          prefix="model", dtype=np.float32, text_mask=None):
    """Pick the decoder's reference points from the encoder's proposals.

    Returns ``(target, reference_points)``; ``target`` is the learned content
    query table and ``reference_points`` the gathered boxes after a sigmoid.
    """
    tokens = int(vision.shape[1])
    width = int(vision.shape[2])
    queries = config["num_queries"]

    proposals, valid = encoder_proposals(shapes, dtype)
    if proposals.shape[1] != tokens:
        raise ValueError("deformable proposal count does not match the token count")

    # Invalid positions contribute nothing to the query embedding.
    keep = g.add_constant(network, valid.shape, valid.astype(dtype), dtype)
    masked = network.add_elementwise(
        vision, keep, trt.ElementWiseOperation.PROD).get_output(0)
    query = g.add_linear(network, masked, weights[f"{prefix}.enc_output.weight"],
                         weights[f"{prefix}.enc_output.bias"], dtype=dtype)
    query = g.add_layer_norm(
        network, query, weights[f"{prefix}.enc_output_norm.weight"],
        weights[f"{prefix}.enc_output_norm.bias"], _LAYER_NORM_EPS, dtype=dtype)

    # Contrastive scores: a dot product against the text, padded to max_text_len.
    scores = network.add_matrix_multiply(
        query, trt.MatrixOperation.NONE, text, trt.MatrixOperation.TRANSPOSE).get_output(0)
    # Padded prompt positions score -inf, exactly as the reference fills them.
    if text_mask is not None:
        scores = network.add_elementwise(
            scores, text_mask, trt.ElementWiseOperation.SUM).get_output(0)
    text_len = int(text.shape[1])
    pad_width = config["max_text_len"] - text_len
    if pad_width > 0:
        filler = np.full((1, tokens, pad_width), -np.inf, dtype=dtype)
        scores = g.concat(
            network, [scores, g.add_constant(network, filler.shape, filler, dtype)], axis=2)

    # One score per token: the maximum over the padded text axis.
    best = network.add_reduce(
        scores, trt.ReduceOperation.MAX, 1 << 2, keep_dims=False).get_output(0)

    # The box each token proposes, in inverse-sigmoid space.
    delta = query
    for index in range(config["box_head_layers"]):
        delta = g.add_linear(
            network, delta,
            weights[f"{prefix}.encoder_output_bbox_embed.layers.{index}.weight"],
            weights[f"{prefix}.encoder_output_bbox_embed.layers.{index}.bias"], dtype=dtype)
        if index + 1 < config["box_head_layers"]:
            delta = g.add_relu(network, delta)
    grid = g.add_constant(network, proposals.shape, proposals, dtype)
    coords = network.add_elementwise(delta, grid, trt.ElementWiseOperation.SUM).get_output(0)

    # Top 900 tokens by score, then gather their boxes.
    topk = network.add_topk(best, trt.TopKOperation.MAX, queries, 1 << 1)
    indices = topk.get_output(1)
    gather = network.add_gather_v2(coords, indices, trt.GatherMode.DEFAULT)
    gather.axis = 1
    gather.num_elementwise_dims = 1
    reference = network.add_activation(
        gather.get_output(0), trt.ActivationType.SIGMOID).get_output(0)

    # The content queries are the learned table, not the gathered proposals.
    table = np.asarray(weights[f"{prefix}.query_position_embeddings.weight"],
                       dtype=dtype).reshape(1, queries, width)
    target = g.add_constant(network, table.shape, table, dtype)
    return target, reference


def build_contrastive_logits(network, hidden, text, config, dtype=np.float32, text_mask=None):
    """Query-by-text similarity, padded to ``max_text_len``."""
    scores = network.add_matrix_multiply(
        hidden, trt.MatrixOperation.NONE, text, trt.MatrixOperation.TRANSPOSE).get_output(0)
    if text_mask is not None:
        scores = network.add_elementwise(
            scores, text_mask, trt.ElementWiseOperation.SUM).get_output(0)
    queries = int(hidden.shape[1])
    pad_width = config["max_text_len"] - int(text.shape[1])
    if pad_width > 0:
        filler = np.full((1, queries, pad_width), -np.inf, dtype=dtype)
        scores = g.concat(
            network, [scores, g.add_constant(network, filler.shape, filler, dtype)], axis=2)
    return scores
