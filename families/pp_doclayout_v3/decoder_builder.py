# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned PP-DocLayoutV3 deformable decoder and layout heads."""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from . import graph as g

_LAYER_NORM_EPS = 1e-5
_GRID_OFFSET = 0.5
_GRID_SIZE = 0.05
_VALID_EPS = 1e-2


def build_anchors(shapes):
    """Anchor boxes in logit space, with invalid positions pushed to infinity."""
    levels = []
    for level, (height, width) in enumerate(shapes):
        grid_y, grid_x = np.meshgrid(
            np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij"
        )
        centres = np.stack([grid_x, grid_y], axis=-1).reshape(1, height * width, 2)
        centres = (centres + _GRID_OFFSET) / np.array([width, height], dtype=np.float32)
        sizes = np.ones_like(centres) * _GRID_SIZE * (2.0**level)
        levels.append(np.concatenate([centres, sizes], axis=-1))
    anchors = np.concatenate(levels, axis=1)
    valid = ((anchors > _VALID_EPS) & (anchors < 1 - _VALID_EPS)).all(-1, keepdims=True)
    logits = np.log(anchors / (1 - anchors))
    return np.where(valid, logits, np.inf).astype(np.float32)


def _mlp(network, x, weights, prefix, layers, dtype, activation="relu"):
    """The repeated Linear/ReLU head shape, with no activation on the last layer."""
    for index in range(layers):
        x = g.add_linear(
            network,
            x,
            weights[f"{prefix}.layers.{index}.weight"],
            weights[f"{prefix}.layers.{index}.bias"],
            dtype=dtype,
        )
        if index < layers - 1:
            x = g.add_relu(network, x) if activation == "relu" else g.add_silu(network, x)
    return x


def _deformable_attention(
    network, query, position, memory, shapes, references, weights, prefix, cfg, dtype, debug=None
):
    """Multi-scale deformable attention, built from grid sample.

    TensorRT's IGridSampleLayer reproduces torch's grid_sample exactly, so this
    needs no plugin. The sample mode must be FILL, which is torch's
    padding_mode="zeros"; CLAMP is "border" and silently differs wherever an
    offset lands outside the feature map, which happens constantly.

    ``memory`` is the flattened encoder output; the module projects it itself.
    """
    queries = cfg["num_queries"]
    heads = cfg["decoder_attention_heads"]
    levels = len(shapes)
    points = cfg["decoder_n_points"]
    hidden = cfg["d_model"]
    head_dim = hidden // heads
    scale = cfg["offset_scale"]

    value = g.add_linear(
        network,
        memory,
        weights[f"{prefix}.value_proj.weight"],
        weights[f"{prefix}.value_proj.bias"],
        dtype=dtype,
    )
    # The offsets and attention weights are predicted from the query *with* its
    # position embedding added; the value is projected from the unposed memory.
    posed = query if position is None else g.add_sum(network, query, position)
    offsets = g.add_linear(
        network,
        posed,
        weights[f"{prefix}.sampling_offsets.weight"],
        weights[f"{prefix}.sampling_offsets.bias"],
        dtype=dtype,
    )
    raw_weights = g.add_linear(
        network,
        posed,
        weights[f"{prefix}.attention_weights.weight"],
        weights[f"{prefix}.attention_weights.bias"],
        dtype=dtype,
    )

    shaped = network.add_shuffle(raw_weights)
    shaped.reshape_dims = (1, queries, heads, levels * points)
    softmax = network.add_softmax(shaped.get_output(0))
    softmax.axes = 1 << 3
    weight_grid = network.add_shuffle(softmax.get_output(0))
    weight_grid.reshape_dims = (1, queries, heads, levels, points)

    offset_grid = network.add_shuffle(offsets)
    offset_grid.reshape_dims = (1, queries, heads, levels, points, 2)

    # locations = ref_xy + offsets / points * ref_wh * offset_scale
    centre = network.add_slice(references, (0, 0, 0), (1, queries, 2), (1, 1, 1)).get_output(0)
    extent = network.add_slice(references, (0, 0, 2), (1, queries, 2), (1, 1, 1)).get_output(0)

    def broadcast(tensor):
        layer = network.add_shuffle(tensor)
        layer.reshape_dims = (1, queries, 1, 1, 1, 2)
        return layer.get_output(0)

    factor = g.add_constant(network, (1, 1, 1, 1, 1, 1), np.full((1,), scale / points), dtype=dtype)
    scaled = network.add_elementwise(
        offset_grid.get_output(0), factor, trt.ElementWiseOperation.PROD
    ).get_output(0)
    scaled = network.add_elementwise(
        scaled, broadcast(extent), trt.ElementWiseOperation.PROD
    ).get_output(0)
    locations = network.add_elementwise(
        scaled, broadcast(centre), trt.ElementWiseOperation.SUM
    ).get_output(0)

    # grid_sample wants coordinates in [-1, 1]
    two = g.add_constant(network, (1, 1, 1, 1, 1, 1), np.full((1,), 2.0), dtype=dtype)
    one = g.add_constant(network, (1, 1, 1, 1, 1, 1), np.full((1,), 1.0), dtype=dtype)
    grid = network.add_elementwise(
        network.add_elementwise(locations, two, trt.ElementWiseOperation.PROD).get_output(0),
        one,
        trt.ElementWiseOperation.SUB,
    ).get_output(0)

    if debug is not None:
        debug["grid"] = grid
    sampled = []
    start = 0
    for level, (height, width) in enumerate(shapes):
        count = height * width
        # [1, count, hidden] -> [heads, head_dim, height, width]
        chunk = network.add_slice(value, (0, start, 0), (1, count, hidden), (1, 1, 1)).get_output(0)
        start += count
        spatial = network.add_shuffle(chunk)
        spatial.reshape_dims = (count, heads, head_dim)
        spatial.second_transpose = (1, 2, 0)
        reshaped = network.add_shuffle(spatial.get_output(0))
        reshaped.reshape_dims = (heads, head_dim, height, width)

        level_grid = network.add_slice(
            grid, (0, 0, 0, level, 0, 0), (1, queries, heads, 1, points, 2), (1, 1, 1, 1, 1, 1)
        ).get_output(0)
        reorder = network.add_shuffle(level_grid)
        reorder.reshape_dims = (queries, heads, points, 2)
        reorder.second_transpose = (1, 0, 2, 3)
        sampler = network.add_grid_sample(reshaped.get_output(0), reorder.get_output(0))
        sampler.interpolation_mode = trt.InterpolationMode.LINEAR
        sampler.sample_mode = trt.SampleMode.FILL
        sampler.align_corners = False
        sampled.append(sampler.get_output(0))  # [heads, head_dim, queries, points]

    # Concatenate on points so the axis runs level-major, matching the weights.
    stacked = g.concat(network, sampled, axis=3)
    if debug is not None:
        debug["stacked"] = stacked
    flat_weights = network.add_shuffle(weight_grid.get_output(0))
    flat_weights.first_transpose = (0, 2, 1, 3, 4)
    flat_weights.reshape_dims = (heads, 1, queries, levels * points)
    weighted = network.add_elementwise(
        stacked, flat_weights.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)
    reduced = network.add_reduce(weighted, trt.ReduceOperation.SUM, 1 << 3, False).get_output(0)
    merged = network.add_shuffle(reduced)  # [heads, head_dim, queries]
    merged.first_transpose = (2, 0, 1)
    merged.reshape_dims = (1, queries, heads * head_dim)
    if debug is not None:
        debug["merged"] = merged.get_output(0)
    return g.add_linear(
        network,
        merged.get_output(0),
        weights[f"{prefix}.output_proj.weight"],
        weights[f"{prefix}.output_proj.bias"],
        dtype=dtype,
    )


def _decoder_layer(
    network, hidden, position, memory, shapes, references, weights, prefix, cfg, dtype
):
    """Self attention, deformable cross attention, feed forward. All post-norm."""
    queries = cfg["num_queries"]
    heads = cfg["decoder_attention_heads"]
    head_dim = cfg["d_model"] // heads

    # Self attention queries and keys carry the position embedding; values do not.
    posed = g.add_sum(network, hidden, position)
    attended = _self_attention_posed(
        network, posed, hidden, weights, f"{prefix}.self_attn", queries, heads, head_dim, dtype
    )
    hidden = g.add_layer_norm(
        network,
        g.add_sum(network, hidden, attended),
        weights[f"{prefix}.self_attn_layer_norm.weight"],
        weights[f"{prefix}.self_attn_layer_norm.bias"],
        _LAYER_NORM_EPS,
        dtype=dtype,
    )

    cross = _deformable_attention(
        network,
        hidden,
        position,
        memory,
        shapes,
        references,
        weights,
        f"{prefix}.encoder_attn",
        cfg,
        dtype,
    )
    hidden = g.add_layer_norm(
        network,
        g.add_sum(network, hidden, cross),
        weights[f"{prefix}.encoder_attn_layer_norm.weight"],
        weights[f"{prefix}.encoder_attn_layer_norm.bias"],
        _LAYER_NORM_EPS,
        dtype=dtype,
    )

    inner = g.add_relu(
        network,
        g.add_linear(
            network,
            hidden,
            weights[f"{prefix}.fc1.weight"],
            weights[f"{prefix}.fc1.bias"],
            dtype=dtype,
        ),
    )
    out = g.add_linear(
        network, inner, weights[f"{prefix}.fc2.weight"], weights[f"{prefix}.fc2.bias"], dtype=dtype
    )
    return g.add_layer_norm(
        network,
        g.add_sum(network, hidden, out),
        weights[f"{prefix}.final_layer_norm.weight"],
        weights[f"{prefix}.final_layer_norm.bias"],
        _LAYER_NORM_EPS,
        dtype=dtype,
    )


def _self_attention_posed(network, posed, values, weights, prefix, tokens, heads, head_dim, dtype):
    query = g.add_linear(
        network,
        posed,
        weights[f"{prefix}.q_proj.weight"],
        weights[f"{prefix}.q_proj.bias"],
        dtype=dtype,
    )
    key = g.add_linear(
        network,
        posed,
        weights[f"{prefix}.k_proj.weight"],
        weights[f"{prefix}.k_proj.bias"],
        dtype=dtype,
    )
    value = g.add_linear(
        network,
        values,
        weights[f"{prefix}.v_proj.weight"],
        weights[f"{prefix}.v_proj.bias"],
        dtype=dtype,
    )

    def split(tensor):
        layer = network.add_shuffle(tensor)
        layer.reshape_dims = (1, tokens, heads, head_dim)
        layer.second_transpose = (0, 2, 1, 3)
        return layer.get_output(0)

    scores = network.add_matrix_multiply(
        split(query), trt.MatrixOperation.NONE, split(key), trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    scaled = network.add_elementwise(
        scores,
        g.add_constant(
            network, (1, 1, 1, 1), np.full((1,), 1.0 / math.sqrt(head_dim)), dtype=dtype
        ),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    softmax = network.add_softmax(scaled)
    softmax.axes = 1 << 3
    context = network.add_matrix_multiply(
        softmax.get_output(0), trt.MatrixOperation.NONE, split(value), trt.MatrixOperation.NONE
    ).get_output(0)
    merge = network.add_shuffle(context)
    merge.first_transpose = (0, 2, 1, 3)
    merge.reshape_dims = (1, tokens, heads * head_dim)
    return g.add_linear(
        network,
        merge.get_output(0),
        weights[f"{prefix}.out_proj.weight"],
        weights[f"{prefix}.out_proj.bias"],
        dtype=dtype,
    )


def _inverse_sigmoid(network, x, dtype):
    one = g.add_constant(network, (1, 1, 1), np.full((1,), 1.0), dtype=dtype)
    complement = network.add_elementwise(one, x, trt.ElementWiseOperation.SUB).get_output(0)
    epsilon = g.add_constant(network, (1, 1, 1), [1e-5], dtype=dtype)
    numerator = network.add_elementwise(x, epsilon, trt.ElementWiseOperation.MAX).get_output(0)
    denominator = network.add_elementwise(
        complement, epsilon, trt.ElementWiseOperation.MAX
    ).get_output(0)
    ratio = network.add_elementwise(
        numerator, denominator, trt.ElementWiseOperation.DIV
    ).get_output(0)
    return network.add_unary(ratio, trt.UnaryOperation.LOG).get_output(0)


def _norm(network, x, weights):
    return g.add_layer_norm(
        network, x, weights["model.decoder_norm.weight"], weights["model.decoder_norm.bias"], 1e-5
    )


def _masks(network, x, features, weights):
    queries = _mlp(
        network, _norm(network, x, weights), weights, "model.mask_query_head", 3, np.float32
    )
    flatten = network.add_shuffle(features)
    flatten.reshape_dims = (
        1,
        int(features.shape[1]),
        int(features.shape[2]) * int(features.shape[3]),
    )
    masks = network.add_matrix_multiply(
        queries, trt.MatrixOperation.NONE, flatten.get_output(0), trt.MatrixOperation.NONE
    ).get_output(0)
    reshape = network.add_shuffle(masks)
    reshape.reshape_dims = (1, int(x.shape[1]), int(features.shape[2]), int(features.shape[3]))
    return reshape.get_output(0)


def _mask_boxes(network, masks):
    """Mask-enhanced initial references, including the empty-mask case."""
    queries, height, width = map(int, masks.shape[1:])
    zero = g.add_constant(network, (1, 1, 1, 1), [0.0])
    foreground = network.add_elementwise(masks, zero, trt.ElementWiseOperation.GREATER).get_output(
        0
    )
    sentinel = g.add_constant(network, (1, 1, 1, 1), [np.finfo(np.float32).max])
    coordinates = []
    for axis, extent in ((3, width), (2, height)):
        shape = [1, 1, 1, 1]
        shape[axis] = extent
        grid = g.add_constant(network, tuple(shape), np.arange(extent, dtype=np.float32))
        minimum = network.add_select(foreground, grid, sentinel).get_output(0)
        minimum = network.add_reduce(
            minimum, trt.ReduceOperation.MIN, (1 << 2) | (1 << 3), True
        ).get_output(0)
        present = network.add_elementwise(
            minimum, sentinel, trt.ElementWiseOperation.LESS
        ).get_output(0)
        minimum = network.add_select(present, minimum, zero).get_output(0)
        maximum = network.add_select(foreground, grid, zero).get_output(0)
        maximum = network.add_reduce(
            maximum, trt.ReduceOperation.MAX, (1 << 2) | (1 << 3), True
        ).get_output(0)
        maximum = g.add_sum(network, maximum, g.add_constant(network, (1, 1, 1, 1), [1.0]))
        maximum = network.add_select(present, maximum, zero).get_output(0)
        centre = g.add_sum(network, minimum, maximum)
        centre = network.add_elementwise(
            centre,
            g.add_constant(network, (1, 1, 1, 1), [0.5 / extent]),
            trt.ElementWiseOperation.PROD,
        ).get_output(0)
        size = network.add_elementwise(maximum, minimum, trt.ElementWiseOperation.SUB).get_output(0)
        size = network.add_elementwise(
            size,
            g.add_constant(network, (1, 1, 1, 1), [1.0 / extent]),
            trt.ElementWiseOperation.PROD,
        ).get_output(0)
        coordinates.append((centre, size))
    references = g.concat(
        network, [coordinates[0][0], coordinates[1][0], coordinates[0][1], coordinates[1][1]], 3
    )
    reshape = network.add_shuffle(references)
    reshape.reshape_dims = (1, queries, 4)
    # Match the reference's inverse_sigmoid -> sigmoid round trip.
    return network.add_activation(
        _inverse_sigmoid(network, reshape.get_output(0), np.float32), trt.ActivationType.SIGMOID
    ).get_output(0)


def build_decoder(network, memory, shapes, mask_features, weights, cfg):
    queries = cfg["num_queries"]
    total = sum(h * w for h, w in shapes)
    valid = np.isfinite(build_anchors(shapes)).all(-1, keepdims=True).astype(np.float32)
    masked = network.add_elementwise(
        memory, g.add_constant(network, (1, total, 1), valid), trt.ElementWiseOperation.PROD
    ).get_output(0)
    projected = g.add_linear(
        network, masked, weights["model.enc_output.0.weight"], weights["model.enc_output.0.bias"]
    )
    output_memory = g.add_layer_norm(
        network,
        projected,
        weights["model.enc_output.1.weight"],
        weights["model.enc_output.1.bias"],
        1e-5,
    )
    scores = g.add_linear(
        network,
        output_memory,
        weights["model.enc_score_head.weight"],
        weights["model.enc_score_head.bias"],
    )
    best = network.add_reduce(scores, trt.ReduceOperation.MAX, 1 << 2, False).get_output(0)
    indices = network.add_topk(best, trt.TopKOperation.MAX, queries, 1 << 1).get_output(1)
    gather = network.add_gather_v2(output_memory, indices, trt.GatherMode.DEFAULT)
    gather.axis = 1
    gather.num_elementwise_dims = 1
    hidden = gather.get_output(0)
    references = _mask_boxes(network, _masks(network, hidden, mask_features, weights))
    for index in range(cfg["decoder_layers"]):
        position = _mlp(network, references, weights, "model.decoder.query_pos_head", 2, np.float32)
        hidden = _decoder_layer(
            network,
            hidden,
            position,
            memory,
            shapes,
            references,
            weights,
            f"model.decoder.layers.{index}",
            cfg,
            np.float32,
        )
        # These heads are shared with the encoder heads in the checkpoint.
        delta = _mlp(network, hidden, weights, "model.enc_bbox_head", 3, np.float32)
        references = network.add_activation(
            g.add_sum(network, delta, _inverse_sigmoid(network, references, np.float32)),
            trt.ActivationType.SIGMOID,
        ).get_output(0)
    out_query = _norm(network, hidden, weights)
    logits = g.add_linear(
        network,
        out_query,
        weights["model.enc_score_head.weight"],
        weights["model.enc_score_head.bias"],
    )
    masks = _masks(network, hidden, mask_features, weights)
    prefix = f"model.decoder_order_head.{cfg['decoder_layers'] - 1}"
    order = g.add_linear(network, out_query, weights[f"{prefix}.weight"], weights[f"{prefix}.bias"])
    order = g.add_linear(
        network,
        order,
        weights["model.decoder_global_pointer.dense.weight"],
        weights["model.decoder_global_pointer.dense.bias"],
    )
    head = cfg["global_pointer_head_size"]
    query = network.add_slice(order, (0, 0, 0), (1, queries, head), (1, 1, 1)).get_output(0)
    key = network.add_slice(order, (0, 0, head), (1, queries, head), (1, 1, 1)).get_output(0)
    order = network.add_matrix_multiply(
        query, trt.MatrixOperation.NONE, key, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    order = network.add_elementwise(
        order,
        g.add_constant(network, (1, 1, 1), [1.0 / math.sqrt(head)]),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    keep = np.triu(np.ones((1, queries, queries), dtype=np.bool_), k=1)
    condition = network.add_constant(keep.shape, keep).get_output(0)
    order = network.add_select(
        condition, order, g.add_constant(network, (1, 1, 1), [-1e4])
    ).get_output(0)
    return logits, references, order, masks
