# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned PP-DocLayoutV3 hybrid encoder."""

from __future__ import annotations

import math

import numpy as np

from . import graph as g

_LAYER_NORM_EPS = 1e-5
_BN_EPS = 1e-5


def sine_position_embedding(height, width, channels, temperature):
    """The embedding AIFI adds to its input, as a constant."""
    pos_dim = channels // 4
    omega = 1.0 / (temperature ** (np.arange(pos_dim, dtype=np.float32) / pos_dim))
    grid_w, grid_h = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32), indexing="xy"
    )
    out_w = grid_w.reshape(-1, 1) * omega.reshape(1, -1)
    out_h = grid_h.reshape(-1, 1) * omega.reshape(1, -1)
    embedding = np.concatenate([np.sin(out_h), np.cos(out_h), np.sin(out_w), np.cos(out_w)], axis=1)
    return embedding.reshape(1, height * width, channels).astype(np.float32)


def _conv_norm(network, x, weights, prefix, stride, padding, dtype, activate=True):
    """Convolution, folded BatchNorm, then SiLU unless told otherwise."""
    folded, bias = g.fold_batch_norm(
        weights[f"{prefix}.conv.weight"],
        weights[f"{prefix}.norm.weight"],
        weights[f"{prefix}.norm.bias"],
        weights[f"{prefix}.norm.running_mean"],
        weights[f"{prefix}.norm.running_var"],
        _BN_EPS,
    )
    out = g.add_conv2d(network, x, folded, bias, stride=stride, padding=padding, dtype=dtype)
    return g.add_silu(network, out) if activate else out


def _rep_vgg(network, x, weights, prefix, dtype):
    """A 3x3 and a 1x1 branch summed, then SiLU. Not fused in the checkpoint."""
    left = _conv_norm(network, x, weights, f"{prefix}.conv1", (1, 1), (1, 1), dtype, activate=False)
    right = _conv_norm(
        network, x, weights, f"{prefix}.conv2", (1, 1), (0, 0), dtype, activate=False
    )
    return g.add_silu(network, g.add_sum(network, left, right))


def _csp_rep(network, x, weights, prefix, bottlenecks, dtype):
    """CSP split, a stack of RepVGG blocks on one side, then rejoin."""
    left = _conv_norm(network, x, weights, f"{prefix}.conv1", (1, 1), (0, 0), dtype)
    for index in range(bottlenecks):
        left = _rep_vgg(network, left, weights, f"{prefix}.bottlenecks.{index}", dtype)
    right = _conv_norm(network, x, weights, f"{prefix}.conv2", (1, 1), (0, 0), dtype)
    return g.add_sum(network, left, right)


def _aifi(network, x, weights, prefix, cfg, dtype):
    """One post-norm transformer layer over the flattened smallest level."""
    shape = tuple(int(v) for v in x.shape)
    height, width = shape[2], shape[3]
    hidden = cfg["encoder_hidden_dim"]
    heads = cfg["encoder_attention_heads"]
    head_dim = hidden // heads
    tokens = height * width

    source = g.spatial_to_tokens(network, x)
    position = g.add_constant(
        network,
        (1, tokens, hidden),
        sine_position_embedding(height, width, hidden, cfg["positional_encoding_temperature"]),
        dtype=dtype,
    )
    # Position is added to the query and key only; the value stays unposed.
    posed = g.add_sum(network, source, position)

    layer = f"{prefix}.layers.0"
    query = g.add_linear(
        network,
        posed,
        weights[f"{layer}.self_attn.q_proj.weight"],
        weights[f"{layer}.self_attn.q_proj.bias"],
        dtype=dtype,
    )
    key = g.add_linear(
        network,
        posed,
        weights[f"{layer}.self_attn.k_proj.weight"],
        weights[f"{layer}.self_attn.k_proj.bias"],
        dtype=dtype,
    )
    value = g.add_linear(
        network,
        source,
        weights[f"{layer}.self_attn.v_proj.weight"],
        weights[f"{layer}.self_attn.v_proj.bias"],
        dtype=dtype,
    )
    context = _attention(
        network, query, key, value, tokens, heads, head_dim, 1.0 / math.sqrt(head_dim), dtype=dtype
    )
    attended = g.add_linear(
        network,
        context,
        weights[f"{layer}.self_attn.out_proj.weight"],
        weights[f"{layer}.self_attn.out_proj.bias"],
        dtype=dtype,
    )
    h = g.add_layer_norm(
        network,
        g.add_sum(network, source, attended),
        weights[f"{layer}.self_attn_layer_norm.weight"],
        weights[f"{layer}.self_attn_layer_norm.bias"],
        _LAYER_NORM_EPS,
        dtype=dtype,
    )

    inner = g.add_gelu(
        network,
        g.add_linear(
            network, h, weights[f"{layer}.fc1.weight"], weights[f"{layer}.fc1.bias"], dtype=dtype
        ),
    )
    out = g.add_linear(
        network, inner, weights[f"{layer}.fc2.weight"], weights[f"{layer}.fc2.bias"], dtype=dtype
    )
    h = g.add_layer_norm(
        network,
        g.add_sum(network, h, out),
        weights[f"{layer}.final_layer_norm.weight"],
        weights[f"{layer}.final_layer_norm.bias"],
        _LAYER_NORM_EPS,
        dtype=dtype,
    )
    return g.tokens_to_spatial(network, h, height, width)


def _attention(network, query, key, value, tokens, heads, head_dim, scale, dtype=np.float32):
    import tensorrt as trt

    def split(tensor):
        layer = network.add_shuffle(tensor)
        layer.reshape_dims = (1, tokens, heads, head_dim)
        layer.second_transpose = (0, 2, 1, 3)
        return layer.get_output(0)

    q, k, v = split(query), split(key), split(value)
    scores = network.add_matrix_multiply(
        q, trt.MatrixOperation.NONE, k, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    scaled = network.add_elementwise(
        scores,
        g.add_constant(network, (1, 1, 1, 1), np.full((1,), scale), dtype=dtype),
        trt.ElementWiseOperation.PROD,
    ).get_output(0)
    weights = network.add_softmax(scaled)
    weights.axes = 1 << 3
    context = network.add_matrix_multiply(
        weights.get_output(0), trt.MatrixOperation.NONE, v, trt.MatrixOperation.NONE
    ).get_output(0)
    merge = network.add_shuffle(context)
    merge.first_transpose = (0, 2, 1, 3)
    merge.reshape_dims = (1, tokens, heads * head_dim)
    return merge.get_output(0)


def build_encoder(network, levels, weights, cfg, dtype=np.float32):
    """Project, run AIFI on one level, fuse top-down then bottom-up."""
    root = "model"
    count = len(levels)
    projected = []
    for index, level in enumerate(levels):
        folded, bias = g.fold_batch_norm(
            weights[f"{root}.encoder_input_proj.{index}.0.weight"],
            weights[f"{root}.encoder_input_proj.{index}.1.weight"],
            weights[f"{root}.encoder_input_proj.{index}.1.bias"],
            weights[f"{root}.encoder_input_proj.{index}.1.running_mean"],
            weights[f"{root}.encoder_input_proj.{index}.1.running_var"],
            _BN_EPS,
        )
        # The projection has no activation of its own.
        projected.append(g.add_conv2d(network, level, folded, bias, dtype=dtype))

    for slot, level_index in enumerate(cfg["encode_proj_layers"]):
        projected[level_index] = _aifi(
            network, projected[level_index], weights, f"{root}.encoder.encoder.{slot}", cfg, dtype
        )

    # FPN, high resolution gained by walking down the list.
    inner = [projected[-1]]
    for index in range(count - 1, 0, -1):
        slot = count - 1 - index
        high = _conv_norm(
            network,
            inner[0],
            weights,
            f"{root}.encoder.lateral_convs.{slot}",
            (1, 1),
            (0, 0),
            dtype,
        )
        inner[0] = high
        shape = tuple(int(v) for v in high.shape)
        upsampled = g.add_resize_nearest(network, high, (shape[2] * 2, shape[3] * 2))
        merged = g.concat(network, [upsampled, projected[index - 1]], axis=1)
        inner.insert(
            0,
            _csp_rep(
                network,
                merged,
                weights,
                f"{root}.encoder.fpn_blocks.{slot}",
                cfg["bottlenecks"],
                dtype,
            ),
        )

    # PAN, resolution given back up the list.
    outputs = [inner[0]]
    for index in range(count - 1):
        down = _conv_norm(
            network,
            outputs[-1],
            weights,
            f"{root}.encoder.downsample_convs.{index}",
            (2, 2),
            (1, 1),
            dtype,
        )
        merged = g.concat(network, [down, inner[index + 1]], axis=1)
        outputs.append(
            _csp_rep(
                network,
                merged,
                weights,
                f"{root}.encoder.pan_blocks.{index}",
                cfg["bottlenecks"],
                dtype,
            )
        )
    return outputs


def project_for_decoder(network, levels, weights, dtype=np.float32):
    """Map the encoder's outputs into the decoder's memory.

    This is a separate 1x1 convolution and BatchNorm per level, distinct from
    ``encoder_input_proj``, and it is easy to miss because the width does not
    change: 256 in, 256 out. Skipping it leaves the encoder outputs looking
    correct while the decoder reads something nearly orthogonal.
    """
    projected = []
    for index, level in enumerate(levels):
        folded, bias = g.fold_batch_norm(
            weights[f"model.decoder_input_proj.{index}.0.weight"],
            weights[f"model.decoder_input_proj.{index}.1.weight"],
            weights[f"model.decoder_input_proj.{index}.1.bias"],
            weights[f"model.decoder_input_proj.{index}.1.running_mean"],
            weights[f"model.decoder_input_proj.{index}.1.running_var"],
            _BN_EPS,
        )
        projected.append(g.add_conv2d(network, level, folded, bias, dtype=dtype))
    return projected


def build_mask_features(network, levels, x4, weights):
    from .backbone_builder import conv

    root = "model.encoder"
    features = []
    for level, x in enumerate(levels):
        prefix = f"{root}.mask_feature_head.scale_heads.{level}.layers"
        for index in range(max(1, level)):
            x = conv(network, x, weights, f"{prefix}.{index * 2}", activation="silu")
            if level:
                x = resize_linear(network, x, int(x.shape[2]) * 2, int(x.shape[3]) * 2)
        features.append(resize_linear(network, x, int(levels[0].shape[2]), int(levels[0].shape[3])))
    x = g.add_sum(network, g.add_sum(network, features[0], features[1]), features[2])
    x = conv(network, x, weights, f"{root}.mask_feature_head.output_conv", activation="silu")
    x = resize_linear(network, x, int(x.shape[2]) * 2, int(x.shape[3]) * 2)
    x = g.add_sum(
        network, x, conv(network, x4, weights, f"{root}.encoder_mask_lateral", activation="silu")
    )
    x = conv(network, x, weights, f"{root}.encoder_mask_output.base_conv", activation="silu")
    prefix = f"{root}.encoder_mask_output.conv"
    return g.add_conv2d(network, x, weights[f"{prefix}.weight"], weights[f"{prefix}.bias"])


def resize_linear(network, x, height, width):
    import tensorrt as trt

    resize = network.add_resize(x)
    resize.shape = (int(x.shape[0]), int(x.shape[1]), height, width)
    resize.resize_mode = trt.InterpolationMode.LINEAR
    resize.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
    return resize.get_output(0)
