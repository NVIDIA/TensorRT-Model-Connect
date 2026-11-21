# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SigLIP vision tower and multimodal projector for the Pi 0.5 prefix graph.

The projections, attention and MLP run in the working dtype; the residual
stream, the pixel normalisation and the patch embedding stay float32, and the
norm statistics, softmax and GELU are taken in float32 by the graph helpers.
In an fp32 build the whole tower is float32, as in LeRobot, which keeps its
vision path in float32 even for its bfloat16 policy. In a bf16 build each
sublayer reads the float32 stream rounded to bfloat16 and its result is added
back in float32. Measured against LeRobot float32 (see the README), this is as
fast as an all-bfloat16 tower with about the mean action error of a float32
tower; a bfloat16 residual stream doubled the worst-case action error.
"""

from __future__ import annotations

import tensorrt as trt

from . import graph
from .checkpoint import require

WIDTH = 1152
DEPTH = 27
HEADS = 16
HEAD_DIM = 72
MLP = 4304
PATCH = 14
IMAGE_SIZE = 224
TOKENS = (IMAGE_SIZE // PATCH) ** 2
LN_EPS = 1e-6
PROJECTION = 2048

_ROOT = "paligemma.model.vision_tower.vision_model."


def _normalise_pixels(network, pixels, height, width):
    """``[N,3,H,W]`` in [0,1] -> ``[N,3,224,224]`` in [-1,1], as ``resize_with_pad`` then ``2x-1``.

    Square inputs only (checked by the config): bilinear with half-pixel
    centres is torch's ``align_corners=False``, and there is nothing to pad.
    """
    batch = int(pixels.shape[0])
    x = pixels
    if (height, width) != (IMAGE_SIZE, IMAGE_SIZE):
        layer = network.add_resize(x)
        layer.shape = (batch, 3, IMAGE_SIZE, IMAGE_SIZE)
        layer.resize_mode = trt.InterpolationMode.LINEAR
        layer.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
        x = layer.get_output(0)
        # The reference clamps the interpolated image back into [0, 1].
        x = network.add_elementwise(x, graph.scalar(network, 0.0, 4), trt.ElementWiseOperation.MAX).get_output(0)
        x = network.add_elementwise(x, graph.scalar(network, 1.0, 4), trt.ElementWiseOperation.MIN).get_output(0)
    return graph.sub(network, graph.mul(network, x, graph.scalar(network, 2.0, 4)), graph.scalar(network, 1.0, 4))


def build_vision(network, pixels, weights, *, height, width, dtype=trt.float32, image_feature_scale=1.0):
    """Return image tokens ``[N, 256, 2048]`` in ``dtype`` for float32 ``pixels[N, 3, H, W]``."""
    batch = int(pixels.shape[0])
    x = _normalise_pixels(network, pixels, height, width)

    kernel = require(weights, _ROOT + "embeddings.patch_embedding.weight", (WIDTH, 3, PATCH, PATCH))
    bias = require(weights, _ROOT + "embeddings.patch_embedding.bias", (WIDTH,))
    conv = network.add_convolution_nd(x, WIDTH, (PATCH, PATCH), trt.Weights(kernel), trt.Weights(bias))
    conv.stride_nd = (PATCH, PATCH)
    patches = graph.shuffle(network, conv.get_output(0), reshape=(batch, WIDTH, TOKENS), second=(0, 2, 1))
    position = require(weights, _ROOT + "embeddings.position_embedding.weight", (TOKENS, WIDTH))
    h = graph.add(network, patches, graph.const(network, position.reshape(1, TOKENS, WIDTH)))

    def normed(t, name):
        """LayerNorm of the float32 stream, rounded to ``dtype`` for the sublayer."""
        return graph.cast(network, graph.layer_norm(
            network, t, weights[name + ".weight"], weights[name + ".bias"], LN_EPS), dtype)

    def residual(t, update):
        return graph.add(network, t, graph.cast(network, update, trt.float32))

    for index in range(DEPTH):
        p = f"{_ROOT}encoder.layers.{index}."
        norm = normed(h, p + "layer_norm1")
        q = graph.linear(network, norm, weights[p + "self_attn.q_proj.weight"], weights[p + "self_attn.q_proj.bias"])
        k = graph.linear(network, norm, weights[p + "self_attn.k_proj.weight"], weights[p + "self_attn.k_proj.bias"])
        v = graph.linear(network, norm, weights[p + "self_attn.v_proj.weight"], weights[p + "self_attn.v_proj.bias"])
        context = graph.attention_mha(network, q, k, v, heads=HEADS, head_dim=HEAD_DIM)
        h = residual(h, graph.linear(
            network, context, weights[p + "self_attn.out_proj.weight"], weights[p + "self_attn.out_proj.bias"]))
        norm = normed(h, p + "layer_norm2")
        hidden = graph.gelu_tanh(network, graph.linear(
            network, norm, weights[p + "mlp.fc1.weight"], weights[p + "mlp.fc1.bias"]))
        h = residual(h, graph.linear(network, hidden, weights[p + "mlp.fc2.weight"], weights[p + "mlp.fc2.bias"]))

    h = normed(h, _ROOT + "post_layernorm")
    projector = "paligemma.model.multi_modal_projector.linear."
    out = graph.linear(network, h, weights[projector + "weight"], weights[projector + "bias"])
    if image_feature_scale != 1.0:
        out = graph.mul(network, out, graph.scalar(network, image_feature_scale, 3, out.dtype))
    return out
