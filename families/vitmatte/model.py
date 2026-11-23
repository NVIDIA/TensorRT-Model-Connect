# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ViTMatte window/global relative attention and detail-capture alpha decoder."""

import json
from pathlib import Path

import numpy as np
import tensorrt as trt

from . import graph as g


def build(request, writer):
    if request.task != "image_trimap_to_alpha" or request.backend != "trt":
        raise ValueError("ViTMatte requires image_trimap_to_alpha with backend=trt")
    if request.dynamic_kv_cache:
        raise NotImplementedError("vitmatte does not support dynamic_kv_cache")
    if (
        request.precision != "fp32"
        or request.tensor_parallel_size != 1
        or request.context_parallel_size != 1
        or request.max_batch_size != 1
        or request.quantization not in (None, "none")
        or request.fp32_layers
        or request.dynamic_kv_cache
        or request.max_sequence_length is not None
        or request.video_num_frames is not None
        or request.mtp_seq_len is not None
    ):
        raise NotImplementedError("ViTMatte supports one fp32 image and categorical trimap")
    directory = Path(request.model_dir)
    from safetensors import safe_open
    import torch
    import torch.nn.functional as functional
    from transformers import VitMatteConfig

    preprocessing = json.loads((directory / "preprocessor_config.json").read_text())
    if (
        preprocessing["image_mean"] != [0.5, 0.5, 0.5]
        or preprocessing["image_std"] != [0.5, 0.5, 0.5]
        or not preprocessing["do_rescale"]
        or not preprocessing["do_normalize"]
        or preprocessing["rescale_factor"] != 1 / 255
    ):
        raise ValueError("Unsupported ViTMatte image normalization")
    cfg = VitMatteConfig.from_pretrained(directory)
    backbone = cfg.backbone_config
    height, width = (
        request.image_height or backbone.image_size,
        request.image_width or backbone.image_size,
    )
    if (
        backbone.model_type != "vitdet"
        or backbone.num_channels != 4
        or backbone.hidden_act != "gelu"
        or backbone.patch_size != 16
        or cfg.hidden_size != backbone.hidden_size
        or not backbone.use_absolute_position_embeddings
        or not backbone.use_relative_position_embeddings
        or backbone.out_indices != [backbone.num_hidden_layers]
        or height % 32
        or width % 32
        or min(height, width) < 32
        or max(height, width) > 1024
    ):
        raise ValueError("Unsupported ViTMatte backbone or image profile")
    with safe_open(str(directory / "model.safetensors"), framework="np") as handle:
        weights = {name: handle.get_tensor(name).astype(np.float32) for name in handle.keys()}
    channels, heads = backbone.hidden_size, backbone.num_attention_heads
    dim = channels // heads
    gh, gw = height // 16, width // 16
    logger = trt.Logger(trt.Logger.VERBOSE if request.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    options = builder.create_builder_config()
    options.builder_optimization_level = 3
    options.clear_flag(trt.BuilderFlag.TF32)

    def const(value):
        value = np.ascontiguousarray(value, dtype=np.float32)
        return net.add_constant(value.shape, value).get_output(0)

    def shuf(x, shape=None, perm=None):
        layer = net.add_shuffle(x)
        if perm is not None:
            layer.first_transpose = perm
        if shape is not None:
            layer.reshape_dims = shape
        return layer.get_output(0)

    def add(a, b):
        return net.add_elementwise(a, b, trt.ElementWiseOperation.SUM).get_output(0)

    def mul(a, b):
        return net.add_elementwise(a, b, trt.ElementWiseOperation.PROD).get_output(0)

    def linear(x, name):
        if name.endswith(".mlp.fc2"):
            from .kernels import mlp_output

            return mlp_output(net, x, weights[name + ".weight"], weights[name + ".bias"])
        return g.add_linear(net, x, weights[name + ".weight"], weights.get(name + ".bias"))

    def norm(x, name, *, spatial=False):
        if spatial:
            x = shuf(x, perm=(0, 2, 3, 1))
        result = g.add_layer_norm(
            net, x, weights[name + ".weight"], weights[name + ".bias"], backbone.layer_norm_eps
        )
        return shuf(result, perm=(0, 3, 1, 2)) if spatial else result

    def conv(x, name, *, stride=1, padding=0):
        weight = weights[name + ".weight"]
        layer = net.add_convolution_nd(
            x,
            weight.shape[0],
            weight.shape[2:],
            weight,
            weights.get(name + ".bias", trt.Weights(trt.float32)),
        )
        layer.stride_nd = (stride, stride)
        layer.padding_nd = (padding, padding)
        return layer.get_output(0)

    def gelu(x):
        from .kernels import gelu as fp32_gelu

        return fp32_gelu(net, x)

    def batch_norm(x, name, epsilon):
        shape = (1, -1, 1, 1)
        centered = net.add_elementwise(
            x, const(weights[name + ".running_mean"].reshape(shape)), trt.ElementWiseOperation.SUB
        ).get_output(0)
        variance = add(
            const(weights[name + ".running_var"].reshape(shape)),
            const(np.full((1, 1, 1, 1), epsilon)),
        )
        deviation = net.add_unary(variance, trt.UnaryOperation.SQRT).get_output(0)
        normalized = net.add_elementwise(
            centered, deviation, trt.ElementWiseOperation.DIV
        ).get_output(0)
        return add(
            mul(normalized, const(weights[name + ".weight"].reshape(shape))),
            const(weights[name + ".bias"].reshape(shape)),
        )

    def basic_conv(x, name, stride):
        x = conv(x, name + ".conv", stride=stride, padding=1)
        x = batch_norm(x, name + ".batch_norm", cfg.batch_norm_eps)
        return net.add_activation(x, trt.ActivationType.RELU).get_output(0)

    def relative_table(name, length):
        table = torch.from_numpy(weights[name])
        if table.shape[0] != 2 * length - 1:
            table = functional.interpolate(table.T[None], size=2 * length - 1, mode="linear")[0].T
        positions = np.arange(length)
        return table.numpy()[positions[:, None] - positions[None, :] + length - 1]

    def attention(x, prefix):
        batch, h, w, _ = tuple(x.shape)
        length, bh = h * w, batch * heads
        qkv = linear(x, prefix + ".qkv")
        tensors = []
        for index in range(3):
            piece = net.add_slice(
                qkv, (0, 0, 0, index * channels), (batch, h, w, channels), (1, 1, 1, 1)
            ).get_output(0)
            piece = shuf(piece, (batch, length, heads, dim))
            tensors.append(shuf(piece, (bh, length, dim), (0, 2, 1, 3)))
        query, key, value = tensors
        scores = net.add_matrix_multiply(
            mul(query, const(np.full((1, 1, 1), dim**-0.5))),
            trt.MatrixOperation.NONE,
            key,
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)
        query_grid = shuf(query, (bh, h, w, dim))
        by_h = shuf(query_grid, (h, bh * w, dim), (1, 0, 2, 3))
        rel_h = net.add_matrix_multiply(
            by_h,
            trt.MatrixOperation.NONE,
            const(relative_table(prefix + ".rel_pos_h", h)),
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)
        rel_h = shuf(shuf(rel_h, (h, bh, w, h)), (bh, h, w, h, 1), (1, 0, 2, 3))
        by_w = shuf(query_grid, (w, bh * h, dim), (2, 0, 1, 3))
        rel_w = net.add_matrix_multiply(
            by_w,
            trt.MatrixOperation.NONE,
            const(relative_table(prefix + ".rel_pos_w", w)),
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)
        rel_w = shuf(shuf(rel_w, (w, bh, h, w)), (bh, h, w, 1, w), (1, 2, 0, 3))
        scores = shuf(add(add(shuf(scores, (bh, h, w, h, w)), rel_h), rel_w), (bh, length, length))
        softmax = net.add_softmax(scores)
        softmax.axes = 4
        context = net.add_matrix_multiply(
            softmax.get_output(0), trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE
        ).get_output(0)
        context = shuf(context, (batch, heads, h, w, dim))
        context = shuf(context, (batch, h, w, channels), (0, 2, 3, 1, 4))
        return linear(context, prefix + ".proj")

    pixels = net.add_input("pixel_values", trt.float32, (1, 4, height, width))
    hidden = shuf(conv(pixels, "backbone.embeddings.projection", stride=16), perm=(0, 2, 3, 1))
    absolute = torch.from_numpy(weights["backbone.embeddings.position_embeddings"][:, 1:])
    side = int(absolute.shape[1] ** 0.5)
    absolute = functional.interpolate(
        absolute.reshape(1, side, side, channels).permute(0, 3, 1, 2),
        size=(gh, gw),
        mode="bicubic",
        align_corners=False,
    )
    hidden = add(hidden, const(absolute.permute(0, 2, 3, 1).numpy()))
    for index in range(backbone.num_hidden_layers):
        prefix = f"backbone.encoder.layer.{index}"
        normalized = norm(hidden, prefix + ".norm1")
        window = backbone.window_size if index in backbone.window_block_indices else 0
        if window:
            ph, pw = ((gh + window - 1) // window) * window, ((gw + window - 1) // window) * window
            pad = net.add_slice(normalized, (0, 0, 0, 0), (1, ph, pw, channels), (1, 1, 1, 1))
            pad.mode = trt.SampleMode.FILL
            normalized = shuf(
                pad.get_output(0), (1, ph // window, window, pw // window, window, channels)
            )
            normalized = shuf(
                normalized,
                ((ph // window) * (pw // window), window, window, channels),
                (0, 1, 3, 2, 4, 5),
            )
        inner = attention(normalized, prefix + ".attention")
        if window:
            inner = shuf(inner, (1, ph // window, pw // window, window, window, channels))
            inner = shuf(inner, (1, ph, pw, channels), (0, 1, 3, 2, 4, 5))
            inner = net.add_slice(
                inner, (0, 0, 0, 0), (1, gh, gw, channels), (1, 1, 1, 1)
            ).get_output(0)
        hidden = add(hidden, inner)
        inner = linear(
            gelu(linear(norm(hidden, prefix + ".norm2"), prefix + ".mlp.fc1")), prefix + ".mlp.fc2"
        )
        hidden = add(hidden, inner)
        if index in backbone.residual_block_indices:
            spatial = shuf(hidden, perm=(0, 3, 1, 2))
            inner = gelu(
                norm(
                    conv(spatial, prefix + ".residual.conv1"),
                    prefix + ".residual.norm1",
                    spatial=True,
                )
            )
            inner = gelu(
                norm(
                    conv(inner, prefix + ".residual.conv2", padding=1),
                    prefix + ".residual.norm2",
                    spatial=True,
                )
            )
            inner = norm(
                conv(inner, prefix + ".residual.conv3"), prefix + ".residual.norm3", spatial=True
            )
            hidden = shuf(add(spatial, inner), perm=(0, 2, 3, 1))
    hidden = shuf(hidden, perm=(0, 3, 1, 2))
    details = [pixels]
    for index in range(len(cfg.convstream_hidden_sizes)):
        details.append(basic_conv(details[-1], f"decoder.convstream.convs.{index}", 2))
    for index in range(len(cfg.fusion_hidden_sizes)):
        resize = net.add_resize(hidden)
        resize.resize_mode = trt.InterpolationMode.LINEAR
        resize.coordinate_transformation = trt.ResizeCoordinateTransformation.HALF_PIXEL
        resize.scales = (1.0, 1.0, 2.0, 2.0)
        joined = net.add_concatenation([details[-index - 1], resize.get_output(0)])
        joined.axis = 1
        hidden = basic_conv(joined.get_output(0), f"decoder.fusion_blocks.{index}.conv", 1)
    hidden = conv(hidden, "decoder.matting_head.matting_convs.0", padding=1)
    hidden = batch_norm(hidden, "decoder.matting_head.matting_convs.1", 1e-5)
    hidden = net.add_activation(hidden, trt.ActivationType.RELU).get_output(0)
    hidden = conv(hidden, "decoder.matting_head.matting_convs.3")
    alpha = shuf(
        net.add_activation(hidden, trt.ActivationType.SIGMOID).get_output(0), (height, width)
    )
    alpha.name = "alpha"
    net.mark_output(alpha)
    plan = builder.build_serialized_network(net, options)
    if plan is None:
        raise RuntimeError("ViTMatte engine build failed")
    writer.set_header(family="vitmatte", task=request.task, backend=request.backend)
    writer.add_bytes("model.plan", bytes(plan))
    writer.add_json("runtime.json", dict(height=height, width=width))
