# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Latent tokens → image for flux3 (``AutoencoderKLFlux2`` decoder) as one engine.

Follows the reference decode step: unpack the ``[N, 128]`` tokens onto the
latent grid, undo the batch-norm latent normalisation in the latents' dtype
(BF16), unpatchify ``2 x 2`` to 32 channels, then ``post_quant_conv`` and the
KL decoder in float32.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .trt_ops import Graph, new_builder_config, new_network, serialize
from .weights import ComponentWeights


class _Decoder:
    def __init__(self, g: Graph, weights: ComponentWeights, config: dict) -> None:
        self.g = g
        self.w = weights
        self.groups = int(config["norm_num_groups"])
        self.eps = 1e-6

    def conv(self, x: trt.ITensor, name: str, kernel: int) -> trt.ITensor:
        weight = np.ascontiguousarray(self.w.get(name + ".weight").astype(np.float32))
        bias = np.ascontiguousarray(self.w.get(name + ".bias").astype(np.float32))
        self.g.keep(weight, bias)
        layer = self.g.net.add_convolution_nd(
            x, weight.shape[0], (kernel, kernel), trt.Weights(weight), trt.Weights(bias)
        )
        layer.padding_nd = (kernel // 2, kernel // 2)
        return layer.get_output(0)

    def group_norm(self, x: trt.ITensor, name: str) -> trt.ITensor:
        g = self.g
        _, channels, height, width = x.shape
        grouped = g.reshape(x, (1, self.groups, channels // self.groups, height, width))
        axes = (1 << 2) | (1 << 3) | (1 << 4)
        mean = g.net.add_reduce(grouped, trt.ReduceOperation.AVG, axes, keep_dims=True).get_output(
            0
        )
        centered = g.sub(grouped, mean)
        var = g.net.add_reduce(
            g.mul(centered, centered), trt.ReduceOperation.AVG, axes, keep_dims=True
        ).get_output(0)
        std = g.net.add_unary(
            g.add(var, g.scalar(self.eps, trt.float32, 5)), trt.UnaryOperation.SQRT
        ).get_output(0)
        normed = g.reshape(
            g.net.add_elementwise(centered, std, trt.ElementWiseOperation.DIV).get_output(0),
            (1, channels, height, width),
        )
        gamma = self.w.get(name + ".weight", (channels,)).astype(np.float32).reshape(1, -1, 1, 1)
        beta = self.w.get(name + ".bias", (channels,)).astype(np.float32).reshape(1, -1, 1, 1)
        return g.add(g.mul(normed, g.constant(gamma)), g.constant(beta))

    def resnet(self, x: trt.ITensor, name: str) -> trt.ITensor:
        g = self.g
        h = self.conv(g.silu(self.group_norm(x, name + ".norm1")), name + ".conv1", 3)
        h = self.conv(g.silu(self.group_norm(h, name + ".norm2")), name + ".conv2", 3)
        shortcut = (
            self.conv(x, name + ".conv_shortcut", 1)
            if self.w.has(name + ".conv_shortcut.weight")
            else x
        )
        return g.add(shortcut, h)

    def project(self, rows: trt.ITensor, name: str, channels: int) -> trt.ITensor:
        g = self.g
        weight = self.w.get(name + ".weight", (channels, channels)).astype(np.float32)
        bias = self.w.get(name + ".bias", (channels,)).astype(np.float32).reshape(1, -1)
        return g.add(g.linear(rows, weight, trt.float32), g.constant(bias))

    def attention(self, x: trt.ITensor, name: str) -> trt.ITensor:
        """Single-head mid-block self-attention with a residual connection."""

        g = self.g
        _, channels, height, width = x.shape
        normed = self.group_norm(x, name + ".group_norm")
        rows = g.reshape(g.permute(normed, (0, 2, 3, 1)), (height * width, channels))
        q = g.scale_query(self.project(rows, name + ".to_q", channels), channels, trt.float32)
        k = self.project(rows, name + ".to_k", channels)
        v = self.project(rows, name + ".to_v", channels)
        attn = g.attention(*(g.reshape(t, (1, 1, height * width, channels)) for t in (q, k, v)))
        context = g.reshape(attn, (height * width, channels))
        out = self.project(context, name + ".to_out.0", channels)
        return g.add(x, g.permute(g.reshape(out, (1, height, width, channels)), (0, 3, 1, 2)))

    def upsample(self, x: trt.ITensor, name: str) -> trt.ITensor:
        layer = self.g.net.add_resize(x)
        layer.resize_mode = trt.InterpolationMode.NEAREST
        layer.scales = [1.0, 1.0, 2.0, 2.0]
        return self.conv(layer.get_output(0), name + ".conv", 3)

    def decode(self, z: trt.ITensor, config: dict) -> trt.ITensor:
        g = self.g
        if config.get("use_post_quant_conv", True):
            z = self.conv(z, "post_quant_conv", 1)
        h = self.conv(z, "decoder.conv_in", 3)
        h = self.resnet(h, "decoder.mid_block.resnets.0")
        h = self.attention(h, "decoder.mid_block.attentions.0")
        h = self.resnet(h, "decoder.mid_block.resnets.1")
        blocks = len(config["up_block_types"])
        for i in range(blocks):
            for j in range(int(config["layers_per_block"]) + 1):
                h = self.resnet(h, f"decoder.up_blocks.{i}.resnets.{j}")
            if i < blocks - 1:
                h = self.upsample(h, f"decoder.up_blocks.{i}.upsamplers.0")
        h = g.silu(self.group_norm(h, "decoder.conv_norm_out"))
        return self.conv(h, "decoder.conv_out", 3)


def build_vae_decoder(
    weights: ComponentWeights,
    config: dict,
    *,
    latent_height: int,
    latent_width: int,
    latent_dtype: trt.DataType,
    optimization_level: int,
    verbose: bool = False,
) -> bytes:
    """Serialized engine: ``latents [1, N, 128]`` float32 → ``image [1, 3, H, W]`` float32."""

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = new_network(builder)
    g = Graph(network, trt.float32)
    patch_h, patch_w = (int(value) for value in config["patch_size"])
    latent_channels = int(config["latent_channels"])
    packed = latent_channels * patch_h * patch_w
    tokens = latent_height * latent_width

    latents = network.add_input("latents", trt.float32, (1, tokens, packed))
    grid = g.permute(g.reshape(latents, (1, latent_height, latent_width, packed)), (0, 3, 1, 2))
    mean = weights.get("bn.running_mean", (packed,)).astype(np.float32)
    var = weights.get("bn.running_var", (packed,)).astype(np.float32)
    std = np.sqrt(var + np.float32(config["batch_norm_eps"])).astype(np.float32)
    grid = g.cast(grid, latent_dtype)
    grid = g.add(
        g.mul(grid, g.constant(std.reshape(1, -1, 1, 1), latent_dtype)),
        g.constant(mean.reshape(1, -1, 1, 1), latent_dtype),
    )
    unpatched = g.reshape(grid, (1, latent_channels, patch_h, patch_w, latent_height, latent_width))
    unpatched = g.permute(unpatched, (0, 1, 4, 2, 5, 3))
    z = g.cast(
        g.reshape(unpatched, (1, latent_channels, latent_height * patch_h, latent_width * patch_w)),
        trt.float32,
    )

    image = _Decoder(g, weights, config).decode(z, config)
    image.name = "image"
    network.mark_output(image)
    upscale = 2 ** (len(config["up_block_types"]) - 1)
    if tuple(image.shape[2:]) != (
        latent_height * patch_h * upscale,
        latent_width * patch_w * upscale,
    ):
        raise RuntimeError("flux3 VAE output size does not match the latent grid")
    return serialize(builder, network, new_builder_config(builder, optimization_level))
