# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP32 seven-latent video-VAE tiles, including the learned register tokens."""

import numpy as np
import tensorrt as trt

LATENT_FRAMES = 7
PIXELS = LATENT_FRAMES * 16 * 16
ROWS = PIXELS + 5


def input_graph(g, cfg):
    hidden = g.net.add_input("latents", trt.float32, (PIXELS, 24))
    weight = g.weights.read("post_quant_conv.weight").reshape(24, 24)
    hidden = g.add(
        g.mm(hidden, g.const(weight, trt.float32), True),
        g.const(g.weights.read("post_quant_conv.bias").reshape(1, 24), trt.float32),
    )
    hidden = g.linear(hidden, "decoder.proj_in")
    registers = g.const(g.weights.read("decoder.register_tokens").reshape(4, -1), trt.float32)
    zero = g.const(np.zeros((1, 2048), np.float32), trt.float32)
    g.output(g.concat([hidden, registers, zero], 0), "result")


def layer_graph(g, cfg, index):
    prefix = f"decoder.transformer_blocks.{index}"
    hidden = g.net.add_input("hidden", trt.float32, (ROWS, 2048))
    axes = [2 * (np.arange(size, dtype=np.float32) + 0.5) / size - 1 for size in (7, 16, 16)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(PIXELS, 3)
    grid = np.concatenate([grid, np.zeros((5, 3), np.float32)])
    rotary_dim = int(cfg["decoder_attention_head_dim"] * cfg["decoder_rope_dim_ratio"])
    frequency = 1 / cfg["decoder_rope_theta"] ** np.arange(0, 1, 6 / rotary_dim, dtype=np.float32)
    angles = (2 * np.float32(np.pi) * grid[..., None] * frequency).reshape(ROWS, -1)
    angles = np.concatenate([angles, angles], -1).reshape(1, ROWS, 1, rotary_dim)
    cos = g.const(np.cos(angles), trt.float32)
    sin = g.const(np.sin(angles), trt.float32)
    normal = g.norm(hidden, prefix + ".norm1", cfg["decoder_norm_eps"])
    attended = g.attention(
        normal,
        prefix + ".attn",
        32,
        64,
        cfg["decoder_norm_eps"],
        cos=cos,
        sin=sin,
        rotary_dim=rotary_dim,
        bias=True,
        affine=False,
    )
    hidden = g.add(
        hidden,
        g.mul(attended, g.const(g.weights.read(prefix + ".scale1").reshape(1, -1), trt.float32)),
    )
    normal = g.norm(hidden, prefix + ".norm2", cfg["decoder_norm_eps"])
    forward = g.ff(normal, prefix + ".ff", 8192, bias=True)
    g.output(
        g.add(
            hidden,
            g.mul(forward, g.const(g.weights.read(prefix + ".scale2").reshape(1, -1), trt.float32)),
        ),
        "result",
    )


def head_graph(g, cfg):
    hidden = g.net.add_input("hidden", trt.float32, (ROWS, 2048))
    normal = g.norm(hidden, "decoder.norm_out", cfg["decoder_norm_eps"], layer=True)
    g.output(g.linear(normal, "decoder.proj_out"), "result")
