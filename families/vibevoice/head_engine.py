# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Diffusion head engine: one v-prediction for up to two conditions.

The runtime stacks the conditional and the unconditional LM state, feeds the
same noisy latent twice and combines the two predictions itself, exactly as
the reference's classifier-free guidance does. Latents and timesteps cross the
boundary as float32; the graph rounds them to the activation type first, as
the reference holds them in the model's dtype.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from . import graph as g


def timestep_frequencies(cfg) -> np.ndarray:
    """``exp(-ln(max_period) * arange(half) / half)`` in float32, as the reference builds it."""
    half = cfg.frequency_embedding_size // 2
    exponent = np.float32(-math.log(cfg.diffusion_max_period)) * np.arange(half, dtype=np.float32)
    return np.exp(exponent / np.float32(half)).astype(np.float32)


def build_head_engine(weights, cfg, *, dtype, verbose: bool = False) -> bytes:
    builder, network, config = g.builder_and_network(verbose)
    net = g.Net(network, dtype)
    hidden = cfg.hidden_size
    prefix = "model.prediction_head."

    noisy = network.add_input("noisy_latent", trt.float32, (-1, cfg.latent_size))
    timestep = network.add_input("timestep", trt.float32, (-1,))
    condition = network.add_input("condition", dtype, (-1, hidden))

    state = net.linear(net.cast(noisy, dtype), weights.f32(prefix + "noisy_images_proj.weight"))
    # The reference casts the timesteps to the model dtype before embedding them.
    steps = net.cast(net.cast(timestep, dtype), trt.float32)
    angles = net.mul(
        net.reshape(steps, (-1, 1)),
        net.const(timestep_frequencies(cfg)[None, :], dtype=trt.float32),
    )
    embedded = net.concat(
        [net.unary(angles, trt.UnaryOperation.COS), net.unary(angles, trt.UnaryOperation.SIN)], 1
    )
    embedded = net.cast(embedded, dtype)
    embedded = net.linear(
        net.silu(net.linear(embedded, weights.f32(prefix + "t_embedder.mlp.0.weight"))),
        weights.f32(prefix + "t_embedder.mlp.2.weight"),
    )
    cond = net.add(net.linear(condition, weights.f32(prefix + "cond_proj.weight")), embedded)
    activated = net.silu(cond)
    one = net.scalar(1.0, 2)

    for layer in range(cfg.head_layers):
        base = f"{prefix}layers.{layer}."
        modulation = net.linear(activated, weights.f32(base + "adaLN_modulation.1.weight"))
        shift, scale, gate = _chunks(net, modulation, 3, hidden)
        normed = net.rms_norm(state, weights.f32(base + "norm.weight"), cfg.head_eps, -1)
        modulated = net.add(net.mul(normed, net.add(one, scale)), shift)
        inner = net.mul(
            net.silu(net.linear(modulated, weights.f32(base + "ffn.gate_proj.weight"))),
            net.linear(modulated, weights.f32(base + "ffn.up_proj.weight")),
        )
        state = net.add(
            state, net.mul(gate, net.linear(inner, weights.f32(base + "ffn.down_proj.weight")))
        )

    modulation = net.linear(
        activated, weights.f32(prefix + "final_layer.adaLN_modulation.1.weight")
    )
    shift, scale = _chunks(net, modulation, 2, hidden)
    normed = net.rms_norm(state, None, cfg.head_eps, -1)
    state = net.add(net.mul(normed, net.add(one, scale)), shift)
    velocity = net.cast(
        net.linear(state, weights.f32(prefix + "final_layer.linear.weight")), trt.float32
    )
    velocity.name = "velocity"
    network.mark_output(velocity)

    profile = builder.create_optimization_profile()
    profile.set_shape(
        "noisy_latent", (1, cfg.latent_size), (2, cfg.latent_size), (2, cfg.latent_size)
    )
    profile.set_shape("timestep", (1,), (2,), (2,))
    profile.set_shape("condition", (1, hidden), (2, hidden), (2, hidden))
    config.add_optimization_profile(profile)
    return g.serialize(builder, network, config, "diffusion head")


def _chunks(net: g.Net, tensor: trt.ITensor, count: int, width: int) -> list[trt.ITensor]:
    """``tensor.chunk(count, dim=-1)`` for a ``[N, count * width]`` tensor with dynamic N."""
    stacked = net.reshape(tensor, (-1, count, width))
    return [
        net.network.add_gather(
            stacked, net.const(np.array(index, dtype=np.int32), (), trt.int32), 1
        ).get_output(0)
        for index in range(count)
    ]
