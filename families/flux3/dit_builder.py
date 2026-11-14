# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flux3 flow-map transformer (text-to-image path) as one TensorRT engine.

Graph of Diffusers ``Flux3Transformer2DModel.forward`` without reference image
tokens: per-token timestep embeddings (image tokens at the flow timestep, text
at 0, both plus the target-timestep embedding), per-mode blocks that attend
within the image or the text tokens only, joint single blocks over
``[text, image]`` with per-row modulation, and the output layer on the image
rows. QK RMSNorm and RoPE run in float32 with one cast back, as in the
reference. The text window is a dynamic dimension.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from . import rope
from .trt_ops import Graph, new_builder_config, new_network, serialize
from .weights import ComponentWeights


class _Dit:
    def __init__(self, g: Graph, weights: ComponentWeights, config: dict) -> None:
        self.g = g
        self.w = weights
        self.heads = int(config["num_attention_heads"])
        self.head_dim = int(config["attention_head_dim"])
        self.dim = self.heads * self.head_dim
        self.inner = int(self.dim * float(config["mlp_ratio"]))
        self.eps = float(config["eps"])
        self.one = g.scalar(1.0, g.compute, 2)
        self.rot_index, rot_sign = rope.interleaved_rotation(self.head_dim)
        self.rot_sign = g.constant(rot_sign.reshape(1, 1, -1))

    def linear(self, x: trt.ITensor, name: str, out_dim: int, in_dim: int) -> trt.ITensor:
        return self.g.linear(x, self.w.get(name, (out_dim, in_dim)))

    def time_embedding(self, proj: trt.ITensor, prefix: str) -> trt.ITensor:
        g = self.g
        hidden = g.silu(
            self.linear(
                g.cast(proj, g.compute), prefix + "linear_1.weight", self.dim, proj.shape[-1]
            )
        )
        return self.linear(hidden, prefix + "linear_2.weight", self.dim, self.dim)

    def modulation(self, activated: trt.ITensor, name: str) -> list[trt.ITensor]:
        """``[1, 3 * dim]`` → ``[shift, scale, gate]``, each ``[1, dim]``."""

        return self.g.chunk_last(self.linear(activated, name, 3 * self.dim, self.dim), 3)

    def _qk(self, x: trt.ITensor, norm: str, cos: trt.ITensor, sin: trt.ITensor) -> trt.ITensor:
        g = self.g
        x32 = g.rms_norm32(g.heads(x, self.heads, self.head_dim), self._norm_weight(norm), self.eps)
        rotated = g.mul(g.take_last(x32, self.rot_index), self.rot_sign)
        return g.add(g.mul(x32, cos), g.mul(rotated, sin))

    def _norm_weight(self, name: str) -> np.ndarray:
        return self.w.get(name, (self.head_dim,)).astype(np.float32)

    def attention(self, n: trt.ITensor, p: str, cos: trt.ITensor, sin: trt.ITensor) -> trt.ITensor:
        g = self.g
        q32 = self._qk(
            self.linear(n, p + "attn.to_q.weight", self.dim, self.dim),
            p + "attn.norm_q.weight",
            cos,
            sin,
        )
        k32 = self._qk(
            self.linear(n, p + "attn.to_k.weight", self.dim, self.dim),
            p + "attn.norm_k.weight",
            cos,
            sin,
        )
        v = g.heads(
            self.linear(n, p + "attn.to_v.weight", self.dim, self.dim), self.heads, self.head_dim
        )
        q = g.to_attention_layout(g.scale_query(q32, self.head_dim, g.compute))
        k = g.to_attention_layout(g.cast(k32, g.compute))
        out = g.from_attention_layout(g.attention(q, k, g.to_attention_layout(v)), self.dim)
        return self.linear(out, p + "attn.to_out.0.weight", self.dim, self.dim)

    def feed_forward(self, n: trt.ITensor, p: str) -> trt.ITensor:
        g = self.g
        weight = self.w.get(p + "ff.linear_in.weight", (2 * self.inner, self.dim))
        gate = g.silu(g.linear(n, weight[: self.inner]))
        up = g.linear(n, weight[self.inner :])
        return self.linear(g.mul(gate, up), p + "ff.linear_out.weight", self.dim, self.inner)

    def block(self, x: trt.ITensor, mods: list[trt.ITensor], p: str, cos, sin) -> trt.ITensor:
        """Parallel attention + feed-forward from one modulated LayerNorm, one gated residual."""

        g = self.g
        shift, scale, gate = mods
        n = g.add(g.mul(g.add(self.one, scale), g.layer_norm(x, self.eps)), shift)
        return g.add(x, g.mul(gate, g.add(self.attention(n, p, cos, sin), self.feed_forward(n, p))))

    def output(self, x: trt.ITensor, temb: trt.ITensor, out_channels: int) -> trt.ITensor:
        g = self.g
        emb = self.linear(g.silu(temb), "norm_out.linear.weight", 2 * self.dim, self.dim)
        scale, shift = g.chunk_last(emb, 2)
        n = g.add(g.mul(g.layer_norm(x, self.eps), g.add(self.one, scale)), shift)
        return self.linear(n, "proj_out.weight", out_channels, self.dim)


def _rows_of_ones(g: Graph, rows: trt.ITensor) -> trt.ITensor:
    """``[rows, 1]`` ones in the compute dtype for a 1-element shape tensor ``rows``."""

    shape = g.concat([rows, g.constant(np.array([1], dtype=np.int64))], axis=0)
    layer = g.net.add_fill(trt.Dims([2]), trt.FillOperation.LINSPACE, trt.float32)
    layer.set_input(0, shape)
    layer.set_input(1, g.constant(np.array(1.0, dtype=np.float32)))
    layer.set_input(2, g.constant(np.zeros(2, dtype=np.float32)))
    return g.cast(layer.get_output(0), g.compute)


def _joint_mods(g: Graph, txt: list, img: list, txt_ones: trt.ITensor, img_ones: trt.ITensor):
    return [g.concat([g.mul(txt_ones, t), g.mul(img_ones, i)], axis=0) for t, i in zip(txt, img)]


def build_denoiser(
    weights: ComponentWeights,
    config: dict,
    *,
    latent_height: int,
    latent_width: int,
    rope_scale: float,
    max_tokens: int,
    opt_tokens: int,
    min_tokens: int,
    compute: trt.DataType,
    optimization_level: int,
    verbose: bool = False,
) -> bytes:
    """Serialized engine for one image geometry and a dynamic text window."""

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = new_network(builder)
    g = Graph(network, compute)
    dit = _Dit(g, weights, config)
    in_channels = int(config["in_channels"])
    text_width = int(config["joint_attention_dim"])
    channels = int(config["timestep_channels"])
    tokens = latent_height * latent_width
    axes = tuple(int(value) for value in config["axes_dims_rope"])
    theta = float(config["rope_theta"])

    latents = network.add_input("latents", trt.float32, (1, tokens, in_channels))
    text = network.add_input("text_embeds", compute, (1, -1, text_width))
    t_proj = network.add_input("timestep_proj", trt.float32, (1, channels))
    target_proj = network.add_input("target_timestep_proj", trt.float32, (1, channels))

    zero_proj = np.concatenate([np.ones(channels // 2), np.zeros(channels // 2)]).astype(np.float32)
    target_emb = dit.time_embedding(target_proj, "target_timestep_embedder.")
    image_temb = g.add(dit.time_embedding(t_proj, "timestep_embedder."), target_emb)
    zero_temb = g.add(
        dit.time_embedding(g.constant(zero_proj.reshape(1, -1)), "timestep_embedder."), target_emb
    )
    image_act, zero_act = g.silu(image_temb), g.silu(zero_temb)

    img = dit.linear(
        g.cast(g.reshape(latents, (tokens, in_channels)), compute),
        "x_embedders.image.weight",
        dit.dim,
        in_channels,
    )
    txt = dit.linear(
        g.reshape(text, (-1, text_width)), "context_embedder.weight", dit.dim, text_width
    )
    text_rows = g.dim(text, 1)

    img_cos, img_sin = rope.interleaved_tables(
        rope.image_ids(latent_height, latent_width, rope_scale), axes, theta
    )
    txt_cos_table, txt_sin_table = rope.interleaved_tables(rope.text_ids(max_tokens), axes, theta)
    text_positions = g.iota(text_rows)
    img_cos_t = g.constant(img_cos.reshape(tokens, 1, -1))
    img_sin_t = g.constant(img_sin.reshape(tokens, 1, -1))
    txt_cos_t = g.reshape(
        g.gather(g.constant(txt_cos_table), text_positions, 0), (-1, 1, dit.head_dim)
    )
    txt_sin_t = g.reshape(
        g.gather(g.constant(txt_sin_table), text_positions, 0), (-1, 1, dit.head_dim)
    )

    txt_early = dit.modulation(zero_act, "early_stream_modulations.txt.linear.weight")
    img_early = dit.modulation(image_act, "early_stream_modulations.image.linear.weight")
    for depth in range(int(config["num_layers"])):
        txt = dit.block(txt, txt_early, f"txt_mode_blocks.{depth}.", txt_cos_t, txt_sin_t)
        img = dit.block(img, img_early, f"mode_blocks.image.{depth}.", img_cos_t, img_sin_t)

    txt_single = dit.modulation(zero_act, "single_stream_modulations.txt.linear.weight")
    img_single = dit.modulation(image_act, "single_stream_modulations.image.linear.weight")
    img_ones = g.constant(np.ones((tokens, 1), dtype=np.float32), compute)
    joint_mods = _joint_mods(g, txt_single, img_single, _rows_of_ones(g, text_rows), img_ones)
    joint = g.concat([txt, img], axis=0)
    joint_cos = g.concat([txt_cos_t, img_cos_t], axis=0)
    joint_sin = g.concat([txt_sin_t, img_sin_t], axis=0)
    for index in range(int(config["num_single_layers"])):
        joint = dit.block(
            joint, joint_mods, f"single_transformer_blocks.{index}.", joint_cos, joint_sin
        )

    start = g.concat([text_rows, g.constant(np.array([0], dtype=np.int64))], axis=0)
    image_rows = network.add_slice(joint, (0, 0), (tokens, dit.dim), (1, 1))
    image_rows.set_input(1, start)
    velocity = dit.output(image_rows.get_output(0), image_temb, in_channels)
    velocity = g.reshape(g.cast(velocity, trt.float32), (1, tokens, in_channels))
    velocity.name = "velocity"
    network.mark_output(velocity)

    config_trt = new_builder_config(builder, optimization_level)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "text_embeds",
        (1, min_tokens, text_width),
        (1, opt_tokens, text_width),
        (1, max_tokens, text_width),
    )
    config_trt.add_optimization_profile(profile)
    return serialize(builder, network, config_trt)
