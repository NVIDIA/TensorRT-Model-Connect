# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The SD3 denoiser (``SD3Transformer2DModel``) as one engine.

Image patches and text tokens keep separate weights but attend jointly
(MMDiT). SD3.5 adds a second, image-only self-attention to the blocks listed
in ``dual_attention_layers`` (MMDiT-X) and RMS-normalizes queries and keys.
The last block computes only the image stream.

The engine takes the sinusoidal timestep projection rather than the
timestep: the runtime evaluates the sines on the host in FP32.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .graph_ops import Graph, build_engine
from .weights import take

LN_EPS = 1e-6
RMS_EPS = 1e-6
TIMESTEP_CHANNELS = 256


def cropped_position_embedding(weights: dict, cfg: dict, patches_h: int, patches_w: int):
    """The centre crop of the stored position-embedding grid, [1, patches_h * patches_w, width]."""
    grid = cfg["pos_embed_max_size"]
    width = cfg["width"]
    table = np.asarray(take(weights, "pos_embed.pos_embed")).reshape(grid, grid, width)
    top = (grid - patches_h) // 2
    left = (grid - patches_w) // 2
    crop = table[top:top + patches_h, left:left + patches_w]
    return np.ascontiguousarray(crop.reshape(1, patches_h * patches_w, width))


class _Block:
    """Weight access and layer helpers for one transformer block."""

    def __init__(self, graph: Graph, weights: dict, cfg: dict, index: int) -> None:
        self.graph = graph
        self.weights = weights
        self.prefix = f"transformer_blocks.{index}."
        self.heads = cfg["num_heads"]
        self.head_dim = cfg["head_dim"]
        self.width = cfg["width"]
        self.qk_norm = cfg["qk_norm"]

    def w(self, name: str):
        return take(self.weights, self.prefix + name)

    def linear(self, tensor, name: str):
        return self.graph.linear(tensor, self.w(name + ".weight"), self.w(name + ".bias"))

    def chunks(self, emb, count: int):
        """``emb.chunk(count, dim=1)``, each reshaped to [B, 1, width] for broadcasting."""
        width = self.width
        return [self.graph.reshape(self.graph.slice_last(emb, index * width, width), (0, 1, width))
                for index in range(count)]

    def modulate(self, normed, shift, scale):
        """``normed * (1 + scale) + shift``."""
        graph = self.graph
        return graph.add(graph.mul(normed, graph.add(scale, graph.scalar(1.0, scale))), shift)

    def heads_of(self, tensor, norm_name: str | None):
        graph = self.graph
        split = graph.split_heads(tensor, self.heads, self.head_dim)
        if self.qk_norm is None or norm_name is None:
            return split
        return graph.rms_norm(split, RMS_EPS, self.w(norm_name + ".weight"))

    def feed_forward(self, tensor, name: str):
        inner = self.graph.gelu_tanh(self.linear(tensor, name + ".net.0.proj"))
        return self.linear(inner, name + ".net.2")


def _joint_block(block: _Block, hidden, context, silu_temb, *, dual: bool, last: bool, image_tokens: int):
    graph = block.graph
    scale = block.head_dim ** -0.5

    norm1 = block.linear(silu_temb, "norm1.linear")
    if dual:
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp,
         shift_msa2, scale_msa2, gate_msa2) = block.chunks(norm1, 9)
    else:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.chunks(norm1, 6)
    normed_hidden = graph.layer_norm(hidden, LN_EPS)
    attn_input = block.modulate(normed_hidden, shift_msa, scale_msa)

    normed_context = graph.layer_norm(context, LN_EPS)
    context_mod = block.linear(silu_temb, "norm1_context.linear")
    if last:
        # AdaLayerNormContinuous: the projection holds (scale, shift), in that order.
        context_scale, context_shift = block.chunks(context_mod, 2)
        context_input = block.modulate(normed_context, context_shift, context_scale)
    else:
        (c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp,
         c_gate_mlp) = block.chunks(context_mod, 6)
        context_input = block.modulate(normed_context, c_shift_msa, c_scale_msa)

    query = graph.concat([block.heads_of(block.linear(attn_input, "attn.to_q"), "attn.norm_q"),
                          block.heads_of(block.linear(context_input, "attn.add_q_proj"),
                                         "attn.norm_added_q")], 2)
    key = graph.concat([block.heads_of(block.linear(attn_input, "attn.to_k"), "attn.norm_k"),
                        block.heads_of(block.linear(context_input, "attn.add_k_proj"),
                                       "attn.norm_added_k")], 2)
    value = graph.concat([block.heads_of(block.linear(attn_input, "attn.to_v"), None),
                          block.heads_of(block.linear(context_input, "attn.add_v_proj"), None)], 2)
    joint = graph.merge_heads(graph.attention(query, key, value, scale=scale),
                              block.heads, block.head_dim)
    image_out = block.linear(graph.slice_axis(joint, 1, 0, image_tokens), "attn.to_out.0")
    hidden = graph.add(hidden, graph.mul(gate_msa, image_out))

    if dual:
        second_input = block.modulate(normed_hidden, shift_msa2, scale_msa2)
        second = graph.attention(
            block.heads_of(block.linear(second_input, "attn2.to_q"), "attn2.norm_q"),
            block.heads_of(block.linear(second_input, "attn2.to_k"), "attn2.norm_k"),
            block.heads_of(block.linear(second_input, "attn2.to_v"), None),
            scale=scale)
        second = block.linear(graph.merge_heads(second, block.heads, block.head_dim),
                              "attn2.to_out.0")
        hidden = graph.add(hidden, graph.mul(gate_msa2, second))

    ff_input = block.modulate(graph.layer_norm(hidden, LN_EPS), shift_mlp, scale_mlp)
    hidden = graph.add(hidden, graph.mul(gate_mlp, block.feed_forward(ff_input, "ff")))
    if last:
        return hidden, None

    context_tokens = int(tuple(joint.shape)[1]) - image_tokens
    context_out = block.linear(graph.slice_axis(joint, 1, image_tokens, context_tokens),
                               "attn.to_add_out")
    context = graph.add(context, graph.mul(c_gate_msa, context_out))
    ff_context = block.modulate(graph.layer_norm(context, LN_EPS), c_shift_mlp, c_scale_mlp)
    context = graph.add(context, graph.mul(c_gate_mlp, block.feed_forward(ff_context, "ff_context")))
    return hidden, context


def denoise(graph: Graph, latents, encoder_hidden_states, pooled, timestep_proj,
            weights: dict, cfg: dict, latent_h: int, latent_w: int):
    """Velocity prediction [B, C, latent_h, latent_w] (FP32)."""
    width = cfg["width"]
    patch = cfg["patch_size"]
    channels = cfg["in_channels"]
    patches_h, patches_w = latent_h // patch, latent_w // patch
    image_tokens = patches_h * patches_w

    def linear(tensor, name):
        return graph.linear(tensor, take(weights, name + ".weight"), take(weights, name + ".bias"))

    patches = graph.conv2d(latents, take(weights, "pos_embed.proj.weight"),
                           take(weights, "pos_embed.proj.bias"), stride=patch)
    hidden = graph.reshape(patches, (0, width, image_tokens), second_transpose=(0, 2, 1))
    hidden = graph.add(hidden, graph.constant(
        cropped_position_embedding(weights, cfg, patches_h, patches_w), dtype=trt.float32))

    temb = linear(graph.silu(linear(timestep_proj, "time_text_embed.timestep_embedder.linear_1")),
                  "time_text_embed.timestep_embedder.linear_2")
    pooled_emb = linear(graph.silu(linear(pooled, "time_text_embed.text_embedder.linear_1")),
                        "time_text_embed.text_embedder.linear_2")
    silu_temb = graph.silu(graph.add(temb, pooled_emb))

    context = linear(encoder_hidden_states, "context_embedder")
    dual_layers = set(cfg["dual_attention_layers"])
    last = cfg["num_layers"] - 1
    for index in range(cfg["num_layers"]):
        block = _Block(graph, weights, cfg, index)
        hidden, context = _joint_block(block, hidden, context, silu_temb,
                                       dual=index in dual_layers, last=index == last,
                                       image_tokens=image_tokens)

    norm_out = linear(silu_temb, "norm_out.linear")
    out_scale = graph.reshape(graph.slice_last(norm_out, 0, width), (0, 1, width))
    out_shift = graph.reshape(graph.slice_last(norm_out, width, width), (0, 1, width))
    hidden = graph.add(graph.mul(graph.layer_norm(hidden, LN_EPS),
                                 graph.add(out_scale, graph.scalar(1.0, out_scale))), out_shift)
    hidden = linear(hidden, "proj_out")
    # [B, h, w, p, q, c] -> [B, c, h, p, w, q] -> [B, c, h * p, w * q]
    hidden = graph.reshape(hidden, (0, patches_h, patches_w, patch, patch, channels),
                           second_transpose=(0, 5, 1, 3, 2, 4))
    return graph.reshape(hidden, (0, channels, latent_h, latent_w))


def build_transformer_engine(weights: dict, cfg: dict, *, latent_h: int, latent_w: int,
                             text_tokens: int, max_batch: int, precision: str, verbose: bool):
    channels = cfg["in_channels"]
    shapes = {
        "hidden_states": (channels, latent_h, latent_w),
        "encoder_hidden_states": (text_tokens, cfg["joint_attention_dim"]),
        "pooled_projections": (cfg["pooled_projection_dim"],),
        "timestep_proj": (TIMESTEP_CHANNELS,),
    }

    def populate(graph: Graph) -> None:
        inputs = {name: graph.network.add_input(name, trt.float32, (-1, *shape))
                  for name, shape in shapes.items()}
        velocity = denoise(graph, inputs["hidden_states"], inputs["encoder_hidden_states"],
                           inputs["pooled_projections"], inputs["timestep_proj"],
                           weights, cfg, latent_h, latent_w)
        graph.output(velocity, "velocity")

    profile = {name: ((1, *shape), (max_batch, *shape), (max_batch, *shape))
               for name, shape in shapes.items()}
    return build_engine(populate, precision=precision, verbose=verbose, profile=profile)
