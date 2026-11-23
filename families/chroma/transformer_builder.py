# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chroma MMDiT denoiser graph (``ChromaTransformer2DModel``).

Engine I/O (one denoiser call, batch 1):

* ``hidden_states`` ``[1, N, in_channels]`` float32 - packed 2x2 latent patches
* ``encoder_hidden_states`` ``[1, L, joint_attention_dim]`` float32 - T5 output
* ``timestep_features`` ``[1, 16]`` float32 - ``[cos, sin]`` sinusoid of the timestep
* ``text_mask`` ``[1, L]`` float32 - 1 for the prompt tokens and the first padding token
* ``noise_pred`` ``[1, N, in_channels]`` float32 out

Every block reads its modulation rows from one approximator output computed
from the timestep embedding. The text/image attention mask is the reference
pipeline's: a 0/1 tensor ``m`` of text-mask and all-ones image entries enters
scaled-dot-product attention as the floating-point additive term
``m[i] * m[j]``, not as a boolean mask.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .checkpoint import SafetensorsDirectory, TransformerConfig
from .graph_ops import Graph, timestep_embedding

NORM_EPS = 1e-6
QK_NORM_EPS = 1e-6
TIMESTEP_FEATURES = 16
GUIDANCE_FEATURES = 16
MODULATION_FEATURES = 32
# torch.nn.RMSNorm(eps=None) in the approximator uses the float32 epsilon, for BF16 inputs too.
APPROXIMATOR_EPS = float(np.finfo(np.float32).eps)


def rope_tables(config: TransformerConfig, text_length: int, grid_height: int, grid_width: int):
    """``FluxPosEmbed(theta=10000)`` cos/sin for zero text ids then (0, row, column) image ids."""
    ids = np.zeros((text_length + grid_height * grid_width, 3), dtype=np.float64)
    rows, columns = np.meshgrid(np.arange(grid_height), np.arange(grid_width), indexing="ij")
    ids[text_length:, 1] = rows.reshape(-1)
    ids[text_length:, 2] = columns.reshape(-1)
    cos_parts, sin_parts = [], []
    for axis, dim in enumerate(config.axes_dims_rope):
        frequencies = 1.0 / (10000.0 ** (np.arange(0, dim, 2, dtype=np.float64)[: dim // 2] / dim))
        angles = np.outer(ids[:, axis].astype(np.float32).astype(np.float64), frequencies)
        cos_parts.append(np.repeat(np.cos(angles), 2, axis=1))
        sin_parts.append(np.repeat(np.sin(angles), 2, axis=1))
    return (
        np.concatenate(cos_parts, axis=1).astype(np.float32),
        np.concatenate(sin_parts, axis=1).astype(np.float32),
    )


def modulation_index_features(rows: int) -> np.ndarray:
    """The approximator's ``mod_proj`` buffer: sinusoid of ``arange(rows) * 1000``."""
    return timestep_embedding(np.arange(rows, dtype=np.int64).astype(np.float32) * 1000.0, MODULATION_FEATURES)


class _Denoiser:
    def __init__(self, graph: Graph, weights: SafetensorsDirectory, config: TransformerConfig,
                 text_length: int, image_tokens: int) -> None:
        self.g = graph
        self.w = weights
        self.c = config
        self.text_length = text_length
        self.image_tokens = image_tokens
        self.tokens = text_length + image_tokens

    # ---------------------------------------------------------------- pieces
    def linear(self, x: trt.ITensor, name: str) -> trt.ITensor:
        return self.g.linear(x, self.w[f"{name}.weight"], self.w[f"{name}.bias"])

    def row(self, modulation: trt.ITensor, index: int) -> trt.ITensor:
        """One ``[1, 1, inner]`` modulation vector."""
        return self.g.slice(modulation, (0, index, 0), (1, 1, self.c.inner_dim))

    def modulate(self, x: trt.ITensor, shift: trt.ITensor, scale: trt.ITensor) -> trt.ITensor:
        normalized = self.g.layer_norm(x, NORM_EPS)
        one_plus = self.g.add(scale, self.g.scalar(1.0, scale))
        return self.g.add(self.g.mul(normalized, one_plus), shift)

    def heads(self, x: trt.ITensor, length: int) -> trt.ITensor:
        return self.g.reshape(x, (1, length, self.c.num_heads, self.c.head_dim))

    def rope(self, x: trt.ITensor, cos: trt.ITensor, sin: trt.ITensor, scale: float) -> trt.ITensor:
        """Interleaved-pair rotary embedding in FP32 on ``[1, S, H, D]``; ``scale`` folds the softmax scale."""
        g = self.g
        half = self.c.head_dim // 2
        x32 = g.cast(x, trt.float32)
        pairs = g.reshape(x32, (1, self.tokens, self.c.num_heads, half, 2))
        even = g.slice(pairs, (0, 0, 0, 0, 0), (1, self.tokens, self.c.num_heads, half, 1))
        odd = g.slice(pairs, (0, 0, 0, 0, 1), (1, self.tokens, self.c.num_heads, half, 1))
        negative_odd = g.mul(odd, g.scalar(-1.0, odd))
        rotated = g.reshape(g.concat([negative_odd, even], axis=4), (1, self.tokens, self.c.num_heads, self.c.head_dim))
        result = g.add(g.mul(x32, cos), g.mul(rotated, sin))
        if scale != 1.0:
            result = g.mul(result, g.scalar(scale, result))
        return g.cast(result, x.dtype)

    def attend(self, query: trt.ITensor, key: trt.ITensor, value: trt.ITensor, rope, mask: trt.ITensor) -> trt.ITensor:
        """``[1, S, H, D]`` inputs after q/k norm -> ``[1, S, H * D]`` attention output."""
        g = self.g
        cos, sin = rope
        query = self.rope(query, cos, sin, float(self.c.head_dim) ** -0.5)
        key = self.rope(key, cos, sin, 1.0)
        q, k, v = (g.transpose(item, (0, 2, 1, 3)) for item in (query, key, value))
        context = g.attention(q, k, v, mask)
        return g.reshape(g.transpose(context, (0, 2, 1, 3)), (1, self.tokens, self.c.inner_dim))

    def qk_norm(self, x: trt.ITensor, name: str) -> trt.ITensor:
        # torch.nn.RMSNorm: weight applied in FP32, one rounding.
        return self.g.rms_norm(x, self.w[f"{name}.weight"], QK_NORM_EPS, scale_in_float32=True)

    def feed_forward(self, x: trt.ITensor, name: str) -> trt.ITensor:
        hidden = self.g.gelu_tanh(self.linear(x, f"{name}.net.0.proj"))
        return self.linear(hidden, f"{name}.net.2")

    # ---------------------------------------------------------------- blocks
    def double_block(self, index: int, image, text, modulation, rope, mask):
        g, c = self.g, self.c
        name = f"transformer_blocks.{index}"
        image_row = 3 * c.num_single_layers + 6 * index
        text_row = 3 * c.num_single_layers + 6 * c.num_layers + 6 * index
        i_mod = [self.row(modulation, image_row + offset) for offset in range(6)]
        t_mod = [self.row(modulation, text_row + offset) for offset in range(6)]

        image_in = self.modulate(image, i_mod[0], i_mod[1])
        text_in = self.modulate(text, t_mod[0], t_mod[1])
        query = g.concat([
            self.qk_norm(self.heads(self.linear(text_in, f"{name}.attn.add_q_proj"), self.text_length),
                         f"{name}.attn.norm_added_q"),
            self.qk_norm(self.heads(self.linear(image_in, f"{name}.attn.to_q"), self.image_tokens),
                         f"{name}.attn.norm_q"),
        ], axis=1)
        key = g.concat([
            self.qk_norm(self.heads(self.linear(text_in, f"{name}.attn.add_k_proj"), self.text_length),
                         f"{name}.attn.norm_added_k"),
            self.qk_norm(self.heads(self.linear(image_in, f"{name}.attn.to_k"), self.image_tokens),
                         f"{name}.attn.norm_k"),
        ], axis=1)
        value = g.concat([
            self.heads(self.linear(text_in, f"{name}.attn.add_v_proj"), self.text_length),
            self.heads(self.linear(image_in, f"{name}.attn.to_v"), self.image_tokens),
        ], axis=1)
        attended = self.attend(query, key, value, rope, mask)
        text_attn = g.slice(attended, (0, 0, 0), (1, self.text_length, c.inner_dim))
        image_attn = g.slice(attended, (0, self.text_length, 0), (1, self.image_tokens, c.inner_dim))

        image = g.add(image, g.mul(i_mod[2], self.linear(image_attn, f"{name}.attn.to_out.0")))
        image_ff = self.feed_forward(self.modulate(image, i_mod[3], i_mod[4]), f"{name}.ff")
        image = g.add(image, g.mul(i_mod[5], image_ff))

        text = g.add(text, g.mul(t_mod[2], self.linear(text_attn, f"{name}.attn.to_add_out")))
        text_ff = self.feed_forward(self.modulate(text, t_mod[3], t_mod[4]), f"{name}.ff_context")
        text = g.add(text, g.mul(t_mod[5], text_ff))
        return image, text

    def single_block(self, index: int, hidden, modulation, rope, mask):
        g = self.g
        name = f"single_transformer_blocks.{index}"
        shift, scale, gate = (self.row(modulation, 3 * index + offset) for offset in range(3))
        x = self.modulate(hidden, shift, scale)
        mlp = g.gelu_tanh(self.linear(x, f"{name}.proj_mlp"))
        query = self.qk_norm(self.heads(self.linear(x, f"{name}.attn.to_q"), self.tokens), f"{name}.attn.norm_q")
        key = self.qk_norm(self.heads(self.linear(x, f"{name}.attn.to_k"), self.tokens), f"{name}.attn.norm_k")
        value = self.heads(self.linear(x, f"{name}.attn.to_v"), self.tokens)
        attended = self.attend(query, key, value, rope, mask)
        projected = self.linear(g.concat([attended, mlp], axis=2), f"{name}.proj_out")
        return g.add(hidden, g.mul(gate, projected))

    def approximator(self, timestep_features: trt.ITensor) -> trt.ITensor:
        """``[1, rows, inner]`` modulation vectors from the timestep sinusoid."""
        g, c = self.g, self.c
        rows = c.modulation_rows
        time = g.reshape(timestep_features, (1, 1, TIMESTEP_FEATURES))
        time = g.mul(time, g.constant(np.ones((1, rows, 1), dtype=np.float32), trt.float32))
        # Chroma passes guidance 0: its sinusoid is cos(0) = 1 then sin(0) = 0.
        guidance = np.concatenate([np.ones(GUIDANCE_FEATURES // 2), np.zeros(GUIDANCE_FEATURES // 2)])
        guidance = g.constant(np.broadcast_to(guidance, (1, rows, GUIDANCE_FEATURES)).astype(np.float32), trt.float32)
        index = g.constant(modulation_index_features(rows)[None], trt.float32)
        features = g.cast(g.concat([time, guidance, index], axis=2), g.dtype)
        x = self.linear(features, "distilled_guidance_layer.in_proj")
        for layer in range(c.approximator_layers):
            normalized = g.rms_norm(x, self.w[f"distilled_guidance_layer.norms.{layer}.weight"], APPROXIMATOR_EPS,
                                    scale_in_float32=True)
            inner = g.silu(self.linear(normalized, f"distilled_guidance_layer.layers.{layer}.linear_1"))
            x = g.add(x, self.linear(inner, f"distilled_guidance_layer.layers.{layer}.linear_2"))
        return self.linear(x, "distilled_guidance_layer.out_proj")


def build_transformer_network(
    graph: Graph,
    weights: SafetensorsDirectory,
    config: TransformerConfig,
    text_length: int,
    grid_height: int,
    grid_width: int,
) -> None:
    network = graph.network
    image_tokens = grid_height * grid_width
    model = _Denoiser(graph, weights, config, text_length, image_tokens)

    hidden_states = network.add_input("hidden_states", trt.float32, (1, image_tokens, config.in_channels))
    encoder_states = network.add_input(
        "encoder_hidden_states", trt.float32, (1, text_length, config.joint_attention_dim)
    )
    timestep_features = network.add_input("timestep_features", trt.float32, (1, TIMESTEP_FEATURES))
    text_mask = network.add_input("text_mask", trt.float32, (1, text_length))

    modulation = model.approximator(timestep_features)
    image = model.linear(graph.cast(hidden_states, graph.dtype), "x_embedder")
    text = model.linear(graph.cast(encoder_states, graph.dtype), "context_embedder")

    cos, sin = rope_tables(config, text_length, grid_height, grid_width)
    rope = (
        graph.constant(cos[None, :, None, :], trt.float32),
        graph.constant(sin[None, :, None, :], trt.float32),
    )
    ones = graph.constant(np.ones((1, image_tokens), dtype=np.float32), trt.float32)
    sequence_mask = graph.concat([text_mask, ones], axis=1)
    mask = graph.mul(
        graph.reshape(sequence_mask, (1, 1, model.tokens, 1)),
        graph.reshape(sequence_mask, (1, 1, 1, model.tokens)),
    )
    mask = graph.cast(mask, graph.dtype)

    for index in range(config.num_layers):
        image, text = model.double_block(index, image, text, modulation, rope, mask)
    hidden = graph.concat([text, image], axis=1)
    for index in range(config.num_single_layers):
        hidden = model.single_block(index, hidden, modulation, rope, mask)
    image = graph.slice(hidden, (0, text_length, 0), (1, image_tokens, config.inner_dim))

    rows = config.modulation_rows
    final = model.modulate(image, model.row(modulation, rows - 2), model.row(modulation, rows - 1))
    output = graph.cast(model.linear(final, "proj_out"), trt.float32)
    output.name = "noise_pred"
    network.mark_output(output)
