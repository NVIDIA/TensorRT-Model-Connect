# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma 3n text decoder, expressed only with TensorRT graph operations.

One engine serves prefill and decode. Its caller supplies token ids, absolute
positions, an additive causal/validity mask over ``[cache rows, new rows]``,
image features for the image placeholder rows, and one K/V cache per layer that
owns one. Every decoder layer carries four AltUp streams; layers from the
sharing point on attend to the keys and values of the last earlier layer of the
same attention type. The engine returns the new cache rows and the soft-capped
logits of the last row.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from .graph import Builder

_TEXT = "model.language_model"


def rope_tables(theta: float, head_dim: int, capacity: int) -> tuple[np.ndarray, np.ndarray]:
    """``cos``/``sin`` of ``position * inv_freq`` over the full rotate-half width.

    Each float32 step of the reference (exponent, power, reciprocal, angle) is
    rounded to float32 here too, with the power correctly rounded; the cosine
    and sine of the float32 angle are evaluated in float64 and rounded once.
    """
    exponents = np.arange(0, head_dim, 2, dtype=np.int64).astype(np.float32) / np.float32(head_dim)
    powers = (np.float64(theta) ** exponents.astype(np.float64)).astype(np.float32)
    inv_freq = (np.float32(1.0) / powers).astype(np.float32)
    angles = np.arange(capacity, dtype=np.float32)[:, None] * inv_freq[None, :]
    angles = np.concatenate([angles, angles], axis=1).astype(np.float64)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def sparsity_multiplier(sparsity: float) -> float:
    """Standard-normal quantile of the target sparsity (float32, as the reference)."""
    return float(np.float32(math.sqrt(2.0) * _erfinv(2.0 * float(np.float32(sparsity)) - 1.0)))


def _erfinv(value: float) -> float:
    if not -1.0 < value < 1.0:
        raise ValueError("Gemma 3n activation sparsity must lie in (0, 1)")
    # Newton iterations on erf from a rational first guess; converges to
    # double precision in a few steps for every |value| < 1.
    guess = math.copysign(math.sqrt(-math.log((1.0 - value) * (1.0 + value))), value) * 0.886
    for _ in range(64):
        error = math.erf(guess) - value
        step = error / (2.0 / math.sqrt(math.pi) * math.exp(-guess * guess))
        guess -= step
        if abs(step) < 1e-17:
            break
    return guess


class TextGraph:
    def __init__(self, builder: Builder, config, capacity: int):
        self.b = builder
        self.config = config
        self.t = config.text
        self.capacity = capacity
        self.owners = self.t.kv_owners()

    def name(self, suffix: str) -> str:
        return f"{_TEXT}.{suffix}"

    # --- helpers ----------------------------------------------------------
    def int_constant(self, values):
        return self.b.constant(np.asarray(values, np.int32), trt.int32)

    def dynamic_reshape(self, tensor, prefix, suffix):
        """Reshape to ``prefix + shape(tensor) + suffix`` with dynamic dimensions kept."""
        n = self.b.net
        parts = []
        if prefix:
            parts.append(self.b.constant(np.asarray(prefix, np.int64), trt.int64))
        parts.append(n.add_shape(tensor).get_output(0))
        if suffix:
            parts.append(self.b.constant(np.asarray(suffix, np.int64), trt.int64))
        shape = self.b.concat(parts, 0)
        layer = n.add_shuffle(tensor)
        layer.set_input(1, shape)
        return layer.get_output(0)

    def magnitude(self, value_f32, floor=None):
        """``sqrt(mean(x^2))`` over the last axis, optionally with a floor on the mean."""
        b = self.b
        mean = b.mean(b.mul(value_f32, value_f32), -1)
        if floor is not None:
            mean = b.op(mean, b.scalar(floor, len(value_f32.shape), trt.float32),
                        trt.ElementWiseOperation.MAX)
        return b.unary(mean, trt.UnaryOperation.SQRT)

    def norm(self, value, weight: str | None):
        return self.b.rms_norm(value, self.t.rms_norm_eps,
                               None if weight is None else self.name(f"{weight}.weight"))

    def linear(self, value, weight: str, shape=None):
        return self.b.linear(value, self.name(weight), shape)

    def rope(self, value, heads, cos, sin):
        """Rotate-half RoPE on ``[T, heads, head_dim]``."""
        b, dim = self.b, self.t.head_dim
        halves = b.reshape(value, (0, heads, 2, dim // 2))
        first = b.gather(halves, self.int_constant([0]), 2)
        second = b.gather(halves, self.int_constant([1]), 2)
        rotated = b.reshape(b.concat([b.unary(second, trt.UnaryOperation.NEG), first], 2),
                            (0, heads, dim))
        return b.add(b.mul(value, cos), b.mul(rotated, sin))

    # --- embeddings -------------------------------------------------------
    def multimodal_hard_embedding(self, tokens, embedder: str, cfg):
        """``embed_{vision,audio}`` applied to token ids of its vocabulary slice."""
        b = self.b
        index = b.sub(tokens, self.int_constant([cfg.vocab_offset]))
        index = b.op(index, self.int_constant([0]), trt.ElementWiseOperation.MAX)
        index = b.op(index, self.int_constant([cfg.vocab_size - 1]), trt.ElementWiseOperation.MIN)
        table = b.tensor(f"model.{embedder}.embedding.weight", shape=(cfg.vocab_size, cfg.hidden_size))
        value = b.gather(table, index, 0)
        value = b.rms_norm(value, cfg.rms_norm_eps, f"model.{embedder}.hard_embedding_norm.weight")
        value = b.linear(value, f"model.{embedder}.embedding_projection.weight",
                         (self.t.hidden_size, cfg.hidden_size))
        return b.rms_norm(value, cfg.rms_norm_eps)

    def in_range(self, tokens, low, high):
        b = self.b
        above = b.op(tokens, self.int_constant([low - 1]), trt.ElementWiseOperation.GREATER)
        below = b.op(tokens, self.int_constant([high]), trt.ElementWiseOperation.LESS)
        return b.op(above, below, trt.ElementWiseOperation.AND)

    def embed(self, tokens, media, media_mask):
        b, t, cfg = self.b, self.t, self.config
        table = b.tensor(self.name("embed_tokens.weight"), shape=(t.vocab_size, t.hidden_size))
        text = b.scale(b.gather(table, tokens, 0), math.sqrt(t.hidden_size))
        vision = self.multimodal_hard_embedding(tokens, "embed_vision", cfg.vision)
        audio = self.multimodal_hard_embedding(tokens, "embed_audio", cfg.audio)
        select = b.net.add_select
        is_vision = b.reshape(self.in_range(tokens, cfg.vision.vocab_offset, cfg.audio.vocab_offset),
                              (-1, 1))
        is_audio = b.reshape(self.in_range(tokens, cfg.audio.vocab_offset, t.vocab_size), (-1, 1))
        value = select(is_vision, vision, text).get_output(0)
        value = select(is_audio, audio, value).get_output(0)
        # Image placeholder rows take the encoder's soft tokens.
        use_media = b.reshape(b.cast(media_mask, trt.bool), (-1, 1))
        return select(use_media, media, value).get_output(0), table

    def per_layer_inputs(self, tokens, embeds):
        """Per-layer token embeddings plus the projected input embedding, per layer."""
        b, t = self.b, self.t
        layers, width = t.num_layers, t.per_layer_hidden
        projected = self.linear(embeds, "per_layer_model_projection.weight",
                                (layers * width, t.hidden_size))
        projected = b.scale(projected, t.hidden_size ** -0.5)
        projected = self.norm(b.reshape(projected, (0, layers, width)), "per_layer_projection_norm")
        # Ids outside the per-layer vocabulary (multimodal ids) read row 0.
        in_table = self.in_range(tokens, 0, t.per_layer_vocab_size)
        ids = b.net.add_select(in_table, tokens, self.int_constant([0])).get_output(0)
        table = b.checkpoint.get(self.name("embed_tokens_per_layer.weight"))
        if tuple(table.shape) != (t.per_layer_vocab_size, layers * width):
            raise ValueError("Gemma 3n per-layer embedding shape does not match its config")
        result = []
        for layer in range(layers):
            # One table per layer keeps every Gather within a modest constant.
            rows = b.constant(np.ascontiguousarray(table[:, layer * width:(layer + 1) * width]))
            token_part = b.scale(b.gather(rows, ids, 0), math.sqrt(width))
            own = b.reshape(b.gather(projected, self.int_constant([layer]), 1), (0, width))
            result.append(b.scale(b.add(own, token_part), 2.0 ** -0.5))
        del table
        return result

    def altup_streams(self, embeds):
        """Stack the embedding with magnitude-matched AltUp projections: [T, N, H]."""
        b, t = self.b, self.t
        reference = b.cast(embeds, trt.float32)
        target = self.magnitude(reference)
        streams = [embeds]
        for index in range(1, t.altup_num_inputs):
            projected = b.cast(self.linear(embeds, f"altup_projections.{index - 1}.weight",
                                           (t.hidden_size, t.hidden_size)), trt.float32)
            scaled = b.div(b.mul(projected, target), self.magnitude(projected, 1e-5))
            streams.append(b.cast(scaled, b.dtype))
        return b.concat([b.reshape(s, (0, 1, t.hidden_size)) for s in streams], 1)

    # --- decoder layer ------------------------------------------------------
    def router(self, value, prefix):
        b, t = self.b, self.t
        routed = b.scale(self.norm(value, f"{prefix}.router_norm"), 1.0 / t.hidden_size)
        routed = self.linear(routed, f"{prefix}.modality_router.weight",
                             (t.altup_num_inputs, t.hidden_size))
        return b.cast(b.activation(b.cast(routed, trt.float32), trt.ActivationType.TANH), b.dtype)

    def attention(self, layer, normed, positions, rope, masks, kv):
        b, t = self.b, self.t
        prefix = f"layers.{layer}.self_attn"
        heads, kv_heads, dim = t.num_heads, t.num_kv_heads, t.head_dim
        kind = t.layer_types[layer]
        cos, sin = rope[kind]
        query = self.linear(normed, f"{prefix}.q_proj.weight", (heads * dim, t.hidden_size))
        query = self.norm(b.reshape(query, (0, heads, dim)), f"{prefix}.q_norm")
        query = self.rope(query, heads, cos, sin)
        owner = self.owners[layer]
        if owner == layer:
            key = self.linear(normed, f"{prefix}.k_proj.weight", (kv_heads * dim, t.hidden_size))
            key = self.norm(b.reshape(key, (0, kv_heads, dim)), f"{prefix}.k_norm")
            key = b.reshape(self.rope(key, kv_heads, cos, sin), (0, kv_heads * dim))
            value = self.linear(normed, f"{prefix}.v_proj.weight", (kv_heads * dim, t.hidden_size))
            value = b.reshape(self.norm(b.reshape(value, (0, kv_heads, dim)), None),
                              (0, kv_heads * dim))
            combined = []
            for label, rows in (("k", key), ("v", value)):
                cache = b.net.add_input(f"cache_{label}_{layer}", b.dtype,
                                        (self.capacity, kv_heads * dim))
                rows.name = f"present_{label}_{layer}"
                b.net.mark_output(rows)
                combined.append(b.concat([cache, rows], 0))
            kv[layer] = combined
        keys, values = kv[owner]
        groups = heads // kv_heads
        # [T, heads, dim] -> [kv_heads, groups, T, dim]; keys -> [kv_heads, 1, S, dim].
        q = b.transpose(b.reshape(query, (0, kv_heads, groups, dim)), (1, 2, 0, 3))
        k = b.transpose(b.reshape(keys, (0, kv_heads, 1, dim)), (1, 2, 0, 3))
        v = b.transpose(b.reshape(values, (0, kv_heads, 1, dim)), (1, 2, 0, 3))
        # Gemma 3n scales attention scores by exactly 1 (q and k are normalized).
        scores = b.matmul(b.cast(q, trt.float32), b.cast(k, trt.float32), transpose_rhs=True)
        scores = b.add(scores, masks[kind])
        softmax = b.net.add_softmax(scores)
        softmax.axes = 1 << 3
        context = b.matmul(b.cast(softmax.get_output(0), b.dtype), v)
        context = b.reshape(b.transpose(context, (2, 0, 1, 3)), (0, heads * dim))
        return self.linear(context, f"{prefix}.o_proj.weight", (t.hidden_size, heads * dim))

    def mlp(self, layer, value):
        b, t = self.b, self.t
        prefix = f"layers.{layer}.mlp"
        width = t.intermediate_sizes[layer]
        gate = self.linear(value, f"{prefix}.gate_proj.weight", (width, t.hidden_size))
        sparsity = t.activation_sparsity[layer]
        if sparsity > 0.0:
            # Gaussian top-k: keep what exceeds mean + std * quantile(sparsity).
            x = b.cast(gate, trt.float32)
            mean = b.mean(x, -1)
            centered = b.sub(x, mean)
            std = b.unary(b.mean(b.mul(centered, centered), -1), trt.UnaryOperation.SQRT)
            multiplier = b.cast(b.scalar(sparsity_multiplier(sparsity), 2, b.dtype), trt.float32)
            cutoff = b.add(mean, b.mul(std, multiplier))
            gate = b.cast(b.activation(b.sub(x, cutoff), trt.ActivationType.RELU), b.dtype)
        up = self.linear(value, f"{prefix}.up_proj.weight", (width, t.hidden_size))
        return self.linear(b.mul(b.gelu_tanh(gate), up), f"{prefix}.down_proj.weight",
                           (t.hidden_size, width))

    def decoder_layer(self, layer, streams, per_layer, positions, rope, masks, kv):
        b, t = self.b, self.t
        prefix = f"layers.{layer}"
        count, hidden = t.altup_num_inputs, t.hidden_size
        # AltUp predict: stream j += sum_k c[j, k] * stream k, c from the router.
        active = b.reshape(b.gather(streams, self.int_constant([0]), 1), (0, hidden))
        coefficients = self.linear(self.router(active, f"{prefix}.altup"),
                                   f"{prefix}.altup.prediction_coefs.weight", (count * count, count))
        predictions = b.add(b.matmul(b.reshape(coefficients, (0, count, count)), streams), streams)
        active = b.reshape(b.gather(predictions, self.int_constant([0]), 1), (0, hidden))

        normed = self.norm(active, f"{prefix}.input_layernorm")
        laurel = self.linear(self.linear(normed, f"{prefix}.laurel.linear_left.weight",
                                         (t.laurel_rank, hidden)),
                             f"{prefix}.laurel.linear_right.weight", (hidden, t.laurel_rank))
        laurel = b.add(normed, self.norm(laurel, f"{prefix}.laurel.post_laurel_norm"))
        attention = self.norm(self.attention(layer, normed, positions, rope, masks, kv),
                              f"{prefix}.post_attention_layernorm")
        mixed = b.divide_fp32(b.add(b.add(active, attention), laurel), math.sqrt(2.0))
        feed_forward = self.norm(self.mlp(layer, self.norm(mixed, f"{prefix}.pre_feedforward_layernorm")),
                                 f"{prefix}.post_feedforward_layernorm")
        activated = b.add(mixed, feed_forward)

        # AltUp correct: every stream moves by its routed share of the innovation.
        correction = self.linear(self.router(activated, f"{prefix}.altup"),
                                 f"{prefix}.altup.correction_coefs.weight", (count, count))
        correction = b.add(correction, b.scalar(1.0, 2))
        innovation = b.reshape(b.sub(activated, active), (0, 1, hidden))
        corrected = b.add(predictions, b.mul(innovation, b.reshape(correction, (0, count, 1))))

        first = b.reshape(b.gather(corrected, self.int_constant([0]), 1), (0, hidden))
        if t.altup_correct_scale:
            scale = b.reshape(b.tensor(self.name(f"{prefix}.altup.correct_output_scale"),
                                       shape=(hidden,)), (1, hidden))
            first = b.mul(first, scale)
        gate = b.gelu_tanh(self.linear(first, f"{prefix}.per_layer_input_gate.weight",
                                       (t.per_layer_hidden, hidden)))
        update = self.linear(b.mul(gate, per_layer[layer]), f"{prefix}.per_layer_projection.weight",
                             (hidden, t.per_layer_hidden))
        update = b.reshape(self.norm(update, f"{prefix}.post_per_layer_input_norm"), (0, 1, hidden))
        # Only streams 1..N-1 receive the per-layer update.
        selector = np.ones((1, count, 1), np.float32)
        selector[0, 0, 0] = 0.0
        return b.add(corrected, b.mul(update, b.constant(selector)))

    # --- head ---------------------------------------------------------------
    def head(self, streams, tokens, table):
        b, t = self.b, self.t
        hidden = t.hidden_size
        last = b.sub(b.cast(b.net.add_shape(tokens).get_output(0), trt.int32), self.int_constant([1]))
        streams = b.gather(streams, last, 0)  # [1, N, H]
        first = b.reshape(b.gather(streams, self.int_constant([0]), 1), (1, hidden))
        reference = b.cast(first, trt.float32)
        target = self.magnitude(reference)
        total = reference
        for index in range(1, t.altup_num_inputs):
            stream = b.reshape(b.gather(streams, self.int_constant([index]), 1), (1, hidden))
            projected = b.cast(self.linear(stream, f"altup_unembed_projections.{index - 1}.weight",
                                           (hidden, hidden)), trt.float32)
            scaled = b.cast(b.div(b.mul(projected, target), self.magnitude(projected, 1e-5)), b.dtype)
            total = b.add(total, b.cast(scaled, trt.float32))
        mean = b.cast(b.scale(total, 1.0 / t.altup_num_inputs), b.dtype)
        final = self.norm(mean, "norm")
        logits = b.cast(b.matmul(final, table, transpose_rhs=True), trt.float32)
        cap = t.final_logit_softcapping
        if cap is not None:
            capped = b.activation(b.divide_fp32(logits, cap), trt.ActivationType.TANH)
            logits = b.scale_fp32(capped, cap)
        logits.name = "logits"
        b.net.mark_output(logits)

    def masks(self, mask, positions):
        """Full and sliding-window additive masks shaped ``[1, 1, T, S]``."""
        b, t = self.b, self.t
        n = b.net
        keys = b.concat([self.int_constant(np.arange(self.capacity)), positions], 0)
        lower = b.sub(b.reshape(positions, (-1, 1)), self.int_constant([[t.sliding_window]]))
        in_window = b.op(b.reshape(keys, (1, -1)), lower, trt.ElementWiseOperation.GREATER)
        blocked = b.constant(np.full((1, 1), -np.inf, np.float32), trt.float32)
        sliding = n.add_select(in_window, mask, blocked).get_output(0)
        return {"full_attention": self.dynamic_reshape(mask, [1, 1], []),
                "sliding_attention": self.dynamic_reshape(sliding, [1, 1], [])}

    def build(self):
        b, t = self.b, self.t
        n = b.net
        tokens = n.add_input("token_id", trt.int32, (-1,))
        positions = n.add_input("position_id", trt.int32, (-1,))
        mask = n.add_input("attention_mask", trt.float32, (-1, -1))
        media = n.add_input("media_features", b.dtype, (-1, t.hidden_size))
        media_mask = n.add_input("media_mask", trt.int32, (-1,))
        embeds, table = self.embed(tokens, media, media_mask)
        per_layer = self.per_layer_inputs(tokens, embeds)
        streams = self.altup_streams(embeds)
        rope = {}
        for kind in sorted(set(t.layer_types)):
            cos, sin = rope_tables(t.rope_theta[kind], t.head_dim, self.capacity)
            rope[kind] = tuple(
                b.reshape(b.cast(b.gather(b.constant(table_values, trt.float32), positions, 0), b.dtype),
                          (0, 1, t.head_dim))
                for table_values in (cos, sin))
        masks = self.masks(mask, positions)
        kv = {}
        for layer in range(t.num_layers):
            streams = self.decoder_layer(layer, streams, per_layer, positions, rope, masks, kv)
        self.head(streams, tokens, table)


__all__ = ["TextGraph", "rope_tables", "sparsity_multiplier"]
