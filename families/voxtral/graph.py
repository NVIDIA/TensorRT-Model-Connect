# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Voxtral TensorRT graphs: the audio encoder with its projector, and the text decoder.

Both networks are strongly typed. Weights are created in the engine precision;
normalization statistics, softmax, rotary products, GELU, the encoder position
sum and the SwiGLU product are computed in float32 and cast back explicitly.
"""

from __future__ import annotations

import math
import os

import numpy as np
import tensorrt as trt

from .checkpoint import MlxCheckpoint, engine_array
from .config import VoxtralConfig

_TRT_DTYPES = {"fp32": trt.float32, "fp16": trt.float16, "bf16": trt.bfloat16}
_FP16_CLAMP = float(np.finfo(np.float16).max) - 1000.0


class _Graph:
    def __init__(self, network: trt.INetworkDefinition, precision: str):
        self.network = network
        self.precision = precision
        self.dtype = _TRT_DTYPES[precision]
        self.keep_alive: list[np.ndarray] = []

    # -- constants and elementary layers -----------------------------------------------------
    def weights(self, values: np.ndarray) -> trt.Weights:
        array = engine_array(values, self.precision)
        self.keep_alive.append(array)
        if self.precision == "bf16":
            return trt.Weights(trt.bfloat16, array.ctypes.data, array.size)
        return trt.Weights(array)

    def const(self, values, *, float32: bool = False) -> trt.ITensor:
        values = np.asarray(values, dtype=np.float32)
        if float32:
            array = np.ascontiguousarray(values)
            self.keep_alive.append(array)
            weights = trt.Weights(array)
        else:
            weights = self.weights(values)
        return self.network.add_constant(values.shape, weights).get_output(0)

    def int_const(self, values) -> trt.ITensor:
        array = np.ascontiguousarray(values, dtype=np.int32)
        self.keep_alive.append(array)
        return self.network.add_constant(array.shape, trt.Weights(array)).get_output(0)

    def cast(self, x: trt.ITensor, dtype) -> trt.ITensor:
        return x if x.dtype == dtype else self.network.add_cast(x, dtype).get_output(0)

    def ew(self, a: trt.ITensor, b: trt.ITensor, op) -> trt.ITensor:
        return self.network.add_elementwise(a, b, op).get_output(0)

    def add(self, a, b):
        return self.ew(a, b, trt.ElementWiseOperation.SUM)

    def mul(self, a, b):
        return self.ew(a, b, trt.ElementWiseOperation.PROD)

    def scalar(self, value: float, like: trt.ITensor) -> trt.ITensor:
        shape = (1,) * len(like.shape)
        if like.dtype == trt.float32:
            return self.const(np.full(shape, value), float32=True)
        return self.cast(self.const(np.full(shape, value), float32=True), like.dtype)

    def shuffle(self, x, reshape=None, *, first=None, second=None) -> trt.ITensor:
        layer = self.network.add_shuffle(x)
        if first is not None:
            layer.first_transpose = first
        if reshape is not None:
            layer.reshape_dims = reshape
        if second is not None:
            layer.second_transpose = second
        return layer.get_output(0)

    def matmul(self, a, b, *, transpose_b: bool = False) -> trt.ITensor:
        op_b = trt.MatrixOperation.TRANSPOSE if transpose_b else trt.MatrixOperation.NONE
        return self.network.add_matrix_multiply(a, trt.MatrixOperation.NONE, b, op_b).get_output(0)

    def linear(self, x, weight: np.ndarray, bias: np.ndarray | None = None) -> trt.ITensor:
        """``x @ weight.T (+ bias)`` for a 2-D activation and an ``[out, in]`` weight."""
        y = self.matmul(x, self.const(weight), transpose_b=True)
        if bias is not None:
            y = self.add(y, self.const(bias.reshape(1, -1)))
        return y

    def activation(self, x, kind) -> trt.ITensor:
        return self.network.add_activation(x, kind).get_output(0)

    def gelu(self, x) -> trt.ITensor:
        """Exact (erf) GELU evaluated in float32 and rounded once, as the reference does;
        a reduced-precision GELU loses accuracy in the fp16 encoder."""
        return self.cast(self.activation(self.cast(x, trt.float32), trt.ActivationType.GELU_ERF),
                         x.dtype)

    def layer_norm(self, x, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
        """LayerNorm over the last axis of a 2-D tensor, statistics in float32."""
        value = self.cast(x, trt.float32)
        scale = self.const(gamma.reshape(1, -1), float32=True)
        shift = self.const(beta.reshape(1, -1), float32=True)
        layer = self.network.add_normalization(value, scale, shift, 1 << 1)
        layer.epsilon = eps
        return self.cast(layer.get_output(0), self.dtype)

    def rms_norm(self, x, weight: np.ndarray, eps: float):
        """Llama RMSNorm: normalize in float32, round to the engine dtype, then apply the gain."""
        value = self.cast(x, trt.float32)
        square = self.mul(value, value)
        mean = self.network.add_reduce(square, trt.ReduceOperation.AVG, 1 << 1, True).get_output(0)
        root = self.network.add_unary(self.add(mean, self.scalar(eps, mean)),
                                      trt.UnaryOperation.SQRT).get_output(0)
        inverse = self.network.add_unary(root, trt.UnaryOperation.RECIP).get_output(0)
        normalized = self.cast(self.mul(value, inverse), self.dtype)
        return self.mul(normalized, self.const(weight.reshape(1, -1)))

    def attention(self, q, k, v, scale: float, mask=None) -> trt.ITensor:
        """``softmax(q k^T * scale + mask) v`` over the last two axes.

        Scores and probabilities are float32; the probabilities are rounded to the
        engine dtype before the value product, as the reference does.
        """
        scores = self.matmul(self.cast(q, trt.float32), self.cast(k, trt.float32), transpose_b=True)
        if scale != 1.0:
            scores = self.mul(scores, self.scalar(scale, scores))
        if mask is not None:
            scores = self.add(scores, mask)
        softmax = self.network.add_softmax(scores)
        softmax.axes = 1 << (len(scores.shape) - 1)
        probabilities = self.cast(softmax.get_output(0), self.dtype)
        return self.matmul(probabilities, v)


# -- audio encoder ------------------------------------------------------------------------------
class _EncoderGraph(_Graph):
    def __init__(self, network, precision, config: VoxtralConfig, weights: MlxCheckpoint):
        super().__init__(network, precision)
        self.config = config
        self.weights_source = weights

    def conv(self, x, module: str, in_channels: int, stride: int) -> trt.ITensor:
        width = self.config.audio.hidden_size
        kernel = self.weights_source.conv1d(module, width, in_channels, 3)
        bias = self.weights_source.vector(f"{module}.bias", width)
        layer = self.network.add_convolution_nd(
            x, width, (1, 3), self.weights(kernel.reshape(width, in_channels, 1, 3)),
            self.weights(bias))
        layer.stride_nd = (1, stride)
        layer.padding_nd = (0, 1)
        return self.gelu(layer.get_output(0))

    def encoder_layer(self, h, index: int):
        audio = self.config.audio
        source = self.weights_source
        prefix = f"audio_tower.layers.{index}"
        width, heads, head_dim = audio.hidden_size, audio.num_attention_heads, audio.head_dim
        frames = audio.max_source_positions

        normed = self.layer_norm(h, source.vector(f"{prefix}.self_attn_layer_norm.weight", width),
                                 source.vector(f"{prefix}.self_attn_layer_norm.bias", width))
        q = self.linear(normed, source.linear(f"{prefix}.self_attn.q_proj", width, width),
                        source.vector(f"{prefix}.self_attn.q_proj.bias", width))
        # The reference scales q before the product; head_dim ** -0.5 is a power of two here.
        q = self.mul(q, self.scalar(head_dim ** -0.5, q))
        k = self.linear(normed, source.linear(f"{prefix}.self_attn.k_proj", width, width))
        v = self.linear(normed, source.linear(f"{prefix}.self_attn.v_proj", width, width),
                        source.vector(f"{prefix}.self_attn.v_proj.bias", width))

        def heads_first(x):
            return self.shuffle(x, (-1, frames, heads, head_dim), second=(0, 2, 1, 3))

        context = self.attention(heads_first(q), heads_first(k), heads_first(v), 1.0)
        context = self.shuffle(context, (-1, width), first=(0, 2, 1, 3))
        attended = self.linear(context, source.linear(f"{prefix}.self_attn.out_proj", width, width),
                               source.vector(f"{prefix}.self_attn.out_proj.bias", width))
        h = self.add(h, attended)

        normed = self.layer_norm(h, source.vector(f"{prefix}.final_layer_norm.weight", width),
                                 source.vector(f"{prefix}.final_layer_norm.bias", width))
        inner = audio.intermediate_size
        hidden = self.linear(normed, source.linear(f"{prefix}.fc1", inner, width),
                             source.vector(f"{prefix}.fc1.bias", inner))
        hidden = self.gelu(hidden)
        hidden = self.linear(hidden, source.linear(f"{prefix}.fc2", width, inner),
                             source.vector(f"{prefix}.fc2.bias", width))
        h = self.add(h, hidden)
        if self.precision == "fp16":
            # The reference clamps fp16 encoder activations after every layer.
            layer = self.network.add_activation(h, trt.ActivationType.CLIP)
            layer.alpha, layer.beta = -_FP16_CLAMP, _FP16_CLAMP
            h = layer.get_output(0)
        return h

    def populate(self) -> None:
        audio = self.config.audio
        text = self.config.text
        source = self.weights_source
        frames = audio.max_source_positions
        features = self.network.add_input(
            "input_features", trt.float32, (-1, audio.num_mel_bins, self.config.mel_frames_per_chunk))
        x = self.shuffle(self.cast(features, self.dtype),
                         (0, audio.num_mel_bins, 1, self.config.mel_frames_per_chunk))
        x = self.conv(x, "audio_tower.conv1", audio.num_mel_bins, 1)
        x = self.conv(x, "audio_tower.conv2", audio.hidden_size, 2)
        x = self.shuffle(x, (0, audio.hidden_size, frames), second=(0, 2, 1))
        # Positions are float32 in the reference; the sum is rounded once.
        positions = source.matrix("audio_tower.embed_positions.weight", (frames, audio.hidden_size))
        x = self.add(self.cast(x, trt.float32),
                     self.const(positions.reshape(1, frames, audio.hidden_size), float32=True))
        h = self.shuffle(self.cast(x, self.dtype), (-1, audio.hidden_size))
        for index in range(audio.num_hidden_layers):
            h = self.encoder_layer(h, index)
        h = self.layer_norm(h, source.vector("audio_tower.layer_norm.weight", audio.hidden_size),
                            source.vector("audio_tower.layer_norm.bias", audio.hidden_size))
        stacked = self.shuffle(h, (-1, audio.intermediate_size))
        projected = self.linear(
            stacked, source.linear("multi_modal_projector.linear_1", text.hidden_size,
                                   audio.intermediate_size))
        projected = self.gelu(projected)
        projected = self.linear(
            projected, source.linear("multi_modal_projector.linear_2", text.hidden_size,
                                     text.hidden_size))
        projected.name = "audio_embeds"
        self.network.mark_output(projected)


# -- text decoder -------------------------------------------------------------------------------
def rotary_tables(capacity: int, head_dim: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Correctly rounded float32 cos/sin tables ``[capacity, head_dim]`` (halves repeated)."""
    inverse = 1.0 / theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    angles = np.arange(capacity, dtype=np.float64)[:, None] * inverse[None, :]
    angles = np.concatenate((angles, angles), axis=-1)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


class _DecoderGraph(_Graph):
    def __init__(self, network, precision, config: VoxtralConfig, weights: MlxCheckpoint,
                 capacity: int):
        super().__init__(network, precision)
        self.config = config
        self.weights_source = weights
        self.capacity = capacity

    def rotary(self, x, cos, sin, heads: int) -> trt.ITensor:
        """HF Llama rotary embedding (rotate_half) in float32, returned in the engine dtype."""
        d = self.config.text.head_dim
        value = self.shuffle(self.cast(x, trt.float32), (-1, heads, d))
        swap = np.concatenate((np.arange(d // 2, d), np.arange(d // 2)))
        swapped = self.network.add_gather(value, self.int_const(swap), 2).get_output(0)
        sign = np.concatenate((-np.ones(d // 2), np.ones(d // 2))).reshape(1, 1, d)
        rotated = self.mul(swapped, self.const(sign, float32=True))
        result = self.add(self.mul(value, cos), self.mul(rotated, sin))
        return self.cast(self.shuffle(result, (-1, heads * d)), self.dtype)

    def last_row(self, x) -> trt.ITensor:
        width = int(x.shape[1])
        one_row = self.network.add_constant((2,), trt.Weights(self._keep(np.array([1, width],
                                                                                  np.int64))))
        shape = self.network.add_shape(x).get_output(0)
        start = self.ew(shape, one_row.get_output(0), trt.ElementWiseOperation.SUB)
        layer = self.network.add_slice(x, (0, 0), (1, width), (1, 1))
        layer.set_input(1, start)
        return layer.get_output(0)

    def _keep(self, array: np.ndarray) -> np.ndarray:
        self.keep_alive.append(array)
        return array

    def decoder_layer(self, x, index: int, position, mask, cos, sin):
        text = self.config.text
        source = self.weights_source
        prefix = f"language_model.layers.{index}"
        hidden, heads, kv_heads, d = (text.hidden_size, text.num_attention_heads,
                                      text.num_key_value_heads, text.head_dim)
        groups = heads // kv_heads

        normed = self.rms_norm(x, source.vector(f"{prefix}.input_layernorm.weight", hidden),
                               text.rms_norm_eps)
        q = self.linear(normed, source.linear(f"{prefix}.self_attn.q_proj", heads * d, hidden))
        k = self.linear(normed, source.linear(f"{prefix}.self_attn.k_proj", kv_heads * d, hidden))
        v = self.linear(normed, source.linear(f"{prefix}.self_attn.v_proj", kv_heads * d, hidden))
        q = self.rotary(q, cos, sin, heads)
        k = self.rotary(k, cos, sin, kv_heads)

        caches = []
        for stem, current in (("k", k), ("v", v)):
            current.name = f"present_{stem}_{index}"
            self.network.mark_output(current)
            cache = self.network.add_input(f"cache_{stem}_{index}", self.dtype,
                                           (self.capacity, kv_heads * d))
            updated = self.network.add_scatter(cache, self.shuffle(position, (-1, 1)), current,
                                               trt.ScatterMode.ND).get_output(0)
            caches.append(self.shuffle(updated, (self.capacity, kv_heads, d), second=(1, 0, 2)))

        # Group the query heads that share a key/value head: [kv, groups * rows, d].
        grouped = self.shuffle(q, (-1, kv_heads, groups, d), second=(1, 2, 0, 3))
        grouped = self.shuffle(grouped, (kv_heads, -1, d))
        rows_mask = self.shuffle(mask, (1, 1, -1, self.capacity))
        scores_shape = (kv_heads, groups, -1, self.capacity)
        scores = self.matmul(self.cast(grouped, trt.float32), self.cast(caches[0], trt.float32),
                             transpose_b=True)
        scores = self.mul(scores, self.scalar(1.0 / math.sqrt(d), scores))
        scores = self.add(self.shuffle(scores, scores_shape), rows_mask)
        softmax = self.network.add_softmax(scores)
        softmax.axes = 1 << 3
        probabilities = self.cast(softmax.get_output(0), self.dtype)
        probabilities = self.shuffle(probabilities, (kv_heads, -1, self.capacity))
        context = self.matmul(probabilities, caches[1])
        context = self.shuffle(context, (kv_heads, groups, -1, d), second=(2, 0, 1, 3))
        context = self.shuffle(context, (-1, heads * d))
        attended = self.linear(context, source.linear(f"{prefix}.self_attn.o_proj", hidden,
                                                      heads * d))
        x = self.add(x, attended)

        normed = self.rms_norm(x, source.vector(f"{prefix}.post_attention_layernorm.weight", hidden),
                               text.rms_norm_eps)
        inner = text.intermediate_size
        gate = self.cast(self.linear(normed, source.linear(f"{prefix}.mlp.gate_proj", inner, hidden)),
                         trt.float32)
        up = self.cast(self.linear(normed, source.linear(f"{prefix}.mlp.up_proj", inner, hidden)),
                       trt.float32)
        sigmoid = self.activation(gate, trt.ActivationType.SIGMOID)
        product = self.cast(self.mul(self.mul(gate, sigmoid), up), self.dtype)
        down = self.linear(product, source.linear(f"{prefix}.mlp.down_proj", hidden, inner))
        return self.add(x, down)

    def populate(self) -> None:
        text = self.config.text
        source = self.weights_source
        embeds = self.network.add_input("inputs_embeds", self.dtype, (-1, text.hidden_size))
        position = self.network.add_input("position_id", trt.int32, (-1,))
        mask = self.network.add_input("attention_mask", trt.float32, (-1, self.capacity))
        cos_table, sin_table = rotary_tables(self.capacity, text.head_dim, text.rope_theta)
        cos = self.network.add_gather(self.const(cos_table, float32=True), position, 0).get_output(0)
        sin = self.network.add_gather(self.const(sin_table, float32=True), position, 0).get_output(0)
        cos = self.shuffle(cos, (-1, 1, text.head_dim))
        sin = self.shuffle(sin, (-1, 1, text.head_dim))
        x = embeds
        for index in range(text.num_hidden_layers):
            x = self.decoder_layer(x, index, position, mask, cos, sin)
        x = self.last_row(x)
        x = self.rms_norm(x, source.vector("language_model.norm.weight", text.hidden_size),
                          text.rms_norm_eps)
        logits = self.linear(x, source.linear("lm_head", text.vocab_size, text.hidden_size))
        logits = self.cast(logits, trt.float32)
        logits.name = "logits"
        self.network.mark_output(logits)


# -- engine builds ------------------------------------------------------------------------------
def _builder(verbose: bool):
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    builder.max_threads = min(16, os.cpu_count() or 1)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    return builder, network


def _settings(builder, precision: str):
    settings = builder.create_builder_config()
    settings.builder_optimization_level = 3
    if precision == "fp32":
        # TF32 would round the float32 products the reference keeps exact.
        settings.clear_flag(trt.BuilderFlag.TF32)
    return settings


def _serialize(builder, network, settings, what: str) -> bytes:
    plan = builder.build_serialized_network(network, settings)
    if plan is None:
        raise RuntimeError(f"TensorRT failed to build the Voxtral {what}; see builder diagnostics")
    return bytes(plan)


def build_encoder(config: VoxtralConfig, weights: MlxCheckpoint, precision: str,
                  max_chunks: int, *, verbose: bool = False) -> bytes:
    """Audio tower plus projector: ``[chunks, mels, frames]`` -> ``[chunks * tokens, hidden]``."""
    builder, network = _builder(verbose)
    graph = _EncoderGraph(network, precision, config, weights)
    graph.populate()
    settings = _settings(builder, precision)
    profile = builder.create_optimization_profile()
    shape = (config.audio.num_mel_bins, config.mel_frames_per_chunk)
    profile.set_shape("input_features", (1, *shape), (1, *shape), (max_chunks, *shape))
    settings.add_optimization_profile(profile)
    return _serialize(builder, network, settings, "audio encoder")


def build_decoder(config: VoxtralConfig, weights: MlxCheckpoint, precision: str, capacity: int,
                  prefill_rows: int, *, verbose: bool = False) -> bytes:
    """Decoder with a multi-row prefill profile (0) and a one-row decode profile (1)."""
    builder, network = _builder(verbose)
    graph = _DecoderGraph(network, precision, config, weights, capacity)
    graph.populate()
    settings = _settings(builder, precision)
    hidden = config.text.hidden_size
    for low, opt, high in ((1, prefill_rows, prefill_rows), (1, 1, 1)):
        profile = builder.create_optimization_profile()
        profile.set_shape("inputs_embeds", (low, hidden), (opt, hidden), (high, hidden))
        profile.set_shape("position_id", (low,), (opt,), (high,))
        profile.set_shape("attention_mask", (low, capacity), (opt, capacity), (high, capacity))
        settings.add_optimization_profile(profile)
    return _serialize(builder, network, settings, "text decoder")
