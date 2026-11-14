# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SmolVLM TensorRT graphs built directly with the TensorRT Python API.

Two strongly typed engines:

* ``vision``: a batch of preprocessed tiles ``[N, 3, S, S]`` through the SigLIP-style
  encoder, the pixel-shuffle connector and the modality projection, giving
  ``image_features [N, image_seq_len, text_hidden]``. Every tile is a complete
  ``S x S`` image (the processor splits and resizes to whole tiles), so the patch
  mask is all ones and the bucketed position ids are the identity ``0 .. P^2-1``.
* ``decoder``: the Llama decoder over ``inputs_embeds [L, hidden]`` with host
  RoPE tables, an explicit additive mask ``[1, L, capacity + L]`` and fixed
  per-layer caches ``[kv_heads, capacity, head_dim]`` bound by the runtime. It
  returns the new key/value rows for host-ordered placement and the logits of
  the last input row only.

Norms, attention scores and softmax run in FP32; everything else runs in the
requested precision.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .config import ModelConfig

_TYPES = {"fp32": trt.float32, "fp16": trt.float16, "bf16": trt.bfloat16}


class _Graph:
    def __init__(self, precision: str, verbose: bool):
        if precision not in _TYPES:
            raise ValueError("SmolVLM precision must be fp32, fp16 or bf16")
        self.dtype = _TYPES[precision]
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.network = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        self.config = self.builder.create_builder_config()
        self.config.builder_optimization_level = 3
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        if precision == "fp32":
            # FP32 means FP32 products, not TF32 tensor-core rounding.
            self.config.clear_flag(trt.BuilderFlag.TF32)
        self.profile = self.builder.create_optimization_profile()
        self.constants = []

    def cast(self, value, dtype):
        return value if value.dtype == dtype else self.network.add_cast(value, dtype).get_output(0)

    def constant(self, value, dtype=None):
        dtype = self.dtype if dtype is None else dtype
        storage = {trt.int32: np.int32, trt.int64: np.int64, trt.float16: np.float16}.get(dtype, np.float32)
        array = np.ascontiguousarray(value, dtype=storage)
        self.constants.append(array)
        return self.cast(self.network.add_constant(array.shape, array).get_output(0), dtype)

    def scalar(self, value, rank, dtype=None):
        return self.constant(np.full((1,) * rank, value), dtype)

    def shuffle(self, value, shape=None, permutation=None, first=None):
        layer = self.network.add_shuffle(value)
        if first is not None:
            layer.first_transpose = first
        if shape is not None:
            layer.reshape_dims = shape
        if permutation is not None:
            layer.second_transpose = permutation
        return layer.get_output(0)

    def binary(self, left, right, operation):
        return self.network.add_elementwise(left, right, operation).get_output(0)

    def add(self, left, right):
        return self.binary(left, right, trt.ElementWiseOperation.SUM)

    def mul(self, left, right):
        return self.binary(left, right, trt.ElementWiseOperation.PROD)

    def linear(self, value, weights, prefix, bias=False):
        """``value [rows, in] @ W[out, in]^T (+ b)`` on a 2-D activation."""
        matrix = self.constant(weights[prefix + ".weight"])
        result = self.network.add_matrix_multiply(
            value, trt.MatrixOperation.NONE, matrix, trt.MatrixOperation.TRANSPOSE).get_output(0)
        if bias:
            result = self.add(result, self.constant(weights[prefix + ".bias"][None, :]))
        return result

    def layer_norm(self, value, weights, prefix, eps):
        x = self.cast(value, trt.float32)
        axis = 1 << (len(value.shape) - 1)
        mean = self.network.add_reduce(x, trt.ReduceOperation.AVG, axis, True).get_output(0)
        x = self.binary(x, mean, trt.ElementWiseOperation.SUB)
        variance = self.network.add_reduce(self.mul(x, x), trt.ReduceOperation.AVG, axis, True).get_output(0)
        rank = len(value.shape)
        denominator = self.network.add_unary(
            self.add(variance, self.scalar(eps, rank, trt.float32)), trt.UnaryOperation.SQRT).get_output(0)
        x = self.binary(x, denominator, trt.ElementWiseOperation.DIV)
        shape = (1,) * (rank - 1) + (-1,)
        x = self.mul(x, self.constant(weights[prefix + ".weight"].reshape(shape), trt.float32))
        x = self.add(x, self.constant(weights[prefix + ".bias"].reshape(shape), trt.float32))
        return self.cast(x, self.dtype)

    def rms_norm(self, value, weights, prefix, eps):
        # Llama: normalize in FP32, cast back, then scale by the weight.
        x = self.cast(value, trt.float32)
        mean_square = self.network.add_reduce(self.mul(x, x), trt.ReduceOperation.AVG, 2, True).get_output(0)
        denominator = self.network.add_unary(
            self.add(mean_square, self.scalar(eps, 2, trt.float32)), trt.UnaryOperation.SQRT).get_output(0)
        x = self.binary(x, denominator, trt.ElementWiseOperation.DIV)
        return self.mul(self.cast(x, self.dtype), self.constant(weights[prefix + ".weight"][None, :]))

    def gelu_tanh(self, value):
        x = value
        cube = self.mul(self.mul(x, x), x)
        inner = self.mul(self.add(x, self.mul(cube, self.scalar(0.044715, 2))),
                         self.scalar(np.sqrt(2 / np.pi), 2))
        tanh = self.network.add_activation(inner, trt.ActivationType.TANH).get_output(0)
        return self.mul(self.mul(x, self.scalar(0.5, 2)), self.add(tanh, self.scalar(1, 2)))

    def silu(self, value):
        sigmoid = self.network.add_activation(value, trt.ActivationType.SIGMOID).get_output(0)
        return self.mul(value, sigmoid)

    def attention(self, query, key, value, head_dim, mask=None):
        """``softmax(q k^T / sqrt(d) + mask) v`` with FP32 scores, batched over leading axes."""
        rank = len(query.shape)
        scores = self.network.add_matrix_multiply(
            self.cast(query, trt.float32), trt.MatrixOperation.NONE,
            self.cast(key, trt.float32), trt.MatrixOperation.TRANSPOSE).get_output(0)
        scores = self.mul(scores, self.scalar(head_dim ** -0.5, rank, trt.float32))
        if mask is not None:
            scores = self.add(scores, mask)
        softmax = self.network.add_softmax(scores)
        softmax.axes = 1 << (rank - 1)
        probabilities = self.cast(softmax.get_output(0), self.dtype)
        return self.network.add_matrix_multiply(
            probabilities, trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE).get_output(0)

    def output(self, value, name, dtype=None):
        result = self.cast(value, dtype) if dtype is not None else value
        result.name = name
        self.network.mark_output(result)

    def finish(self, what: str) -> bytes:
        self.config.add_optimization_profile(self.profile)
        plan = self.builder.build_serialized_network(self.network, self.config)
        if plan is None:
            raise RuntimeError(f"SmolVLM {what} TensorRT build failed")
        return bytes(plan)


def max_tiles_per_image(config: ModelConfig) -> int:
    """Tiles of the largest split image plus its global view."""
    grid = -(-config.processing.longest_edge // config.processing.tile_size)
    return grid * grid + 1


def build_vision(config: ModelConfig, weights, max_tiles: int, *, precision="fp32", verbose=False) -> bytes:
    vision = config.vision
    side, patch, width = vision.patches_per_side, vision.patch_size, vision.hidden_size
    heads, head_dim, patches = vision.num_attention_heads, vision.head_dim, side * side
    factor, text_width = config.scale_factor, config.text.hidden_size
    tokens = patches // (factor * factor)
    g = _Graph(precision, verbose)
    n = g.network
    size = vision.image_size
    pixels = n.add_input("pixel_values", g.dtype, (-1, 3, size, size))
    g.profile.set_shape("pixel_values", (1, 3, size, size), (max_tiles, 3, size, size),
                        (max_tiles, 3, size, size))
    # Non-overlapping patch convolution == patchify (channel, row, column order) + matmul.
    x = g.shuffle(pixels, (0, 3, side, patch, side, patch), (0, 2, 4, 1, 3, 5))
    x = g.shuffle(x, (-1, 3 * patch * patch))
    kernel = weights["model.vision_model.embeddings.patch_embedding.weight"].reshape(width, -1)
    x = n.add_matrix_multiply(x, trt.MatrixOperation.NONE, g.constant(kernel),
                              trt.MatrixOperation.TRANSPOSE).get_output(0)
    x = g.add(x, g.constant(weights["model.vision_model.embeddings.patch_embedding.bias"][None, :]))
    positions = weights["model.vision_model.embeddings.position_embedding.weight"]
    if positions.shape != (patches, width):
        raise ValueError("SmolVLM vision position table does not match the patch grid")
    x = g.shuffle(g.add(g.shuffle(x, (-1, patches, width)), g.constant(positions[None])), (-1, width))
    for layer in range(vision.num_hidden_layers):
        prefix = f"model.vision_model.encoder.layers.{layer}"
        normed = g.layer_norm(x, weights, prefix + ".layer_norm1", vision.layer_norm_eps)
        q, k, v = [g.shuffle(g.linear(normed, weights, f"{prefix}.self_attn.{kind}_proj", True),
                             (-1, patches, heads, head_dim), (0, 2, 1, 3)) for kind in ("q", "k", "v")]
        attended = g.shuffle(g.attention(q, k, v, head_dim), (-1, width), None, (0, 2, 1, 3))
        x = g.add(x, g.linear(attended, weights, prefix + ".self_attn.out_proj", True))
        normed = g.layer_norm(x, weights, prefix + ".layer_norm2", vision.layer_norm_eps)
        hidden = g.gelu_tanh(g.linear(normed, weights, prefix + ".mlp.fc1", True))
        x = g.add(x, g.linear(hidden, weights, prefix + ".mlp.fc2", True))
    x = g.layer_norm(x, weights, "model.vision_model.post_layernorm", vision.layer_norm_eps)
    # Pixel shuffle: token (row // f, col // f) gathers its f x f block as (row % f, col % f, channel).
    blocks = side // factor
    x = g.shuffle(x, (-1, blocks, factor, blocks, factor, width), (0, 1, 3, 2, 4, 5))
    x = g.shuffle(x, (-1, factor * factor * width))
    x = g.linear(x, weights, "model.connector.modality_projection.proj")
    g.output(g.shuffle(x, (-1, tokens, text_width)), "image_features")
    return g.finish("vision")


def build_decoder(config: ModelConfig, weights, capacity: int, max_input: int, *,
                  precision="fp32", verbose=False) -> bytes:
    text = config.text
    if not 1 <= max_input <= capacity <= text.max_position_embeddings:
        raise ValueError("SmolVLM decoder input/cache capacity exceeds the checkpoint context")
    g = _Graph(precision, verbose)
    n = g.network
    hidden, d = text.hidden_size, text.head_dim
    heads, kv_heads = text.num_attention_heads, text.num_key_value_heads
    x = n.add_input("inputs_embeds", g.dtype, (-1, hidden))
    cos = n.add_input("rope_cos", trt.float32, (-1, d))
    sin = n.add_input("rope_sin", trt.float32, (-1, d))
    mask = n.add_input("attention_mask", trt.float32, (1, -1, -1))
    opt = min(64, max_input)
    for name, width in (("inputs_embeds", hidden), ("rope_cos", d), ("rope_sin", d)):
        g.profile.set_shape(name, (1, width), (opt, width), (max_input, width))
    g.profile.set_shape("attention_mask", (1, 1, capacity + 1), (1, opt, capacity + opt),
                        (1, max_input, capacity + max_input))
    cos, sin = g.shuffle(cos, (1, -1, d)), g.shuffle(sin, (1, -1, d))
    rotation = g.constant(np.r_[np.arange(d // 2, d), np.arange(d // 2)], trt.int32)
    signs = g.constant(np.r_[-np.ones(d // 2), np.ones(d // 2)].reshape(1, 1, d), trt.float32)
    repeat = g.constant(np.repeat(np.arange(kv_heads), heads // kv_heads), trt.int32)

    def rotary(value):
        # q * cos + rotate_half(q) * sin, rotate_half(q) = cat(-q[d/2:], q[:d/2]).
        value = g.cast(value, trt.float32)
        rotated = g.mul(n.add_gather(value, rotation, 2).get_output(0), signs)
        return g.cast(g.add(g.mul(value, cos), g.mul(rotated, sin)), g.dtype)

    for layer in range(text.num_hidden_layers):
        prefix = f"model.text_model.layers.{layer}"
        normed = g.rms_norm(x, weights, prefix + ".input_layernorm", text.rms_norm_eps)
        q, k, v = [g.shuffle(g.linear(normed, weights, f"{prefix}.self_attn.{kind}_proj"),
                             (-1, count, d), (1, 0, 2))
                   for kind, count in (("q", heads), ("k", kv_heads), ("v", kv_heads))]
        q, k = rotary(q), rotary(k)
        g.output(k, f"present_k_{layer}")
        g.output(v, f"present_v_{layer}")
        cached = []
        for kind, current in (("k", k), ("v", v)):
            past = n.add_input(f"cache_{kind}_{layer}", g.dtype, (kv_heads, capacity, d))
            joined = n.add_concatenation([past, current])
            joined.axis = 1
            cached.append(n.add_gather(joined.get_output(0), repeat, 0).get_output(0))
        attended = g.shuffle(g.attention(q, cached[0], cached[1], d, mask), (-1, heads * d), None, (1, 0, 2))
        x = g.add(x, g.linear(attended, weights, prefix + ".self_attn.o_proj"))
        normed = g.rms_norm(x, weights, prefix + ".post_attention_layernorm", text.rms_norm_eps)
        gate = g.silu(g.linear(normed, weights, prefix + ".mlp.gate_proj"))
        up = g.linear(normed, weights, prefix + ".mlp.up_proj")
        x = g.add(x, g.linear(g.mul(gate, up), weights, prefix + ".mlp.down_proj"))
    # Only the last row's logits are needed: slice [L-1, 0] .. [L-1, hidden).
    rows = n.add_gather(n.add_shape(x).get_output(0), g.constant([0], trt.int64), 0).get_output(0)
    last = g.binary(rows, g.constant([1], trt.int64), trt.ElementWiseOperation.SUB)
    start = n.add_concatenation([last, g.constant([0], trt.int64)])
    start.axis = 0
    tail = n.add_slice(x, (0, 0), (1, hidden), (1, 1))
    tail.set_input(1, start.get_output(0))
    x = g.rms_norm(tail.get_output(0), weights, "model.text_model.norm", text.rms_norm_eps)
    g.output(g.linear(x, weights, "lm_head"), "logits", trt.float32)
    return g.finish("decoder")
