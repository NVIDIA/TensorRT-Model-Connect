# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PaliGemma graphs built directly with the TensorRT Python API.

The vision engine runs the SigLIP tower, the linear projector and the
reference's division by the square root of the Gemma width for one image. The
decoder engine runs the Gemma language model over embeddings the C++ owner
assembles. The owner also supplies the additive attention mask -- the whole
prompt (image tokens, BOS, prefix text and its newline) attends
bidirectionally, every generated token causally -- the 1-based RoPE positions
of the reference, and the placement of the returned K/V rows in its cache.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .config import ModelConfig

_TYPES = {"fp32": trt.float32, "bf16": trt.bfloat16}


class _Graph:
    def __init__(self, precision: str, verbose: bool):
        if precision not in _TYPES:
            raise ValueError("PaliGemma precision must be fp32 or bf16")
        self.dtype = _TYPES[precision]
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.network = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        self.config = self.builder.create_builder_config()
        self.config.builder_optimization_level = 3
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        if precision == "fp32":
            # TensorRT enables TF32 by default; an FP32 build must stay FP32.
            self.config.clear_flag(trt.BuilderFlag.TF32)
        self.profile = self.builder.create_optimization_profile()
        self.constants = []

    def cast(self, value, dtype):
        return value if value.dtype == dtype else self.network.add_cast(value, dtype).get_output(0)

    def constant(self, value, dtype=None):
        dtype = self.dtype if dtype is None else dtype
        storage = np.int32 if dtype == trt.int32 else np.float32
        array = np.ascontiguousarray(value, dtype=storage)
        self.constants.append(array)
        return self.cast(self.network.add_constant(array.shape, array).get_output(0), dtype)

    def scalar(self, value, rank, dtype=None):
        return self.constant(np.full((1,) * rank, value), dtype)

    def reshape(self, value, shape, permutation=None):
        layer = self.network.add_shuffle(value)
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

    def matmul(self, left, right, transpose_right=False):
        operation = trt.MatrixOperation.TRANSPOSE if transpose_right else trt.MatrixOperation.NONE
        return self.network.add_matrix_multiply(left, trt.MatrixOperation.NONE,
                                                right, operation).get_output(0)

    def linear(self, value, weights, prefix, bias=False):
        result = self.matmul(value, self.constant(weights[prefix + ".weight"]), True)
        if bias:
            result = self.add(result, self.constant(weights[prefix + ".bias"][None, :]))
        return result

    def mean(self, value):
        return self.network.add_reduce(value, trt.ReduceOperation.AVG, 1 << 1, True).get_output(0)

    def layer_norm(self, value, weights, prefix, eps):
        """SigLIP LayerNorm, computed in FP32 like the reference kernels."""
        x = self.cast(value, trt.float32)
        x = self.binary(x, self.mean(x), trt.ElementWiseOperation.SUB)
        x = self.binary(x, self._root(self.mean(self.mul(x, x)), eps), trt.ElementWiseOperation.DIV)
        x = self.mul(x, self.constant(weights[prefix + ".weight"][None, :], trt.float32))
        return self.cast(self.add(x, self.constant(weights[prefix + ".bias"][None, :], trt.float32)),
                         self.dtype)

    def rms_norm(self, value, weights, prefix, eps):
        """Gemma RMSNorm: normalize and scale by (1 + weight) in FP32, then cast."""
        x = self.cast(value, trt.float32)
        x = self.binary(x, self._root(self.mean(self.mul(x, x)), eps), trt.ElementWiseOperation.DIV)
        scale = 1.0 + weights[prefix + ".weight"].astype(np.float32)
        return self.cast(self.mul(x, self.constant(scale[None, :], trt.float32)), self.dtype)

    def _root(self, value, eps):
        return self.network.add_unary(self.add(value, self.scalar(eps, 2, trt.float32)),
                                      trt.UnaryOperation.SQRT).get_output(0)

    def gelu_tanh(self, value):
        """GELU with the tanh approximation (``gelu_pytorch_tanh``), evaluated in FP32."""
        x = self.cast(value, trt.float32)
        cube = self.mul(self.mul(x, x), x)
        inner = self.mul(self.add(x, self.mul(cube, self.scalar(0.044715, 2, trt.float32))),
                         self.scalar(np.sqrt(2 / np.pi), 2, trt.float32))
        tanh = self.network.add_activation(inner, trt.ActivationType.TANH).get_output(0)
        half = self.mul(x, self.scalar(0.5, 2, trt.float32))
        return self.cast(self.mul(half, self.add(tanh, self.scalar(1.0, 2, trt.float32))), self.dtype)

    def attention(self, query, key, value, head_dim, mask=None):
        scores = self.matmul(self.cast(query, trt.float32), self.cast(key, trt.float32), True)
        scores = self.mul(scores, self.scalar(head_dim ** -0.5, 3, trt.float32))
        if mask is not None:
            scores = self.add(scores, mask)
        softmax = self.network.add_softmax(scores)
        softmax.axes = 1 << 2
        return self.matmul(self.cast(softmax.get_output(0), self.dtype), value)

    def merge_heads(self, value, width):
        layer = self.network.add_shuffle(value)
        layer.first_transpose = (1, 0, 2)
        layer.reshape_dims = (-1, width)
        return layer.get_output(0)

    def output(self, value, name, dtype=None):
        result = self.cast(value, dtype) if dtype is not None else value
        result.name = name
        self.network.mark_output(result)

    def finish(self, what: str) -> bytes:
        self.config.add_optimization_profile(self.profile)
        plan = self.builder.build_serialized_network(self.network, self.config)
        if plan is None:
            raise RuntimeError(f"PaliGemma {what} TensorRT build failed")
        return bytes(plan)


def build_vision(config: ModelConfig, weights, *, precision: str, verbose: bool = False) -> bytes:
    """SigLIP tower and projector for one normalized CHW image -> [patches, Gemma width]."""
    vision, width = config.vision, config.vision.hidden_size
    grid, patch = vision.grid, vision.patch_size
    g = _Graph(precision, verbose)
    n = g.network
    pixels = n.add_input("pixel_values", g.dtype, (1, 3, vision.image_size, vision.image_size))
    # Non-overlapping patches flattened as (channel, row, column): the stride-P
    # convolution becomes one matrix product with the reshaped kernel.
    patches = g.reshape(g.reshape(pixels, (3, grid, patch, grid, patch), (1, 3, 0, 2, 4)),
                        (vision.num_patches, 3 * patch * patch))
    kernel = weights["vision.embeddings.patch_embedding.weight"].reshape(width, -1)
    x = g.add(g.matmul(patches, g.constant(kernel), True),
              g.constant(weights["vision.embeddings.patch_embedding.bias"][None, :]))
    x = g.add(x, g.constant(weights["vision.embeddings.position_embedding.weight"]))
    heads, head_dim = vision.num_attention_heads, vision.head_dim
    for layer in range(vision.num_hidden_layers):
        prefix = f"vision.encoder.layers.{layer}"
        norm = g.layer_norm(x, weights, prefix + ".layer_norm1", vision.layer_norm_eps)
        q, k, v = [g.reshape(g.linear(norm, weights, f"{prefix}.self_attn.{kind}_proj", True),
                             (-1, heads, head_dim), (1, 0, 2)) for kind in ("q", "k", "v")]
        attended = g.merge_heads(g.attention(q, k, v, head_dim), width)
        x = g.add(x, g.linear(attended, weights, prefix + ".self_attn.out_proj", True))
        norm = g.layer_norm(x, weights, prefix + ".layer_norm2", vision.layer_norm_eps)
        hidden = g.gelu_tanh(g.linear(norm, weights, prefix + ".mlp.fc1", True))
        x = g.add(x, g.linear(hidden, weights, prefix + ".mlp.fc2", True))
    x = g.layer_norm(x, weights, "vision.post_layernorm", vision.layer_norm_eps)
    x = g.linear(x, weights, "projector.linear", True)
    # The reference divides by a Python float, i.e. in FP32 arithmetic for every dtype.
    root = g.scalar(np.float32(config.text.hidden_size) ** np.float32(0.5), 2, trt.float32)
    x = g.binary(g.cast(x, trt.float32), root, trt.ElementWiseOperation.DIV)
    g.output(g.cast(x, g.dtype), "image_features")
    return g.finish("vision")


def build_decoder(config: ModelConfig, weights, max_sequence_length: int, *, precision: str,
                  verbose: bool = False) -> bytes:
    """Gemma prefill/decode over a fixed cache; return the new K/V rows and last-row logits.

    ``attention_mask`` is [1, rows, cache + rows]; cache slots that hold no
    token yet are masked by the caller, so an empty-cache prefill needs no
    zero-sized tensors.
    """
    text = config.text
    if not 1 <= max_sequence_length <= text.max_position_embeddings:
        raise ValueError("PaliGemma sequence capacity exceeds the Gemma context")
    g = _Graph(precision, verbose)
    n, width, d = g.network, text.hidden_size, text.head_dim
    capacity = max_sequence_length
    x = n.add_input("inputs_embeds", g.dtype, (-1, width))
    cos = n.add_input("rope_cos", trt.float32, (-1, d))
    sin = n.add_input("rope_sin", trt.float32, (-1, d))
    mask = n.add_input("attention_mask", trt.float32, (1, -1, -1))
    opt = min(config.num_image_tokens + 16, capacity)
    for name, columns in (("inputs_embeds", width), ("rope_cos", d), ("rope_sin", d)):
        g.profile.set_shape(name, (1, columns), (opt, columns), (capacity, columns))
    g.profile.set_shape("attention_mask", (1, 1, capacity + 1), (1, opt, capacity + opt),
                        (1, capacity, 2 * capacity))
    # Gemma scales every input row, image rows included, by sqrt(width) in the
    # embedding dtype.
    x = g.mul(x, g.scalar(np.float32(width) ** np.float32(0.5), 2))
    cos, sin = g.reshape(cos, (1, -1, d)), g.reshape(sin, (1, -1, d))
    rotate = g.constant(np.r_[np.arange(d // 2, d), np.arange(d // 2)], trt.int32)
    signs = g.constant(np.r_[-np.ones(d // 2), np.ones(d // 2)].reshape(1, 1, d), trt.float32)
    groups = text.num_attention_heads // text.num_key_value_heads
    repeat = g.constant(np.repeat(np.arange(text.num_key_value_heads), groups), trt.int32)

    def rotary(value):
        value = g.cast(value, trt.float32)
        rotated = g.mul(n.add_gather(value, rotate, 2).get_output(0), signs)
        return g.cast(g.add(g.mul(value, cos), g.mul(rotated, sin)), g.dtype)

    for layer in range(text.num_hidden_layers):
        prefix = f"text.layers.{layer}"
        norm = g.rms_norm(x, weights, prefix + ".input_layernorm", text.rms_norm_eps)
        q, k, v = [g.reshape(g.linear(norm, weights, f"{prefix}.self_attn.{kind}_proj"),
                             (-1, heads, d), (1, 0, 2))
                   for kind, heads in (("q", text.num_attention_heads),
                                       ("k", text.num_key_value_heads),
                                       ("v", text.num_key_value_heads))]
        q, k = rotary(q), rotary(k)
        g.output(k, f"present_k_{layer}")
        g.output(v, f"present_v_{layer}")
        cached = []
        for kind, current in (("k", k), ("v", v)):
            past = n.add_input(f"cache_{kind}_{layer}", g.dtype,
                               (text.num_key_value_heads, capacity, d))
            joined = n.add_concatenation([past, current])
            joined.axis = 1
            cached.append(n.add_gather(joined.get_output(0), repeat, 0).get_output(0))
        attended = g.merge_heads(g.attention(q, cached[0], cached[1], d, mask),
                                 text.num_attention_heads * d)
        x = g.add(x, g.linear(attended, weights, prefix + ".self_attn.o_proj"))
        norm = g.rms_norm(x, weights, prefix + ".post_attention_layernorm", text.rms_norm_eps)
        gate = g.gelu_tanh(g.linear(norm, weights, prefix + ".mlp.gate_proj"))
        up = g.linear(norm, weights, prefix + ".mlp.up_proj")
        x = g.add(x, g.linear(g.mul(gate, up), weights, prefix + ".mlp.down_proj"))
    x = g.rms_norm(x, weights, "text.norm", text.rms_norm_eps)
    rows = n.add_shape(x).get_output(0)
    first = n.add_slice(rows, (0,), (1,), (1,)).get_output(0)
    one = n.add_constant((1,), np.ones(1, np.int64)).get_output(0)
    last = g.cast(g.binary(first, one, trt.ElementWiseOperation.SUB), trt.int32)
    x = n.add_gather(x, last, 0).get_output(0)
    # The language-model head is tied to the token embedding.
    g.output(g.linear(x, {"head.weight": weights["text.embed_tokens.weight"]}, "head"),
             "logits", trt.float32)
    return g.finish("decoder")
