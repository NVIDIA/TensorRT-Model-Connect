# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3 reasoner graphs built directly with the TensorRT Python API.

No Edge plugins, ONNX, or device-specific graph branches are used. The decoder
accepts fused text/visual embeddings and explicit MRoPE cosines/sines. The C++
owner controls cache placement, multimodal positions, and request lifecycle.
The visual engine processes one frame per invocation, preserving frame-local
bidirectional attention for both images and video.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .config import ModelConfig


class _Graph:
    def __init__(self, precision: str, verbose: bool):
        types = {"fp32": trt.float32, "fp16": trt.float16, "bf16": trt.bfloat16}
        if precision not in types:
            raise ValueError("Cosmos3-Edge precision must be fp32, fp16, or bf16")
        self.dtype = types[precision]
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        self.config = self.builder.create_builder_config()
        self.config.builder_optimization_level = 3
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        if precision == "fp32":
            self.config.clear_flag(trt.BuilderFlag.TF32)
        self.profile = self.builder.create_optimization_profile()
        self.constants = []

    def cast(self, value, dtype):
        return value if value.dtype == dtype else self.network.add_cast(value, dtype).get_output(0)

    def constant(self, value, dtype=None):
        dtype = self.dtype if dtype is None else dtype
        storage = np.int32 if dtype == trt.int32 else np.float16 if dtype == trt.float16 else np.float32
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

    def linear(self, value, weights, prefix, bias=False):
        matrix = self.constant(weights[prefix + ".weight"])
        result = self.network.add_matrix_multiply(value, trt.MatrixOperation.NONE,
                                                  matrix, trt.MatrixOperation.TRANSPOSE).get_output(0)
        if bias:
            result = self.add(result, self.constant(weights[prefix + ".bias"][None, :]))
        return result

    def norm(self, value, weights, prefix, eps, centered=False):
        x = self.cast(value, trt.float32)
        if centered:
            mean = self.network.add_reduce(x, trt.ReduceOperation.AVG, 2, True).get_output(0)
            x = self.binary(x, mean, trt.ElementWiseOperation.SUB)
        mean_square = self.network.add_reduce(self.mul(x, x), trt.ReduceOperation.AVG, 2, True).get_output(0)
        denominator = self.network.add_unary(self.add(mean_square, self.scalar(eps, 2, trt.float32)),
                                             trt.UnaryOperation.SQRT).get_output(0)
        x = self.binary(x, denominator, trt.ElementWiseOperation.DIV)
        if centered:
            x = self.mul(x, self.constant(weights[prefix + ".weight"][None, :], trt.float32))
            x = self.add(x, self.constant(weights[prefix + ".bias"][None, :], trt.float32))
            return self.cast(x, self.dtype)
        return self.mul(self.cast(x, self.dtype), self.constant(weights[prefix + ".weight"][None, :]))

    def gelu(self, value, approximate=False):
        x = value
        if approximate:
            cube = self.mul(self.mul(x, x), x)
            inner = self.mul(self.add(x, self.mul(cube, self.scalar(0.044715, 2))),
                             self.scalar(np.sqrt(2 / np.pi), 2))
            sigmoid = self.network.add_activation(inner, trt.ActivationType.TANH).get_output(0)
        else:
            scaled = self.mul(x, self.scalar(1 / np.sqrt(2), 2))
            sigmoid = self.network.add_unary(scaled, trt.UnaryOperation.ERF).get_output(0)
        return self.mul(self.mul(x, self.scalar(0.5, 2)), self.add(sigmoid, self.scalar(1, 2)))

    def attention(self, query, key, value, head_dim, mask=None):
        scores = self.network.add_matrix_multiply(self.cast(query, trt.float32), trt.MatrixOperation.NONE,
                                                  self.cast(key, trt.float32), trt.MatrixOperation.TRANSPOSE).get_output(0)
        scores = self.mul(scores, self.scalar(head_dim ** -0.5, 3, trt.float32))
        if mask is not None:
            scores = self.add(scores, mask)
        softmax = self.network.add_softmax(scores)
        softmax.axes = 4
        probabilities = self.cast(softmax.get_output(0), self.dtype)
        return self.network.add_matrix_multiply(probabilities, trt.MatrixOperation.NONE,
                                                value, trt.MatrixOperation.NONE).get_output(0)

    def output(self, value, name, dtype=None):
        result = self.cast(value, dtype) if dtype is not None else value
        result.name = name
        self.network.mark_output(result)

    def finish(self):
        self.config.add_optimization_profile(self.profile)
        plan = self.builder.build_serialized_network(self.network, self.config)
        if plan is None:
            raise RuntimeError("Cosmos3-Edge native TensorRT graph build failed")
        return bytes(plan)


def build_decoder(config: ModelConfig, weights, max_cache_length: int, *,
                  precision="fp32", max_input_length=None, verbose=False) -> bytes:
    """Build dynamic prefill/decode; return only new K/V rows for host placement.

    The mask is [1, sequence, cache_capacity + sequence], with invalid fixed
    cache slots masked by the caller. This permits empty-cache prefill without
    zero-sized runtime allocations and retains compact grouped K/V storage.
    """
    if max_input_length is None:
        max_input_length = max_cache_length
    if not 1 <= max_input_length <= max_cache_length <= config.max_position_embeddings:
        raise ValueError("Cosmos3-Edge input/cache capacity exceeds checkpoint context")
    g = _Graph(precision, verbose)
    n, h, d = g.network, config.hidden_size, config.head_dim
    x = n.add_input("inputs_embeds", g.dtype, (-1, h))
    cos = n.add_input("rope_cos", trt.float32, (-1, d))
    sin = n.add_input("rope_sin", trt.float32, (-1, d))
    mask = n.add_input("attention_mask", trt.float32, (1, -1, -1))
    opt = min(64, max_input_length)
    for name, width in (("inputs_embeds", h), ("rope_cos", d), ("rope_sin", d)):
        g.profile.set_shape(name, (1, width), (opt, width), (max_input_length, width))
    g.profile.set_shape("attention_mask", (1, 1, max_cache_length + 1),
                        (1, opt, max_cache_length + opt),
                        (1, max_input_length, max_cache_length + max_input_length))
    cos, sin = g.reshape(cos, (1, -1, d)), g.reshape(sin, (1, -1, d))
    indices = g.constant(np.r_[np.arange(d // 2, d), np.arange(d // 2)], trt.int32)
    signs = g.constant(np.r_[-np.ones(d // 2), np.ones(d // 2)].reshape(1, 1, d), trt.float32)
    repeat = g.constant(np.repeat(np.arange(config.num_key_value_heads),
                                  config.num_attention_heads // config.num_key_value_heads), trt.int32)

    def rotary(value):
        value = g.cast(value, trt.float32)
        rotated = g.mul(n.add_gather(value, indices, 2).get_output(0), signs)
        return g.cast(g.add(g.mul(value, cos), g.mul(rotated, sin)), g.dtype)

    for layer in range(config.num_hidden_layers):
        prefix = f"layers.{layer}"
        normalized = g.norm(x, weights, prefix + ".input_layernorm", config.rms_norm_eps)
        q, k, v = [g.reshape(g.linear(normalized, weights, prefix + ".self_attn.to_" + kind),
                             (-1, heads, d), (1, 0, 2))
                   for kind, heads in (("q", config.num_attention_heads),
                                       ("k", config.num_key_value_heads), ("v", config.num_key_value_heads))]
        q, k = rotary(q), rotary(k)
        g.output(k, f"present_k_{layer}")
        g.output(v, f"present_v_{layer}")
        cached = []
        for kind, current in (("k", k), ("v", v)):
            past = n.add_input(f"cache_{kind}_{layer}", g.dtype,
                               (config.num_key_value_heads, max_cache_length, d))
            concat = n.add_concatenation([past, current])
            concat.axis = 1
            cached.append(n.add_gather(concat.get_output(0), repeat, 0).get_output(0))
        attended = g.attention(q, cached[0], cached[1], d, mask)
        reorder = n.add_shuffle(attended)
        reorder.first_transpose = (1, 0, 2)
        reorder.reshape_dims = (-1, h)
        x = g.add(x, g.linear(reorder.get_output(0), weights, prefix + ".self_attn.to_out"))
        normalized = g.norm(x, weights, prefix + ".post_attention_layernorm", config.rms_norm_eps)
        up = g.linear(normalized, weights, prefix + ".mlp.up_proj")
        relu = n.add_activation(up, trt.ActivationType.RELU).get_output(0)
        x = g.add(x, g.linear(g.mul(relu, relu), weights, prefix + ".mlp.down_proj"))
    x = g.norm(x, weights, "norm", config.rms_norm_eps)
    g.output(g.linear(x, weights, "lm_head"), "logits", trt.float32)
    return g.finish()


def build_visual(config: ModelConfig, weights, max_patches: int, *, precision="fp32", verbose=False) -> bytes:
    """Build SigLIP2 plus the exact-GELU pre-shuffle patch merger for one frame.

    Input patch columns preserve the checkpoint's HWC ordering. Patch rows and
    positional interpolation rows use 2x2 merge-block order. Neither convention
    may be borrowed unchanged from Edge's CHW runtime preprocessing.
    """
    vision, projector = config.vision, config.projector
    width, heads = vision["hidden_size"], vision["num_attention_heads"]
    merge = projector["spatial_merge_size"] ** 2
    if max_patches < merge or max_patches % merge:
        raise ValueError("Cosmos3-Edge visual profile must contain complete merge blocks")
    g = _Graph(precision, verbose)
    n = g.network
    patch_width = vision["num_channels"] * vision["patch_size"] ** 2
    pixels = n.add_input("pixel_values", g.dtype, (-1, patch_width))
    table_size = vision["num_patches"]
    fractions = n.add_input("position_weights", trt.float32, (-1, table_size))
    opt = min(256, max_patches)
    g.profile.set_shape("pixel_values", (merge, patch_width), (opt, patch_width), (max_patches, patch_width))
    g.profile.set_shape("position_weights", (merge, table_size), (opt, table_size),
                        (max_patches, table_size))
    # Antialiasing widens support when shrinking either table axis. A dense
    # interpolation matrix represents every ratio without a fixed tap limit;
    # TensorRT owns its lowering and the learned table stays in the engine.
    positions = n.add_matrix_multiply(fractions, trt.MatrixOperation.NONE,
        g.constant(weights["model.visual.embeddings.position_embedding.weight"], trt.float32),
        trt.MatrixOperation.NONE).get_output(0)
    x = g.add(g.linear(pixels, weights, "model.visual.embeddings.patch_embedding", True),
              g.cast(positions, g.dtype))
    for layer in range(vision["num_hidden_layers"]):
        prefix = f"model.visual.encoder.layers.{layer}"
        norm = g.norm(x, weights, prefix + ".layer_norm1", vision["layer_norm_eps"], True)
        q, k, v = [g.reshape(g.linear(norm, weights, prefix + ".self_attn." + kind + "_proj", True),
                             (-1, heads, width // heads), (1, 0, 2)) for kind in ("q", "k", "v")]
        attended = g.attention(q, k, v, width // heads)
        shuffle = n.add_shuffle(attended)
        shuffle.first_transpose = (1, 0, 2)
        shuffle.reshape_dims = (-1, width)
        x = g.add(x, g.linear(shuffle.get_output(0), weights, prefix + ".self_attn.out_proj", True))
        norm = g.norm(x, weights, prefix + ".layer_norm2", vision["layer_norm_eps"], True)
        x = g.add(x, g.linear(g.gelu(g.linear(norm, weights, prefix + ".mlp.fc1", True), True),
                              weights, prefix + ".mlp.fc2", True))
    x = g.norm(x, weights, "model.visual.post_layernorm", vision["layer_norm_eps"], True)
    x = g.norm(x, weights, "model.projector.norm", vision["layer_norm_eps"], True)
    x = g.reshape(x, (-1, width * merge))
    x = g.gelu(g.linear(x, weights, "model.projector.linear_fc1", True))
    g.output(g.linear(x, weights, "model.projector.linear_fc2", True), "image_features")
    return g.finish()
