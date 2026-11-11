# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HunYuanVL decoder and vision graphs built directly with the TensorRT Python API.

The decoder consumes fused text/image embeddings plus explicit multi-axis RoPE
cosines/sines and returns only the new key/value rows and the last row's
logits; the native runtime owns positions, cache placement and the request
lifecycle. The vision engine encodes one image per invocation (attention never
crosses images) and emits the merged grid tokens followed by the learned
row-separator, image-begin and image-end tokens, all after the projector norm.
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .config import ModelConfig

_DTYPES = {"fp32": trt.float32, "bf16": trt.bfloat16}


class _Graph:
    def __init__(self, precision: str, verbose: bool):
        if precision not in _DTYPES:
            raise ValueError("HunYuanVL precision must be fp32 or bf16")
        self.dtype = _DTYPES[precision]
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        self.builder = trt.Builder(self.logger)
        self.network = self.builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        )
        self.config = self.builder.create_builder_config()
        self.config.builder_optimization_level = 3
        if precision == "fp32":
            # TF32 would change FP32 matrix products relative to the FP32 reference.
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

    def transpose(self, value, permutation, shape=None):
        layer = self.network.add_shuffle(value)
        layer.first_transpose = permutation
        if shape is not None:
            layer.reshape_dims = shape
        return layer.get_output(0)

    def binary(self, left, right, operation):
        return self.network.add_elementwise(left, right, operation).get_output(0)

    def add(self, left, right):
        return self.binary(left, right, trt.ElementWiseOperation.SUM)

    def mul(self, left, right):
        return self.binary(left, right, trt.ElementWiseOperation.PROD)

    def matmul(self, value, matrix):
        """``value @ matrix.T`` for a constant [out, in] matrix."""
        return self.network.add_matrix_multiply(
            value, trt.MatrixOperation.NONE, self.constant(matrix), trt.MatrixOperation.TRANSPOSE
        ).get_output(0)

    def linear(self, value, weights, prefix, bias=False):
        result = self.matmul(value, weights[prefix + ".weight"])
        if bias:
            result = self.add(result, self.constant(weights[prefix + ".bias"][None, :]))
        return result

    def rms_norm(self, value, weight, eps):
        """Reference RMSNorm: FP32 statistics, rounded to the graph dtype, then scaled."""
        rank = len(value.shape)
        x = self.cast(value, trt.float32)
        mean_square = self.network.add_reduce(
            self.mul(x, x), trt.ReduceOperation.AVG, 1 << (rank - 1), True
        ).get_output(0)
        denominator = self.network.add_unary(
            self.add(mean_square, self.scalar(eps, rank, trt.float32)), trt.UnaryOperation.SQRT
        ).get_output(0)
        x = self.binary(x, denominator, trt.ElementWiseOperation.DIV)
        return self.mul(
            self.cast(x, self.dtype), self.constant(weight.reshape((1,) * (rank - 1) + (-1,)))
        )

    def layer_norm(self, value, weights, prefix, eps):
        """Biased LayerNorm computed in FP32 and rounded once to the graph dtype."""
        x = self.cast(value, trt.float32)
        mean = self.network.add_reduce(x, trt.ReduceOperation.AVG, 2, True).get_output(0)
        x = self.binary(x, mean, trt.ElementWiseOperation.SUB)
        variance = self.network.add_reduce(
            self.mul(x, x), trt.ReduceOperation.AVG, 2, True
        ).get_output(0)
        denominator = self.network.add_unary(
            self.add(variance, self.scalar(eps, 2, trt.float32)), trt.UnaryOperation.SQRT
        ).get_output(0)
        x = self.binary(x, denominator, trt.ElementWiseOperation.DIV)
        x = self.mul(x, self.constant(weights[prefix + ".weight"][None, :], trt.float32))
        x = self.add(x, self.constant(weights[prefix + ".bias"][None, :], trt.float32))
        return self.cast(x, self.dtype)

    def gelu(self, value):
        """Exact (erf) GELU evaluated in FP32 and rounded once."""
        x = self.cast(value, trt.float32)
        rank = len(value.shape)
        erf = self.network.add_unary(
            self.mul(x, self.scalar(1 / np.sqrt(2), rank, trt.float32)), trt.UnaryOperation.ERF
        ).get_output(0)
        x = self.mul(
            self.mul(x, self.scalar(0.5, rank, trt.float32)),
            self.add(erf, self.scalar(1, rank, trt.float32)),
        )
        return self.cast(x, self.dtype)

    def silu(self, value):
        """SiLU evaluated in FP32 and rounded once."""
        x = self.cast(value, trt.float32)
        sigmoid = self.network.add_activation(x, trt.ActivationType.SIGMOID).get_output(0)
        return self.cast(self.mul(x, sigmoid), self.dtype)

    def explicit_attention(self, query, key, value, head_dim, mask):
        """[heads, q, d] x [heads, k, d] with an additive FP32 mask; FP32 scores and softmax."""
        scores = self.network.add_matrix_multiply(
            self.cast(query, trt.float32),
            trt.MatrixOperation.NONE,
            self.cast(key, trt.float32),
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)
        scores = self.add(self.mul(scores, self.scalar(head_dim**-0.5, 3, trt.float32)), mask)
        softmax = self.network.add_softmax(scores)
        softmax.axes = 1 << 2
        probabilities = self.cast(softmax.get_output(0), self.dtype)
        return self.network.add_matrix_multiply(
            probabilities, trt.MatrixOperation.NONE, value, trt.MatrixOperation.NONE
        ).get_output(0)

    def fused_attention(self, query, key, value, head_dim):
        """Unmasked bidirectional attention on [1, heads, n, d] through IAttention.

        IAttention does not scale scores, so the query carries the scale; it is applied in FP32
        (a BF16 constant would be 0.15 % off) and rounded once to the activation dtype.
        """
        query = self.cast(
            self.mul(self.cast(query, trt.float32), self.scalar(head_dim**-0.5, 4, trt.float32)),
            self.dtype,
        )
        layer = self.network.add_attention(
            query, key, value, trt.AttentionNormalizationOp.SOFTMAX, False
        )
        if layer is None:
            raise RuntimeError("HunYuanVL vision attention could not be created")
        # Fall back to primitive layers when no fused kernel covers the head size or dtype.
        layer.decomposable = True
        return layer.get_output(0)

    def output(self, value, name, dtype=None):
        result = self.cast(value, dtype) if dtype is not None else value
        result.name = name
        self.network.mark_output(result)

    def finish(self, what):
        self.config.add_optimization_profile(self.profile)
        plan = self.builder.build_serialized_network(self.network, self.config)
        if plan is None:
            raise RuntimeError(f"HunYuanVL {what} TensorRT build failed")
        return bytes(plan)


def build_decoder(
    config: ModelConfig,
    weights,
    max_cache_length: int,
    *,
    max_input_length: int,
    precision: str = "bf16",
    verbose: bool = False,
) -> bytes:
    """Build dynamic prefill/decode over a fixed-capacity host-placed cache.

    Inputs: ``inputs_embeds`` [S, hidden], ``rope_cos``/``rope_sin`` [S, head_dim] (FP32, the
    reference's per-axis layout), ``attention_mask`` [1, S, capacity + S] (additive FP32; the caller
    masks unused cache slots and future rows) and ``cache_{k,v}_{layer}`` [kv_heads, capacity,
    head_dim]. Outputs: ``logits`` [1, vocab] (FP32, last row only) and the new rotated, normalized
    key rows and value rows ``present_{k,v}_{layer}`` [kv_heads, S, head_dim].
    """
    if not 1 <= max_input_length <= max_cache_length <= config.max_position_embeddings:
        raise ValueError("HunYuanVL input/cache capacity exceeds the checkpoint context")
    g = _Graph(precision, verbose)
    n, h, d = g.network, config.hidden_size, config.head_dim
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    x = n.add_input("inputs_embeds", g.dtype, (-1, h))
    cos = n.add_input("rope_cos", trt.float32, (-1, d))
    sin = n.add_input("rope_sin", trt.float32, (-1, d))
    mask = n.add_input("attention_mask", trt.float32, (1, -1, -1))
    opt = min(256, max_input_length)
    for name, width in (("inputs_embeds", h), ("rope_cos", d), ("rope_sin", d)):
        g.profile.set_shape(name, (1, width), (opt, width), (max_input_length, width))
    g.profile.set_shape(
        "attention_mask",
        (1, 1, max_cache_length + 1),
        (1, opt, max_cache_length + opt),
        (1, max_input_length, max_cache_length + max_input_length),
    )
    cos, sin = g.reshape(cos, (1, -1, d)), g.reshape(sin, (1, -1, d))
    rotate = g.constant(np.r_[np.arange(d // 2, d), np.arange(d // 2)], trt.int32)
    signs = g.constant(np.r_[-np.ones(d // 2), np.ones(d // 2)].reshape(1, 1, d), trt.float32)
    repeat = g.constant(np.repeat(np.arange(kv_heads), heads // kv_heads), trt.int32)

    def rotary(value):
        value = g.cast(value, trt.float32)
        rotated = g.mul(n.add_gather(value, rotate, 2).get_output(0), signs)
        return g.cast(g.add(g.mul(value, cos), g.mul(rotated, sin)), g.dtype)

    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        normalized = g.rms_norm(x, weights[prefix + ".input_layernorm.weight"], config.rms_norm_eps)
        q, k, v = [
            g.reshape(
                g.linear(normalized, weights, f"{prefix}.self_attn.{kind}_proj"),
                (-1, count, d),
                (1, 0, 2),
            )
            for kind, count in (("q", heads), ("k", kv_heads), ("v", kv_heads))
        ]
        q = g.rms_norm(
            rotary(q), weights[prefix + ".self_attn.query_layernorm.weight"], config.rms_norm_eps
        )
        k = g.rms_norm(
            rotary(k), weights[prefix + ".self_attn.key_layernorm.weight"], config.rms_norm_eps
        )
        g.output(k, f"present_k_{layer}")
        g.output(v, f"present_v_{layer}")
        cached = []
        for kind, current in (("k", k), ("v", v)):
            past = n.add_input(f"cache_{kind}_{layer}", g.dtype, (kv_heads, max_cache_length, d))
            concat = n.add_concatenation([past, current])
            concat.axis = 1
            cached.append(n.add_gather(concat.get_output(0), repeat, 0).get_output(0))
        attended = g.explicit_attention(q, cached[0], cached[1], d, mask)
        attended = g.transpose(attended, (1, 0, 2), (-1, heads * d))
        x = g.add(x, g.linear(attended, weights, prefix + ".self_attn.o_proj"))
        normalized = g.rms_norm(
            x, weights[prefix + ".post_attention_layernorm.weight"], config.rms_norm_eps
        )
        gate = g.silu(g.linear(normalized, weights, prefix + ".mlp.gate_proj"))
        up = g.linear(normalized, weights, prefix + ".mlp.up_proj")
        x = g.add(x, g.linear(g.mul(gate, up), weights, prefix + ".mlp.down_proj"))
    x = g.rms_norm(x, weights["model.norm.weight"], config.rms_norm_eps)
    # Only the last row feeds the sampler; the head is tied to the token embedding.
    rows = n.add_slice(n.add_shape(x).get_output(0), (0,), (1,), (1,)).get_output(0)
    last = g.binary(
        rows,
        n.add_constant((1,), np.array([1], dtype=np.int64)).get_output(0),
        trt.ElementWiseOperation.SUB,
    )
    x = n.add_gather(x, last, 0).get_output(0)
    g.output(g.matmul(x, weights["model.embed_tokens.weight"]), "logits", trt.float32)
    return g.finish("decoder")


def merge_projection(weight: np.ndarray) -> np.ndarray:
    """The 2x2-stride merge convolution as a matrix over (dy, dx, channel) block rows."""
    out_channels, channels, height, width = weight.shape
    return weight.transpose(0, 2, 3, 1).reshape(out_channels, height * width * channels)


def build_visual(
    config: ModelConfig,
    weights,
    max_patches: int,
    *,
    precision: str = "bf16",
    verbose: bool = False,
) -> bytes:
    """Build the vision tower and patch merger for one image.

    Inputs: ``pixel_values`` [N, 3 * patch * patch] in 2x2 merge-block order (channel-major
    within a patch), ``position_ids`` [N, 4] (rows of the learned position table for the
    bilinear corners 00, 01, 10, 11) and ``position_lambdas`` [N, 4] (FP32 h0, h1, w0, w1).
    Output: ``image_features`` [N / 4 + 3, hidden]: merged grid tokens in row-major order,
    then the row separator, the image-begin and the image-end tokens.
    """
    merge = config.merge_size**2
    if max_patches < merge or max_patches % merge:
        raise ValueError("HunYuanVL vision profile must hold complete merge blocks")
    g = _Graph(precision, verbose)
    n = g.network
    width, heads = config.vision_hidden_size, config.vision_heads
    head_dim, eps = width // heads, config.vision_eps
    patch_width = 3 * config.patch_size**2
    pixels = n.add_input("pixel_values", g.dtype, (-1, patch_width))
    ids = n.add_input("position_ids", trt.int32, (-1, 4))
    lambdas = n.add_input("position_lambdas", trt.float32, (-1, 4))
    opt = min(1024, max_patches)
    for name, columns in (
        ("pixel_values", patch_width),
        ("position_ids", 4),
        ("position_lambdas", 4),
    ):
        g.profile.set_shape(name, (merge, columns), (opt, columns), (max_patches, columns))

    patch = weights["vit.embeddings.patch_embedding.weight"]
    x = g.add(
        g.matmul(pixels, patch.reshape(patch.shape[0], -1)),
        g.constant(weights["vit.embeddings.patch_embedding.bias"][None, :]),
    )
    table = g.constant(weights["vit.embeddings.position_embedding.weight"], trt.float32)
    corners = n.add_gather(table, ids, 0).get_output(0)  # [N, 4, width]

    def column(tensor, index):
        return n.add_gather(tensor, g.constant(np.array([index]), trt.int32), 1).get_output(0)

    h0, h1, w0, w1 = (column(lambdas, i) for i in range(4))  # [N, 1] each
    x00, x01, x10, x11 = (
        g.reshape(column(corners, i), (-1, width)) for i in range(4)
    )  # [N, width]
    # The reference's bilinear evaluation order: h0 * (w0 * x00 + w1 * x01) + h1 * (w0 * x10 + w1 * x11).
    top = g.add(g.mul(w0, x00), g.mul(w1, x01))
    bottom = g.add(g.mul(w0, x10), g.mul(w1, x11))
    x = g.add(x, g.cast(g.add(g.mul(h0, top), g.mul(h1, bottom)), g.dtype))

    def split_heads(value):
        return g.reshape(
            g.reshape(value, (-1, heads, head_dim), (1, 0, 2)), (1, heads, -1, head_dim)
        )

    for layer in range(config.vision_layers):
        prefix = f"vit.layers.{layer}"
        normalized = g.layer_norm(x, weights, prefix + ".input_layernorm", eps)
        q, k, v = [
            split_heads(g.linear(normalized, weights, f"{prefix}.self_attn.{kind}_proj", True))
            for kind in ("q", "k", "v")
        ]
        attended = g.fused_attention(q, k, v, head_dim)
        attended = g.transpose(g.reshape(attended, (heads, -1, head_dim)), (1, 0, 2), (-1, width))
        x = g.add(x, g.linear(attended, weights, prefix + ".self_attn.o_proj", True))
        normalized = g.layer_norm(x, weights, prefix + ".post_attention_layernorm", eps)
        hidden = g.gelu(g.linear(normalized, weights, prefix + ".mlp.dense_h_to_4h", True))
        x = g.add(x, g.linear(hidden, weights, prefix + ".mlp.dense_4h_to_h", True))

    x = g.rms_norm(x, weights["vit.perceive.before_rms.weight"], eps)
    x = g.reshape(x, (-1, width * merge))
    x = g.add(
        g.matmul(x, merge_projection(weights["vit.perceive.proj.0.weight"])),
        g.constant(weights["vit.perceive.proj.0.bias"][None, :]),
    )
    x = g.gelu(x)
    out = weights["vit.perceive.proj.2.weight"]
    x = g.add(
        g.matmul(x, out.reshape(out.shape[0], -1)),
        g.constant(weights["vit.perceive.proj.2.bias"][None, :]),
    )
    newline = n.add_concatenation([x, g.constant(weights["vit.perceive.image_newline"][None, :])])
    newline.axis = 0
    x = g.linear(newline.get_output(0), weights, "vit.perceive.mlp", True)
    bounded = n.add_concatenation(
        [
            x,
            g.constant(weights["vit.perceive.image_begin"][None, :]),
            g.constant(weights["vit.perceive.image_end"][None, :]),
        ]
    )
    bounded.axis = 0
    x = g.rms_norm(bounded.get_output(0), weights["vit.perceive.after_rms.weight"], eps)
    g.output(x, "image_features")
    return g.finish("vision")
