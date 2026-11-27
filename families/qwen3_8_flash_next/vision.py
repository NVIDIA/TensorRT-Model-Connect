# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flash-Next native visual tower: checkpoint weights and TensorRT graph.

One invocation processes one temporal patch group. Attention never crosses these
boundaries in the reference; callers concatenate groups in provider order. Image
patches repeat the image across the temporal extent, while video patches retain
all source frames. No Edge runtime, ONNX, or custom GPU kernel is required.
"""

from dataclasses import dataclass
from math import isqrt

import numpy as np
import tensorrt as trt


@dataclass(frozen=True)
class VisionConfig:
    hidden: int
    output: int
    depth: int
    heads: int
    intermediate: int
    channels: int
    patch: int
    temporal: int
    merge: int
    positions: int
    theta: float
    activation: str

    @property
    def head_dim(self):
        return self.hidden // self.heads

    @property
    def patch_width(self):
        return self.channels * self.temporal * self.patch * self.patch

    @classmethod
    def from_dict(cls, raw, text_hidden):
        config = cls(
            int(raw["hidden_size"]),
            int(raw["out_hidden_size"]),
            int(raw["depth"]),
            int(raw.get("num_heads", raw.get("num_attention_heads", 0))),
            int(raw["intermediate_size"]),
            int(raw.get("in_channels", 3)),
            int(raw["patch_size"]),
            int(raw.get("temporal_patch_size", 2)),
            int(raw.get("spatial_merge_size", 2)),
            int(raw["num_position_embeddings"]),
            float(raw.get("rope_theta", 10000.0)),
            raw.get("hidden_act", "gelu_pytorch_tanh"),
        )
        if (
            min(
                config.hidden,
                config.output,
                config.depth,
                config.heads,
                config.intermediate,
                config.channels,
                config.patch,
                config.temporal,
                config.merge,
                config.positions,
            )
            <= 0
            or config.output != text_hidden
            or config.hidden % config.heads
            or config.head_dim % 4
            or isqrt(config.positions) ** 2 != config.positions
            or not np.isfinite(config.theta)
            or config.theta <= 1
            or config.activation not in {"gelu_pytorch_tanh", "gelu_new", "gelu"}
            or raw.get("deepstack_visual_indexes")
        ):
            raise ValueError("Unsupported Flash-Next native visual geometry/activation")
        return config


def grid_inputs(config, height, width):
    """Provider-order rotary angles and bilinear learned-position inputs.

    This describes a spatial patch grid, not raw image pixel dimensions. The
    four interpolation weights remain FP32 through the bilinear sum, then
    the model rounds the resulting position embedding. No learned math runs on CPU.
    """
    m = config.merge
    if min(height, width) <= 0 or height % m or width % m:
        raise ValueError("Visual patch grid must be positive and divisible by spatial merge")
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    def ordered(array):
        return array.reshape(height // m, m, width // m, m).transpose(0, 2, 1, 3).reshape(-1)

    row, col = ordered(rows), ordered(cols)
    inverse = np.power(
        np.float32(config.theta),
        -np.arange(0, config.head_dim // 2, 2, dtype=np.float32) / (config.head_dim // 2),
    )
    rotary = np.concatenate((row[:, None] * inverse, col[:, None] * inverse), axis=1)
    side = isqrt(config.positions)
    hs = np.linspace(0, side - 1, height, dtype=np.float32)
    ws = np.linspace(0, side - 1, width, dtype=np.float32)
    hf, wf = hs.astype(np.int32), ws.astype(np.int32)
    hc, wc = np.minimum(hf + 1, side - 1), np.minimum(wf + 1, side - 1)
    dh, dw = hs - hf.astype(np.float32), ws - wf.astype(np.float32)
    indices = [
        hf[:, None] * side + wf,
        hf[:, None] * side + wc,
        hc[:, None] * side + wf,
        hc[:, None] * side + wc,
    ]
    factors = [
        (1 - dh[:, None]) * (1 - dw),
        (1 - dh[:, None]) * dw,
        dh[:, None] * (1 - dw),
        dh[:, None] * dw,
    ]
    return {
        "rotary_angles": np.ascontiguousarray(rotary, dtype=np.float32),
        "position_indices": np.ascontiguousarray([ordered(x) for x in indices], dtype=np.int32),
        "position_weights": np.ascontiguousarray([ordered(x) for x in factors], dtype=np.float32),
    }


def load_vision_weights(checkpoint, config):
    """Read BF16 from the original visual tower; no expert quantization applies."""
    weights = {}

    def dense(name, shape):
        weights[name] = checkpoint.bf16("model.visual." + name, shape)

    def linear(name, rows, columns):
        dense(name + ".weight", (rows, columns))
        dense(name + ".bias", (rows,))

    h = config.hidden
    dense(
        "patch_embed.proj.weight", (h, config.channels, config.temporal, config.patch, config.patch)
    )
    weights["patch_embed.proj.weight"] = np.ascontiguousarray(
        weights["patch_embed.proj.weight"].reshape(h, config.patch_width)
    )
    dense("patch_embed.proj.bias", (h,))
    dense("pos_embed.weight", (config.positions, h))
    for i in range(config.depth):
        block = f"blocks.{i}"
        for norm in ("norm1", "norm2"):
            for kind in ("weight", "bias"):
                dense(f"{block}.{norm}.{kind}", (h,))
        for name, rows, columns in (
            ("attn.qkv", 3 * h, h),
            ("attn.proj", h, h),
            ("mlp.linear_fc1", config.intermediate, h),
            ("mlp.linear_fc2", h, config.intermediate),
        ):
            linear(f"{block}.{name}", rows, columns)
    for kind in ("weight", "bias"):
        dense(f"merger.norm.{kind}", (h,))
    merged = h * config.merge**2
    linear("merger.linear_fc1", merged, merged)
    linear("merger.linear_fc2", config.output, merged)
    return weights


def build_vision_engine(config, weights, max_patches):
    """Build a dynamic spatial-grid tower; all learned math stays in TensorRT."""
    unit = config.merge**2
    if max_patches < unit or max_patches % unit:
        raise ValueError("Visual patch capacity must be a positive merge-unit multiple")
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    build = builder.create_builder_config()
    build.builder_optimization_level = 3
    h, heads, dim = config.hidden, config.heads, config.head_dim
    patches = network.add_input("patches", trt.float16, (-1, config.patch_width))
    angles = network.add_input("rotary_angles", trt.float32, (-1, dim // 2))
    indices = network.add_input("position_indices", trt.int32, (4, -1))
    factors = network.add_input("position_weights", trt.float32, (4, -1))
    for tensor in (patches, angles):
        tensor.set_dimension_name(0, "patch_count")
    for tensor in (indices, factors):
        tensor.set_dimension_name(1, "patch_count")
    profile = builder.create_optimization_profile()
    optimum = min(max_patches, 256)
    for name, last in (("patches", config.patch_width), ("rotary_angles", dim // 2)):
        profile.set_shape(name, (unit, last), (optimum, last), (max_patches, last))
    for name in ("position_indices", "position_weights"):
        profile.set_shape(name, (4, unit), (4, optimum), (4, max_patches))
    build.add_optimization_profile(profile)

    def cast(x, dtype):
        return x if x.dtype == dtype else network.add_cast(x, dtype).get_output(0)

    def constant(x, dtype=np.float32):
        a = np.ascontiguousarray(x, dtype=dtype)
        return network.add_constant(a.shape, a).get_output(0)

    def reshape(x, shape):
        layer = network.add_shuffle(x)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def binary(a, b, operation):
        return network.add_elementwise(a, b, operation).get_output(0)

    def add(a, b):
        return binary(a, b, trt.ElementWiseOperation.SUM)

    def mul(a, b):
        return binary(a, b, trt.ElementWiseOperation.PROD)

    def gather(x, index, axis):
        return network.add_gather(x, constant([index], np.int32), axis).get_output(0)

    def linear(x, name, input_width, output_width):
        w = weights[name + ".weight"]
        if w.shape != (output_width, input_width):
            raise ValueError(f"Invalid Flash-Next visual projection: {name}")
        w = cast(constant(w, np.float16), trt.float32)
        y = network.add_matrix_multiply(
            cast(x, trt.float32), trt.MatrixOperation.NONE, w, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)
        if name == "patch_embed.proj":
            # The trained Conv3D rounds its dot product before the bias add.
            y = cast(cast(y, trt.float16), trt.float32)
        bias = constant(weights[name + ".bias"].reshape(1, output_width))
        return cast(add(y, bias), trt.float16)

    def norm(x, name):
        x = cast(x, trt.float32)
        mean = network.add_reduce(x, trt.ReduceOperation.AVG, 1 << 1, True).get_output(0)
        centered = binary(x, mean, trt.ElementWiseOperation.SUB)
        var = network.add_reduce(
            mul(centered, centered), trt.ReduceOperation.AVG, 1 << 1, True
        ).get_output(0)
        denom = network.add_unary(add(var, constant([[1e-6]])), trt.UnaryOperation.SQRT).get_output(
            0
        )
        normalized = binary(centered, denom, trt.ElementWiseOperation.DIV)
        gamma = constant(weights[name + ".weight"].reshape(1, h))
        bias = constant(weights[name + ".bias"].reshape(1, h))
        return cast(add(mul(normalized, gamma), bias), trt.float16)

    def gelu(x, approximate):
        kind = trt.ActivationType.GELU_TANH if approximate else trt.ActivationType.GELU_ERF
        return cast(network.add_activation(cast(x, trt.float32), kind).get_output(0), trt.float16)

    hidden = linear(patches, "patch_embed.proj", config.patch_width, h)
    embedded = network.add_gather(
        constant(weights["pos_embed.weight"], np.float16), indices, 0
    ).get_output(0)
    # The checkpoint processor supplies FP32 interpolation factors. HF
    # promotes the learned positional rows for the weighted sum and rounds
    # only the result back to the visual hidden dtype.
    weighted = mul(cast(embedded, trt.float32), reshape(factors, (4, -1, 1)))
    position = cast(
        network.add_reduce(weighted, trt.ReduceOperation.SUM, 1, False).get_output(0), trt.float16
    )
    hidden = add(hidden, position)
    phase = network.add_concatenation([angles, angles])
    phase.axis = 1
    cosine = reshape(
        network.add_unary(phase.get_output(0), trt.UnaryOperation.COS).get_output(0), (-1, 1, dim)
    )
    sine = reshape(
        network.add_unary(phase.get_output(0), trt.UnaryOperation.SIN).get_output(0), (-1, 1, dim)
    )

    def rotate(x):
        x = cast(x, trt.float32)
        halves = reshape(x, (-1, heads, 2, dim // 2))
        a, b = gather(halves, 0, 2), gather(halves, 1, 2)
        neg = network.add_unary(b, trt.UnaryOperation.NEG).get_output(0)
        rotated = network.add_concatenation([neg, a])
        rotated.axis = 2
        return cast(
            add(mul(x, cosine), mul(reshape(rotated.get_output(0), (-1, heads, dim)), sine)),
            trt.float16,
        )

    def transpose_heads(x):
        layer = network.add_shuffle(x)
        layer.first_transpose = trt.Permutation([1, 0, 2])
        return layer.get_output(0)

    for i in range(config.depth):
        prefix = f"blocks.{i}"
        normalized = norm(hidden, prefix + ".norm1")
        projected = reshape(linear(normalized, prefix + ".attn.qkv", h, 3 * h), (-1, 3, heads, dim))
        query, key, value = [
            reshape(gather(projected, part, 1), (-1, heads, dim)) for part in range(3)
        ]
        query, key, value = (
            transpose_heads(rotate(query)),
            transpose_heads(rotate(key)),
            transpose_heads(value),
        )
        scores = network.add_matrix_multiply(
            cast(query, trt.float32),
            trt.MatrixOperation.NONE,
            cast(key, trt.float32),
            trt.MatrixOperation.TRANSPOSE,
        ).get_output(0)
        # Provider eager attention rounds the QK product and scaled scores
        # to the query dtype before its explicit FP32 softmax.
        scores = cast(cast(scores, trt.float16), trt.float32)
        scores = cast(cast(mul(scores, constant([[[dim**-0.5]]])), trt.float16), trt.float32)
        softmax = network.add_softmax(scores)
        softmax.axes = 1 << 2
        probabilities = cast(softmax.get_output(0), trt.float16)
        attention = network.add_matrix_multiply(
            cast(probabilities, trt.float32),
            trt.MatrixOperation.NONE,
            cast(value, trt.float32),
            trt.MatrixOperation.NONE,
        ).get_output(0)
        attention = cast(reshape(transpose_heads(attention), (-1, h)), trt.float16)
        hidden = add(hidden, linear(attention, prefix + ".attn.proj", h, h))
        normalized = norm(hidden, prefix + ".norm2")
        feed = linear(normalized, prefix + ".mlp.linear_fc1", h, config.intermediate)
        feed = gelu(feed, config.activation != "gelu")
        hidden = add(hidden, linear(feed, prefix + ".mlp.linear_fc2", config.intermediate, h))
    merged = reshape(norm(hidden, "merger.norm"), (-1, h * unit))
    merged = linear(merged, "merger.linear_fc1", h * unit, h * unit)
    # The Flash-Next PatchMerger uses exact nn.GELU independently of the
    # block activation. The decoder block activation does not control the merger.
    merged = gelu(merged, False)
    result = linear(merged, "merger.linear_fc2", h * unit, config.output)
    result.name = "visual_features"
    network.mark_output(result)
    plan = builder.build_serialized_network(network, build)
    if plan is None:
        raise RuntimeError("Native Flash-Next visual engine build failed")
    return bytes(plan)
