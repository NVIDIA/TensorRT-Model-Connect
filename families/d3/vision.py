# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""D3 native visual tower: checkpoint weights and TensorRT graph.

One invocation processes one temporal patch group. Attention never crosses these
boundaries in the reference; callers concatenate groups in provider order. Image
patches repeat the image across the temporal extent, while video patches retain
all source frames. No Edge runtime, ONNX, or custom GPU kernel is required.
"""
from dataclasses import dataclass
from math import isqrt
from pathlib import Path

import numpy as np
import ml_dtypes
import tensorrt as trt

from . import graph_ops
from .weights import load_projection
from .weights import _load_tensor

BF16 = ml_dtypes.bfloat16


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
            int(raw["hidden_size"]), int(raw["out_hidden_size"]), int(raw["depth"]),
            int(raw.get("num_heads", raw.get("num_attention_heads", 0))),
            int(raw["intermediate_size"]), int(raw.get("in_channels", 3)),
            int(raw["patch_size"]), int(raw.get("temporal_patch_size", 2)),
            int(raw.get("spatial_merge_size", 2)), int(raw["num_position_embeddings"]),
            float(raw.get("rope_parameters", {}).get("rope_theta", 10000.0)), raw.get("hidden_act", "gelu_pytorch_tanh"),
        )
        if (min(config.hidden, config.output, config.depth, config.heads, config.intermediate,
                config.channels, config.patch, config.temporal, config.merge, config.positions) <= 0
                or config.output != text_hidden or config.hidden % config.heads
                or config.head_dim % 4 or isqrt(config.positions) ** 2 != config.positions
                or not np.isfinite(config.theta) or config.theta <= 1
                or config.activation not in {"gelu_pytorch_tanh", "gelu_new", "gelu"}
                or raw.get("deepstack_visual_indexes")):
            raise ValueError("Unsupported D3 native visual geometry/activation")
        return config


def grid_inputs(config, height, width):
    """Provider-order rotary angles and bilinear learned-position inputs.

    This describes a spatial patch grid, not raw image pixel dimensions. The
    interpolation weights stay FP32 through embedding multiplication and
    reduction. No learned computation is done on CPU.
    """
    m = config.merge
    if min(height, width) <= 0 or height % m or width % m:
        raise ValueError("Visual patch grid must be positive and divisible by spatial merge")
    rows, cols = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    def ordered(array):
        return array.reshape(height // m, m, width // m, m).transpose(0, 2, 1, 3).reshape(-1)

    row, col = ordered(rows), ordered(cols)
    inverse = np.float32(1.0) / np.power(np.float32(config.theta),
                       np.arange(0, config.head_dim // 2, 2, dtype=np.float32) / (config.head_dim // 2))
    rotary = np.concatenate((row[:, None] * inverse, col[:, None] * inverse), axis=1)
    side = isqrt(config.positions)
    hs = np.linspace(0, side - 1, height, dtype=np.float32)
    ws = np.linspace(0, side - 1, width, dtype=np.float32)
    hf, wf = hs.astype(np.int32), ws.astype(np.int32)
    hc, wc = np.minimum(hf + 1, side - 1), np.minimum(wf + 1, side - 1)
    dh, dw = hs - hf.astype(np.float32), ws - wf.astype(np.float32)
    indices = [hf[:, None] * side + wf, hf[:, None] * side + wc,
               hc[:, None] * side + wf, hc[:, None] * side + wc]
    factors = [(1 - dh[:, None]) * (1 - dw), (1 - dh[:, None]) * dw,
               dh[:, None] * (1 - dw), dh[:, None] * dw]
    return {
        "rotary_angles": np.ascontiguousarray(rotary, dtype=np.float32),
        "position_indices": np.ascontiguousarray([ordered(x) for x in indices], dtype=np.int32),
        "position_weights": np.ascontiguousarray([ordered(x) for x in factors], dtype=np.float32),
    }


def load_vision_weights(model_dir: Path, config: VisionConfig, *, readers, prefix):
    """Read once, retaining original prepared projection bytes and scale tensors."""
    weights = {}

    def dense(name, shape):
        value = _load_tensor(readers, prefix + name)
        if value.shape != shape:
            raise ValueError(f"{prefix}{name}: expected {shape}, got {value.shape}")
        weights[name] = np.ascontiguousarray(value, dtype=BF16)

    def linear(name, out_width, in_width):
        source = prefix + name
        projection = load_projection(readers, source)
        if projection.shape != (in_width, out_width):
            raise ValueError(f"Invalid D3 visual projection: {source}")
        weights[name + ".weight"] = projection
        dense(name + ".bias", (out_width,))

    h = config.hidden
    dense("patch_embed.proj.weight", (h, config.channels, config.temporal, config.patch, config.patch))
    weights["patch_embed.proj.weight"] = np.ascontiguousarray(
        weights["patch_embed.proj.weight"].reshape(h, config.patch_width).T)
    dense("patch_embed.proj.bias", (h,))
    dense("pos_embed.weight", (config.positions, h))
    for i in range(config.depth):
        block = f"blocks.{i}"
        for norm in ("norm1", "norm2"):
            for kind in ("weight", "bias"):
                dense(f"{block}.{norm}.{kind}", (h,))
        for name, rows, columns in (("attn.qkv", 3 * h, h), ("attn.proj", h, h),
                                    ("mlp.linear_fc1", config.intermediate, h),
                                    ("mlp.linear_fc2", h, config.intermediate)):
            linear(f"{block}.{name}", rows, columns)
    for kind in ("weight", "bias"):
        dense(f"merger.norm.{kind}", (h,))
    merged = h * config.merge ** 2
    linear("merger.linear_fc1", merged, merged)
    linear("merger.linear_fc2", config.output, merged)
    return weights


def build_vision_engine(config, weights, max_patches, *, patches_per_image, debug_outputs=False, verbose=False):
    """Build a dynamic spatial-grid tower; all learned math stays in TensorRT."""
    unit = config.merge ** 2
    if (max_patches < patches_per_image or patches_per_image < unit
            or patches_per_image % unit or max_patches % patches_per_image):
        raise ValueError("Visual patch capacity must be a positive merge-unit multiple")
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    build = builder.create_builder_config()
    build.clear_flag(trt.BuilderFlag.TF32)
    build.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    build.builder_optimization_level = 3
    h, heads, dim = config.hidden, config.heads, config.head_dim
    patches = network.add_input("patches", trt.bfloat16, (-1, config.patch_width))
    angles = network.add_input("rotary_angles", trt.float32, (-1, dim // 2))
    indices = network.add_input("position_indices", trt.int32, (4, -1))
    factors = network.add_input("position_weights", trt.float32, (4, -1))
    for tensor in (patches, angles):
        tensor.set_dimension_name(0, "patch_count")
    for tensor in (indices, factors):
        tensor.set_dimension_name(1, "patch_count")
    profile = builder.create_optimization_profile()
    optimum = min(max_patches, patches_per_image * 4)
    for name, last in (("patches", config.patch_width), ("rotary_angles", dim // 2)):
        profile.set_shape(name, (patches_per_image, last), (optimum, last), (max_patches, last))
    for name in ("position_indices", "position_weights"):
        profile.set_shape(name, (4, patches_per_image), (4, optimum), (4, max_patches))
    build.add_optimization_profile(profile)

    def cast(x, dtype):
        return x if x.dtype == dtype else network.add_cast(x, dtype).get_output(0)

    def constant(x, dtype=np.float32):
        bf16 = np.dtype(dtype) == np.dtype(BF16)
        a = np.ascontiguousarray(x, dtype=np.float32 if bf16 else dtype)
        out = network.add_constant(a.shape, a).get_output(0)
        return cast(out, trt.bfloat16) if bf16 else out

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
        # F.linear accumulates the bias before the BF16 output is stored.
        # Keep that boundary explicit instead of rounding the dot first.
        y = graph_ops.add_matmul_rhs_constant(network, cast(x, trt.float32), input_width,
                                              output_width, weights[name + ".weight"], np.float32)
        return cast(add(y, constant(weights[name + ".bias"].reshape(1, output_width))), trt.bfloat16)

    def norm(x, name):
        return graph_ops.add_layer_norm(network, x, h, weights[name + ".weight"],
                                        weights[name + ".bias"], constant([[1e-6]]), BF16)

    def gelu(x, approximate):
        kind = trt.ActivationType.GELU_TANH if approximate else trt.ActivationType.GELU_ERF
        return cast(network.add_activation(cast(x, trt.float32), kind).get_output(0), trt.bfloat16)

    def debug(x, name):
        if debug_outputs:
            x = network.add_identity(x).get_output(0)
            x.name = "debug_" + name
            network.mark_output(x)

    hidden = linear(patches, "patch_embed.proj", config.patch_width, h)
    debug(hidden, "patch_projection")
    embedded = network.add_gather(constant(weights["pos_embed.weight"], BF16), indices, 0).get_output(0)
    # The pinned D3 reference multiplies BF16 embeddings by FP32 interpolation
    # factors and reduces all four taps in FP32 before one BF16 rounding.
    products = mul(cast(embedded, trt.float32), reshape(factors, (4, -1, 1)))
    position = network.add_reduce(products, trt.ReduceOperation.SUM, 1, keep_dims=False).get_output(0)
    position = cast(position, trt.bfloat16)
    debug(position, "position_embedding")
    hidden = add(hidden, position)
    debug(hidden, "patch_embedding")
    phase = network.add_concatenation([angles, angles])
    phase.axis = 1
    cosine = reshape(network.add_unary(phase.get_output(0), trt.UnaryOperation.COS).get_output(0), (-1, 1, dim))
    sine = reshape(network.add_unary(phase.get_output(0), trt.UnaryOperation.SIN).get_output(0), (-1, 1, dim))

    def rotate(x):
        x = cast(x, trt.float32)
        halves = reshape(x, (-1, heads, 2, dim // 2))
        a, b = gather(halves, 0, 2), gather(halves, 1, 2)
        neg = network.add_unary(b, trt.UnaryOperation.NEG).get_output(0)
        rotated = network.add_concatenation([neg, a])
        rotated.axis = 2
        return cast(add(mul(x, cosine), mul(reshape(rotated.get_output(0), (-1, heads, dim)), sine)), trt.bfloat16)

    def transpose_heads(x):
        layer = network.add_shuffle(reshape(x, (-1, patches_per_image, heads, dim)))
        layer.first_transpose = trt.Permutation([0, 2, 1, 3])
        return layer.get_output(0)

    def flatten_heads(x):
        layer = network.add_shuffle(x)
        layer.first_transpose = trt.Permutation([0, 2, 1, 3])
        layer.reshape_dims = (-1, h)
        return layer.get_output(0)

    for i in range(config.depth):
        prefix = f"blocks.{i}"
        normalized = norm(hidden, prefix + ".norm1")
        projected = reshape(linear(normalized, prefix + ".attn.qkv", h, 3 * h), (-1, 3, heads, dim))
        query, key, value = [reshape(gather(projected, part, 1), (-1, heads, dim)) for part in range(3)]
        query, key, value = transpose_heads(rotate(query)), transpose_heads(rotate(key)), transpose_heads(value)
        scores = network.add_matrix_multiply(cast(query, trt.float32), trt.MatrixOperation.NONE,
                                             cast(key, trt.float32), trt.MatrixOperation.TRANSPOSE).get_output(0)
        # D3 selects SDPA: retain FP32 attention arithmetic until its BF16 output.
        scores = mul(scores, constant([[[[dim ** -0.5]]]]))
        softmax = network.add_softmax(scores)
        softmax.axes = 1 << 3
        probabilities = softmax.get_output(0)
        attention = network.add_matrix_multiply(cast(probabilities, trt.float32), trt.MatrixOperation.NONE,
                                                cast(value, trt.float32), trt.MatrixOperation.NONE).get_output(0)
        attention = cast(flatten_heads(attention), trt.bfloat16)
        hidden = add(hidden, linear(attention, prefix + ".attn.proj", h, h))
        normalized = norm(hidden, prefix + ".norm2")
        feed = linear(normalized, prefix + ".mlp.linear_fc1", h, config.intermediate)
        feed = gelu(feed, config.activation != "gelu")
        hidden = add(hidden, linear(feed, prefix + ".mlp.linear_fc2", config.intermediate, h))
        debug(hidden, f"block_{i}")
    merged = reshape(norm(hidden, "merger.norm"), (-1, h * unit))
    merged = linear(merged, "merger.linear_fc1", h * unit, h * unit)
    # HF D3/3.6/3.8 PatchMerger uses exact nn.GELU independently of the
    # block activation. The pinned Edge implementation uses an approximation.
    merged = gelu(merged, False)
    result = linear(merged, "merger.linear_fc2", h * unit, config.output)
    result.name = "visual_features"
    network.mark_output(result)
    plan = builder.build_serialized_network(network, build)
    if plan is None:
        raise RuntimeError("Native D3 visual engine build failed")
    return bytes(plan)
