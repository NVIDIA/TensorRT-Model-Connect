# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MiniCPM-V 4.6 vision tower, windowed merger and projector as one TensorRT engine.

One launch encodes one image slice of ``N`` patches (a ``h x w`` patch grid,
both multiples of four). Attention never crosses slices in the reference, so
the runtime encodes the source image and every slice separately and
concatenates the features in provider order.

Dataflow (16x token compression, the checkpoint's default):

1. patch embedding (a 14 x 14 stride-14 convolution, written as a matmul over
   flattened patches) plus a learned position row per patch;
2. encoder layers ``0..insert_layer_id`` over all ``N`` patches;
3. the windowed merger: attention inside every 2 x 2 window, then one MLP per
   window that maps the four concatenated patches to one token (plus their
   mean), giving ``N / 4`` tokens in window order;
4. the remaining encoder layers and the post layer norm over ``N / 4`` tokens;
5. the projector: every 2 x 2 block of merged tokens through LayerNorm, an
   exact-GELU MLP and a projection to the language model width (``N / 16``).

The runtime supplies the integer inputs that depend only on the grid shape:
``position_ids`` (bucketed learned-position rows), ``window_index`` (patches in
window order) and ``merge_index`` (merged tokens in projector block order).
"""

from __future__ import annotations

import numpy as np
import tensorrt as trt

from .checkpoint_mapper import _load_tensor, narrowed
from .config import VisionConfig

# Patches per slice: 4 x 4 is the smallest grid of the 16x mode; the largest
# bounds attention memory for very elongated images.
MIN_PATCHES = 16
OPT_PATCHES = 1024
MAX_PATCHES = 4096
PROJECTOR_EPS = 1e-6


def load_vision_weights(readers, config: VisionConfig, text_hidden: int) -> dict[str, np.ndarray]:
    """Read the vision tower, windowed merger and projector as FP32 arrays."""
    weights: dict[str, np.ndarray] = {}
    h = config.hidden_size
    window = 4 * h

    def dense(name: str, shape: tuple[int, ...]) -> np.ndarray:
        return np.ascontiguousarray(_load_tensor(readers, name, shape), dtype=np.float32)

    def linear(source: str, target: str, out_width: int, in_width: int) -> None:
        weights[target + ".weight"] = np.ascontiguousarray(dense(source + ".weight", (out_width, in_width)).T)
        weights[target + ".bias"] = dense(source + ".bias", (out_width,))

    def norm(source: str, target: str, width: int) -> None:
        weights[target + ".weight"] = dense(source + ".weight", (width,))
        weights[target + ".bias"] = dense(source + ".bias", (width,))

    tower = "model.vision_tower"
    patch = dense(f"{tower}.embeddings.patch_embedding.weight", (h, 3, config.patch_size, config.patch_size))
    weights["patch.weight"] = np.ascontiguousarray(patch.reshape(h, config.patch_width).T)
    weights["patch.bias"] = dense(f"{tower}.embeddings.patch_embedding.bias", (h,))
    weights["position"] = dense(
        f"{tower}.embeddings.position_embedding.weight", (config.positions_per_side ** 2, h)
    )
    for index in range(config.num_hidden_layers):
        source = f"{tower}.encoder.layers.{index}"
        target = f"layer.{index}"
        norm(f"{source}.layer_norm1", f"{target}.norm1", h)
        norm(f"{source}.layer_norm2", f"{target}.norm2", h)
        for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
            linear(f"{source}.self_attn.{projection}", f"{target}.{projection}", h, h)
        linear(f"{source}.mlp.fc1", f"{target}.fc1", config.intermediate_size, h)
        linear(f"{source}.mlp.fc2", f"{target}.fc2", h, config.intermediate_size)
    merger = f"{tower}.vit_merger"
    norm(f"{merger}.layer_norm1", "window.norm1", h)
    for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
        linear(f"{merger}.self_attn.{projection}", f"window.{projection}", h, h)
    norm(f"{merger}.pre_norm", "window.pre_norm", window)
    linear(f"{merger}.linear_1", "window.linear_1", 4 * config.intermediate_size, window)
    linear(f"{merger}.linear_2", "window.linear_2", h, 4 * config.intermediate_size)
    norm(f"{tower}.post_layernorm", "post_norm", h)
    projector = "model.merger.mlp.0"
    norm(f"{projector}.pre_norm", "projector.pre_norm", window)
    linear(f"{projector}.linear_1", "projector.linear_1", window, window)
    linear(f"{projector}.linear_2", "projector.linear_2", text_hidden, window)
    return weights


def build_vision_engine(config: VisionConfig, weights: dict[str, np.ndarray], *, precision: str,
                        verbose: bool = False) -> bytes:
    """Build the per-slice vision engine; all learned math stays in TensorRT."""
    if precision == "fp16":
        work, np_work = trt.float16, np.float16
    elif precision == "fp32":
        work, np_work = trt.float32, np.float32
    else:
        raise ValueError(f"Unsupported MiniCPM-V 4.6 vision precision {precision!r}")
    h = config.hidden_size
    heads = config.num_attention_heads
    head_dim = config.head_dim
    text_hidden = weights["projector.linear_2.bias"].shape[0]

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    build_config = builder.create_builder_config()
    build_config.builder_optimization_level = 3
    # FP32 math stays FP32 (attention scores, FP32 builds), as in the reference.
    build_config.clear_flag(trt.BuilderFlag.TF32)

    patches = network.add_input("patches", trt.float32, (-1, config.patch_width))
    position_ids = network.add_input("position_ids", trt.int32, (-1,))
    window_index = network.add_input("window_index", trt.int32, (-1,))
    merge_index = network.add_input("merge_index", trt.int32, (-1,))
    profile = builder.create_optimization_profile()
    profile.set_shape("patches", (MIN_PATCHES, config.patch_width), (OPT_PATCHES, config.patch_width),
                      (MAX_PATCHES, config.patch_width))
    for name in ("position_ids", "window_index"):
        profile.set_shape(name, (MIN_PATCHES,), (OPT_PATCHES,), (MAX_PATCHES,))
    profile.set_shape("merge_index", (MIN_PATCHES // 4,), (OPT_PATCHES // 4,), (MAX_PATCHES // 4,))
    build_config.add_optimization_profile(profile)

    def cast(x, dtype):
        return x if x.dtype == dtype else network.add_cast(x, dtype).get_output(0)

    def constant(values, dtype):
        array = narrowed(values, dtype, "vision weights")
        return network.add_constant(array.shape, trt.Weights(array)).get_output(0)

    def reshape(x, shape):
        layer = network.add_shuffle(x)
        layer.reshape_dims = shape
        return layer.get_output(0)

    def transpose(x, order):
        layer = network.add_shuffle(x)
        layer.first_transpose = trt.Permutation(order)
        return layer.get_output(0)

    def binary(a, b, operation):
        return network.add_elementwise(a, b, operation).get_output(0)

    def linear(x, name):
        weight = weights[name + ".weight"]
        rank = len(tuple(x.shape))
        rhs = constant(weight.reshape((1,) * (rank - 2) + weight.shape), np_work)
        y = network.add_matrix_multiply(x, trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.NONE).get_output(0)
        bias = weights[name + ".bias"]
        return binary(y, constant(bias.reshape((1,) * (rank - 1) + bias.shape), np_work),
                      trt.ElementWiseOperation.SUM)

    def layer_norm(x, name, eps):
        # The reference normalizes in FP32 and rounds the result to the model dtype.
        width = weights[name + ".weight"].shape[0]
        rank = len(tuple(x.shape))
        shape = (1,) * (rank - 1) + (width,)
        layer = network.add_normalization_v2(
            cast(x, trt.float32), constant(weights[name + ".weight"].reshape(shape), np.float32),
            constant(weights[name + ".bias"].reshape(shape), np.float32), 1 << (rank - 1))
        layer.epsilon = eps
        return cast(layer.get_output(0), work)

    def gelu(x, approximate):
        kind = trt.ActivationType.GELU_TANH if approximate else trt.ActivationType.GELU_ERF
        return cast(network.add_activation(cast(x, trt.float32), kind).get_output(0), work)

    def attention(x, prefix, groups, group_size):
        """Multi-head attention inside ``groups`` independent sets of ``group_size`` rows."""
        query = reshape(linear(x, prefix + ".q_proj"), (groups, group_size, heads, head_dim))
        key = reshape(linear(x, prefix + ".k_proj"), (groups, group_size, heads, head_dim))
        value = reshape(linear(x, prefix + ".v_proj"), (groups, group_size, heads, head_dim))
        query, key, value = (transpose(t, (0, 2, 1, 3)) for t in (query, key, value))
        scores = network.add_matrix_multiply(cast(query, trt.float32), trt.MatrixOperation.NONE,
                                             cast(key, trt.float32), trt.MatrixOperation.TRANSPOSE).get_output(0)
        scores = binary(scores, constant(np.full((1, 1, 1, 1), head_dim ** -0.5, np.float32), np.float32),
                        trt.ElementWiseOperation.PROD)
        softmax = network.add_softmax(scores)
        softmax.axes = 1 << 3
        probabilities = cast(softmax.get_output(0), work)
        context = network.add_matrix_multiply(probabilities, trt.MatrixOperation.NONE,
                                              value, trt.MatrixOperation.NONE).get_output(0)
        context = reshape(transpose(context, (0, 2, 1, 3)), (-1, h))
        return linear(context, prefix + ".out_proj")

    def encoder_layer(x, index):
        prefix = f"layer.{index}"
        residual = x
        x = attention(layer_norm(x, prefix + ".norm1", config.layer_norm_eps), prefix, 1, -1)
        x = binary(residual, x, trt.ElementWiseOperation.SUM)
        residual = x
        x = layer_norm(x, prefix + ".norm2", config.layer_norm_eps)
        x = linear(gelu(linear(x, prefix + ".fc1"), True), prefix + ".fc2")
        return binary(residual, x, trt.ElementWiseOperation.SUM)

    # 1. Patch embedding and learned positions (pixel values rounded to the model dtype first).
    hidden = linear(cast(patches, work), "patch")
    positions = network.add_gather(constant(weights["position"], np_work), position_ids, 0).get_output(0)
    hidden = binary(hidden, positions, trt.ElementWiseOperation.SUM)

    # 2. Full-resolution encoder layers.
    for index in range(config.insert_layer_id + 1):
        hidden = encoder_layer(hidden, index)

    # 3. Windowed merger: attention inside each 2 x 2 window, then one token per window.
    normalized = network.add_gather(layer_norm(hidden, "window.norm1", config.layer_norm_eps),
                                    window_index, 0).get_output(0)
    residual = network.add_gather(hidden, window_index, 0).get_output(0)
    hidden = binary(residual, attention(normalized, "window", -1, 4), trt.ElementWiseOperation.SUM)
    windows = reshape(hidden, (-1, 4, h))
    window_mean = reshape(cast(network.add_reduce(cast(windows, trt.float32), trt.ReduceOperation.AVG,
                                                  1 << 1, keep_dims=True).get_output(0), work), (-1, h))
    flat = reshape(windows, (-1, 4 * h))
    merged = linear(gelu(linear(layer_norm(flat, "window.pre_norm", config.layer_norm_eps), "window.linear_1"),
                         True), "window.linear_2")
    hidden = binary(merged, window_mean, trt.ElementWiseOperation.SUM)

    # 4. Remaining encoder layers over the merged tokens and the post layer norm.
    for index in range(config.insert_layer_id + 1, config.num_hidden_layers):
        hidden = encoder_layer(hidden, index)
    hidden = layer_norm(hidden, "post_norm", config.layer_norm_eps)

    # 5. Projector: 2 x 2 blocks of merged tokens to the language model width.
    blocks = reshape(network.add_gather(hidden, merge_index, 0).get_output(0), (-1, 4 * h))
    projected = linear(gelu(linear(layer_norm(blocks, "projector.pre_norm", PROJECTOR_EPS),
                                   "projector.linear_1"), False), "projector.linear_2")
    projected = reshape(projected, (-1, text_hidden))
    projected.name = "visual_features"
    network.mark_output(projected)

    plan = builder.build_serialized_network(network, build_config)
    if plan is None:
        raise RuntimeError("MiniCPM-V 4.6 vision engine build failed")
    return bytes(plan)
