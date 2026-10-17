# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned Qwen3-VL visual conditioner for original H3 reference inputs.

One call encodes one spatial attention segment (an image or one temporal patch
of a video). Upstream attention does not cross these segment boundaries; the
runtime must concatenate outputs in presentation order, never attend over all
reference images jointly. Inputs use the processor's merge-block patch order.
Interpolation indices/weights and full FP32 rotary tables are separate inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import math

import ml_dtypes
import numpy as np
import tensorrt as trt

from . import graph_ops as op


@dataclass(frozen=True)
class VisionConfig:
    depth: int = 27
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 5120
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple[int, ...] = (8, 16, 24)

    @property
    def head_dim(self):
        return self.hidden_size // self.num_heads

    @property
    def patch_dim(self):
        return self.in_channels * self.temporal_patch_size * self.patch_size**2

    @property
    def merge_unit(self):
        return self.spatial_merge_size**2

    def validate(self):
        values = (
            self.depth,
            self.hidden_size,
            self.intermediate_size,
            self.num_heads,
            self.in_channels,
            self.patch_size,
            self.temporal_patch_size,
            self.spatial_merge_size,
            self.out_hidden_size,
            self.num_position_embeddings,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values
        ):
            raise ValueError("H3 vision dimensions must be positive integers")
        if self.hidden_size % self.num_heads or self.head_dim % 4:
            raise ValueError(
                "H3 vision heads must divide hidden size with a four-aligned rotary dimension"
            )
        if math.isqrt(self.num_position_embeddings) ** 2 != self.num_position_embeddings:
            raise ValueError("H3 vision learned position embeddings must form a square grid")
        indexes = self.deepstack_visual_indexes
        if (
            len(indexes) != 3
            or any(
                isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < self.depth
                for value in indexes
            )
            or tuple(sorted(set(indexes))) != tuple(indexes)
        ):
            raise ValueError("H3 vision requires three ordered, distinct DeepStack layer indexes")


def checkpoint_shapes(config: VisionConfig = VisionConfig()):
    config.validate()
    shapes = {}
    root = "model.visual."
    width, merged = config.hidden_size, config.hidden_size * config.merge_unit

    def affine(prefix, channels):
        shapes[root + prefix + ".weight"] = (channels,)
        shapes[root + prefix + ".bias"] = (channels,)

    def linear(prefix, inputs, outputs):
        shapes[root + prefix + ".weight"] = (outputs, inputs)
        shapes[root + prefix + ".bias"] = (outputs,)

    shapes[root + "patch_embed.proj.weight"] = (
        width,
        config.in_channels,
        config.temporal_patch_size,
        config.patch_size,
        config.patch_size,
    )
    shapes[root + "patch_embed.proj.bias"] = (width,)
    shapes[root + "pos_embed.weight"] = (config.num_position_embeddings, width)
    for index in range(config.depth):
        prefix = f"blocks.{index}"
        affine(prefix + ".norm1", width)
        affine(prefix + ".norm2", width)
        linear(prefix + ".attn.qkv", width, 3 * width)
        linear(prefix + ".attn.proj", width, width)
        linear(prefix + ".mlp.linear_fc1", width, config.intermediate_size)
        linear(prefix + ".mlp.linear_fc2", config.intermediate_size, width)
    for index, prefix in enumerate(("merger", *(f"deepstack_merger_list.{i}" for i in range(3)))):
        affine(prefix + ".norm", width if index == 0 else merged)
        linear(prefix + ".linear_fc1", merged, merged)
        linear(prefix + ".linear_fc2", merged, config.out_hidden_size)
    return shapes


def checkpoint_keys(config: VisionConfig = VisionConfig()):
    return tuple(checkpoint_shapes(config))


def _validate_weights(weights, config, precision):
    if precision not in ("bf16", "fp32"):
        raise ValueError(
            "H3 vision supports original BF16 or explicit FP32 component validation only"
        )
    expected = checkpoint_shapes(config)
    if set(weights) != set(expected):
        raise ValueError("H3 vision requires exactly its visual-tower checkpoint tensors")
    dtype = np.dtype(ml_dtypes.bfloat16 if precision == "bf16" else np.float32)
    for name, shape in expected.items():
        array = np.asarray(weights[name])
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(
                f"H3 vision tensor {name} requires {dtype} shape {shape}, got {array.dtype} {array.shape}"
            )


def _linear(network, value, weights, prefix):
    return op.linear(
        network,
        value,
        weights[prefix + ".weight"],
        weights[prefix + ".bias"],
        bf16=False,
        compute_dtype=value.dtype,
    )


def _layer_norm(network, value, weights, prefix):
    width = int(value.shape[-1])
    gamma = op.cast(
        network,
        op.weight_constant(network, weights[prefix + ".weight"].reshape(1, width)),
        trt.float32,
    )
    beta = op.cast(
        network,
        op.weight_constant(network, weights[prefix + ".bias"].reshape(1, width)),
        trt.float32,
    )
    layer = network.add_normalization_v2(op.cast(network, value, trt.float32), gamma, beta, 1 << 1)
    if layer is None:
        raise RuntimeError(f"TensorRT rejected H3 vision LayerNorm {prefix}")
    layer.epsilon = 1e-6
    layer.name = prefix
    return op.cast(network, layer.get_output(0), value.dtype)


def _gelu(network, value, *, approximate):
    kind = trt.ActivationType.GELU_TANH if approximate else trt.ActivationType.GELU_ERF
    # PyTorch's low-precision GELU evaluates its arithmetic in FP32 and rounds
    # once. The merger uses exact GELU; the vision MLP uses tanh GELU.
    activated = network.add_activation(op.cast(network, value, trt.float32), kind)
    return op.cast(network, activated.get_output(0), value.dtype)


def _learned_position(network, indices, interpolation_weights, table, dtype):
    gathered = network.add_gather(op.weight_constant(network, table), indices, 0).get_output(0)
    scale = network.add_shuffle(interpolation_weights)
    scale.reshape_dims = (-1, 4, 1)
    # Transformers 5.15 promotes BF16 embeddings * FP32 interpolation weights
    # to FP32, sums the four taps, and casts once before the residual addition.
    terms = network.add_elementwise(
        op.cast(network, gathered, trt.float32), scale.get_output(0), trt.ElementWiseOperation.PROD
    ).get_output(0)
    summed = network.add_reduce(terms, trt.ReduceOperation.SUM, 1 << 1, False).get_output(0)
    return op.cast(network, summed, dtype)


def _rotary(network, value, cos, sin, config):
    shaped = network.add_shuffle(op.cast(network, value, trt.float32))
    shaped.reshape_dims = (-1, config.num_heads, config.head_dim)
    half = config.head_dim // 2
    first = op.dynamic_slice(
        network, shaped.get_output(0), (0, 0, 0), (None, config.num_heads, half)
    )
    second = op.dynamic_slice(
        network, shaped.get_output(0), (0, 0, half), (None, config.num_heads, half)
    )
    negative = network.add_unary(second, trt.UnaryOperation.NEG).get_output(0)
    rotate = network.add_concatenation((negative, first))
    rotate.axis = 2

    def multiply(tensor, table):
        broadcast = network.add_shuffle(table)
        broadcast.reshape_dims = (-1, 1, config.head_dim)
        return network.add_elementwise(
            tensor, broadcast.get_output(0), trt.ElementWiseOperation.PROD
        ).get_output(0)

    result = network.add_elementwise(
        multiply(shaped.get_output(0), cos),
        multiply(rotate.get_output(0), sin),
        trt.ElementWiseOperation.SUM,
    ).get_output(0)
    flatten = network.add_shuffle(result)
    flatten.reshape_dims = (-1, config.hidden_size)
    return op.cast(network, flatten.get_output(0), value.dtype)


def _attention(network, value, cos, sin, weights, prefix, config):
    packed = _linear(network, value, weights, prefix + ".qkv")
    q, k, v = (
        op.dynamic_slice(
            network, packed, (0, index * config.hidden_size), (None, config.hidden_size)
        )
        for index in range(3)
    )
    q = _rotary(network, q, cos, sin, config)
    k = _rotary(network, k, cos, sin, config)
    scale = op.constant(network, np.asarray([[config.head_dim**-0.5]], np.float32))
    scaled = network.add_elementwise(
        op.cast(network, q, trt.float32), scale, trt.ElementWiseOperation.PROD
    ).get_output(0)
    q = op.cast(network, scaled, q.dtype)
    operands = [
        op.rows_to_heads(network, tensor, -1, config.num_heads, config.head_dim)
        for tensor in (q, k, v)
    ]
    layer = network.add_attention(*operands, trt.AttentionNormalizationOp.SOFTMAX, False)
    if layer is None:
        raise RuntimeError(f"TensorRT rejected H3 vision attention {prefix}")
    layer.name = prefix + ".native_attention"
    layer.metadata = f"trtmc.native_op=IAttention;source={layer.name}"
    # The FP32 option is for component validation; production BF16 retains a
    # fused attention contract. TensorRT, not family kernels, owns lowering.
    layer.decomposable = value.dtype == trt.float32
    output = op.heads_to_rows(network, layer.get_output(0), -1, config.hidden_size)
    return _linear(network, output, weights, prefix + ".proj")


def _merger(network, value, weights, prefix, config, *, postshuffle):
    merged_width = config.hidden_size * config.merge_unit
    if not postshuffle:
        value = _layer_norm(network, value, weights, prefix + ".norm")
    merge = network.add_shuffle(value)
    merge.reshape_dims = (-1, merged_width)
    value = merge.get_output(0)
    if postshuffle:
        value = _layer_norm(network, value, weights, prefix + ".norm")
    value = _linear(network, value, weights, prefix + ".linear_fc1")
    value = _gelu(network, value, approximate=False)
    return _linear(network, value, weights, prefix + ".linear_fc2")


def build_vision_encoder_engine(
    weights: dict,
    *,
    config: VisionConfig = VisionConfig(),
    patch_rows: tuple[int, int, int] = (4, 4096, 65536),
    precision: str = "bf16",
    workspace_bytes: int | None = 8 << 30,
    consume_weights: bool = False,
    verbose: bool = False,
) -> bytes:
    config.validate()
    if (
        len(patch_rows) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < config.merge_unit
            or value % config.merge_unit
            for value in patch_rows
        )
        or not patch_rows[0] <= patch_rows[1] <= patch_rows[2]
    ):
        raise ValueError("H3 vision profile requires ordered, merge-aligned patch row bounds")
    _validate_weights(weights, config, precision)
    dtype = trt.bfloat16 if precision == "bf16" else trt.float32
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    options = builder.create_builder_config()
    options.builder_optimization_level = 1
    op.configure_builder(options)
    op.configure_workspace(options, workspace_bytes, default_bytes=8 << 30)
    options.clear_flag(trt.BuilderFlag.TF32)
    shapes = {
        "pixel_values": (config.patch_dim,),
        "position_indices": (4,),
        "position_weights": (4,),
        "rotary_cos": (config.head_dim,),
        "rotary_sin": (config.head_dim,),
    }
    inputs, profile = {}, builder.create_optimization_profile()
    for name, tail in shapes.items():
        inputs[name] = network.add_input(
            name, trt.int32 if name == "position_indices" else trt.float32, (-1, *tail)
        )
        inputs[name].set_dimension_name(0, "vision_patch_rows")
        profile.set_shape(
            name, min=(patch_rows[0], *tail), opt=(patch_rows[1], *tail), max=(patch_rows[2], *tail)
        )
    options.add_optimization_profile(profile)
    try:
        # Each Conv3d input is exactly one [C,T,P,P] patch and the convolution
        # has matching kernel/stride. Its one-position dot product is a native
        # linear map over the unchanged checkpoint coefficient order.
        patch = "model.visual.patch_embed.proj"
        value = op.linear(
            network,
            op.cast(network, inputs["pixel_values"], dtype),
            weights[patch + ".weight"].reshape(config.hidden_size, config.patch_dim),
            weights[patch + ".bias"],
            bf16=False,
            compute_dtype=dtype,
        )
        position = _learned_position(
            network,
            inputs["position_indices"],
            inputs["position_weights"],
            weights["model.visual.pos_embed.weight"],
            dtype,
        )
        value = network.add_elementwise(value, position, trt.ElementWiseOperation.SUM).get_output(0)
        for index in range(config.depth):
            prefix = f"model.visual.blocks.{index}"
            normalized = _layer_norm(network, value, weights, prefix + ".norm1")
            update = _attention(
                network,
                normalized,
                inputs["rotary_cos"],
                inputs["rotary_sin"],
                weights,
                prefix + ".attn",
                config,
            )
            value = network.add_elementwise(value, update, trt.ElementWiseOperation.SUM).get_output(
                0
            )
            normalized = _layer_norm(network, value, weights, prefix + ".norm2")
            update = _linear(network, normalized, weights, prefix + ".mlp.linear_fc1")
            update = _gelu(network, update, approximate=True)
            update = _linear(network, update, weights, prefix + ".mlp.linear_fc2")
            value = network.add_elementwise(value, update, trt.ElementWiseOperation.SUM).get_output(
                0
            )
            if index in config.deepstack_visual_indexes:
                branch = config.deepstack_visual_indexes.index(index)
                feature = _merger(
                    network,
                    value,
                    weights,
                    f"model.visual.deepstack_merger_list.{branch}",
                    config,
                    postshuffle=True,
                )
                feature.name = f"deepstack_features_{branch}"
                network.mark_output(feature)
        feature = _merger(network, value, weights, "model.visual.merger", config, postshuffle=False)
        feature.name = "visual_features"
        network.mark_output(feature)
        op.validate_native_network(
            network, expected_attentions=config.depth, label="reference vision encoder"
        )
        plan = builder.build_serialized_network(network, options)
    finally:
        op.release_weight_buffers(network)
        if consume_weights:
            weights.clear()
    if plan is None:
        raise RuntimeError("TensorRT failed to build H3 reference vision encoder")
    result = bytes(plan)
    del network, options, builder, logger
    gc.collect()
    return result
