# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct TensorRT construction of SAM3 recurrent tracker conditioning.

The C++ tracker session owns temporal policy and supplies bounded spatial
memory and object-pointer history.  This module reconstructs the learned
four-layer memory-attention stack with TensorRT Network API layers.  It emits
the conditioned 72x72 feature consumed by the tracker mask decoder.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from tensorrt_model_connect import trt_compat

from .graph_ops import (
    add_attention_core,
    add_bias_sum,
    add_constant,
    add_layer_norm_native,
    add_matmul_rhs_constant,
)


trt = trt_compat.get_trt()

_CHECKPOINT_PREFIX = "tracker_model"
_MEMORY_ATTENTION_PREFIX = f"{_CHECKPOINT_PREFIX}.memory_attention"
_TEMPORAL_MEMORY_KEY = f"{_CHECKPOINT_PREFIX}.memory_temporal_positional_encoding"
_POINTER_PROJECTION_PREFIX = f"{_CHECKPOINT_PREFIX}.temporal_positional_encoding_projection_layer"

_BATCH_SIZES = (1, 2)
_HEIGHT = 72
_WIDTH = 72
_SPATIAL_TOKENS = _HEIGHT * _WIDTH
_HIDDEN_SIZE = 256
_MEMORY_SIZE = 64
_POINTER_TOKENS = _HIDDEN_SIZE // _MEMORY_SIZE
_TEMPORAL_MEMORY_SLOTS = 7
_NUM_LAYERS = 4
_NUM_HEADS = 1
_HEAD_DIM = _HIDDEN_SIZE // _NUM_HEADS
_FEED_FORWARD_SIZE = 2048
_LAYER_NORM_EPSILON = 1e-5
_ROPE_THETA = 10000.0
_POSITION_SCALE = 0.1


@dataclass(frozen=True)
class _PreparedMemory:
    """Batch-first recurrent values and their positional encodings."""

    spatial_values: trt.ITensor
    spatial_position: trt.ITensor
    pointer_values: trt.ITensor
    pointer_position: trt.ITensor


@dataclass(frozen=True)
class _RopeConstants:
    """Shared constants for SAM3's adjacent-pair axial rotation."""

    cosine: trt.ITensor
    sine: trt.ITensor
    rotated_indices: trt.ITensor
    rotated_sign: trt.ITensor


def _weight(weights: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    try:
        value = weights[key]
    except KeyError as error:
        raise KeyError(f"Missing SAM3 tracker attention weight: {key}") from error
    return np.ascontiguousarray(np.asarray(value), dtype=np.float32)


def _cast(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    dtype: trt.DataType,
) -> trt.ITensor:
    if inp.dtype == dtype:
        return inp
    return network.add_cast(inp, dtype).get_output(0)


def _fp32_sum(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    rhs: trt.ITensor,
) -> trt.ITensor:
    """Add position or residual tensors at Meta's FP32 boundary."""

    lhs = _cast(network, lhs, trt.float32)
    rhs = _cast(network, rhs, trt.float32)
    return network.add_elementwise(lhs, rhs, trt.ElementWiseOperation.SUM).get_output(0)


def _bf16_sum(
    network: trt.INetworkDefinition,
    lhs: trt.ITensor,
    rhs: trt.ITensor,
) -> trt.ITensor:
    """Round Meta's recurrent residual stream at its BF16 boundary."""

    lhs = _cast(network, lhs, trt.bfloat16)
    rhs = _cast(network, rhs, trt.bfloat16)
    return network.add_elementwise(lhs, rhs, trt.ElementWiseOperation.SUM).get_output(0)


def _bf16_linear(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    in_size: int,
    out_size: int,
) -> trt.ITensor:
    raw_weight = _weight(weights, f"{prefix}.weight")
    raw_bias = _weight(weights, f"{prefix}.bias")
    if raw_weight.shape != (out_size, in_size):
        raise ValueError(
            f"SAM3 tracker projection {prefix!r} must have checkpoint shape "
            f"{(out_size, in_size)}, got {raw_weight.shape}"
        )
    if raw_bias.shape != (out_size,):
        raise ValueError(
            f"SAM3 tracker projection {prefix!r} bias must have shape "
            f"{(out_size,)}, got {raw_bias.shape}"
        )
    inp = _cast(network, inp, trt.bfloat16)
    output = add_matmul_rhs_constant(
        network,
        inp,
        in_size,
        out_size,
        np.ascontiguousarray(raw_weight.T),
        dtype=np.float32,
    )
    return add_bias_sum(
        network,
        output,
        out_size,
        raw_bias,
        dtype=np.float32,
    )


def _layer_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: Mapping[str, np.ndarray],
    prefix: str,
) -> trt.ITensor:
    gamma = _weight(weights, f"{prefix}.weight")
    beta = _weight(weights, f"{prefix}.bias")
    if gamma.shape != (_HIDDEN_SIZE,) or beta.shape != (_HIDDEN_SIZE,):
        raise ValueError(
            f"SAM3 tracker normalization {prefix!r} must have {_HIDDEN_SIZE} "
            f"channels; got gamma={gamma.shape}, beta={beta.shape}"
        )
    inp = _cast(network, inp, trt.float32)
    return add_layer_norm_native(
        network,
        inp,
        _HIDDEN_SIZE,
        gamma,
        beta,
        _LAYER_NORM_EPSILON,
        dtype=np.float32,
    )


def _shuffle(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    shape: tuple[int, ...],
) -> trt.ITensor:
    layer = network.add_shuffle(inp)
    layer.reshape_dims = shape
    return layer.get_output(0)


def _transpose(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    order: tuple[int, ...],
) -> trt.ITensor:
    layer = network.add_shuffle(inp)
    layer.first_transpose = order
    return layer.get_output(0)


def _expand_batch(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    batch_size: int,
) -> trt.ITensor:
    input_batch = int(inp.shape[0])
    if input_batch == batch_size:
        return inp
    if input_batch != 1 or batch_size != 2:
        raise ValueError(
            f"Cannot broadcast SAM3 tracker tensor batch {input_batch} to {batch_size}"
        )
    layer = network.add_concatenation([inp, inp])
    layer.axis = 0
    return layer.get_output(0)


def _validate_inputs(
    current_features: trt.ITensor,
    current_position: trt.ITensor,
    memory_features: trt.ITensor,
    memory_position: trt.ITensor,
    memory_temporal_offsets: trt.ITensor,
    object_pointers: trt.ITensor,
    object_pointer_temporal_offsets: trt.ITensor,
    max_object_pointers_to_use: trt.ITensor,
    batch_size: int,
) -> None:
    if batch_size not in _BATCH_SIZES:
        raise ValueError(f"SAM3 recurrent conditioning supports batch 1 or 2, got {batch_size}")

    expected_feature_tail = (_HIDDEN_SIZE, _HEIGHT, _WIDTH)
    for name, tensor in (
        ("current_features", current_features),
        ("current_position", current_position),
    ):
        shape = tuple(int(dim) for dim in tensor.shape)
        if len(shape) != 4 or shape[1:] != expected_feature_tail:
            raise ValueError(f"SAM3 tracker {name} must have shape [B, 256, 72, 72], got {shape}")
        if shape[0] not in (1, batch_size):
            raise ValueError(f"SAM3 tracker {name} batch must be 1 or {batch_size}, got {shape[0]}")

    if batch_size == 1:
        expected = {
            "memory_features": (1, -1, _SPATIAL_TOKENS, _MEMORY_SIZE),
            "memory_position": (1, -1, _SPATIAL_TOKENS, _MEMORY_SIZE),
            "memory_temporal_offsets": (1, -1),
            "object_pointers": (1, -1, _HIDDEN_SIZE),
            "object_pointer_temporal_offsets": (1, -1),
        }
    else:
        expected = {
            "memory_features": (2, -1, _SPATIAL_TOKENS, _MEMORY_SIZE),
            "memory_position": (2, -1, _SPATIAL_TOKENS, _MEMORY_SIZE),
            "memory_temporal_offsets": (2, -1),
            "object_pointers": (2, -1, _HIDDEN_SIZE),
            "object_pointer_temporal_offsets": (2, -1),
        }
    tensors = {
        "memory_features": memory_features,
        "memory_position": memory_position,
        "memory_temporal_offsets": memory_temporal_offsets,
        "object_pointers": object_pointers,
        "object_pointer_temporal_offsets": object_pointer_temporal_offsets,
    }
    for name, expected_shape in expected.items():
        shape = tuple(int(dim) for dim in tensors[name].shape)
        if shape != expected_shape:
            raise ValueError(f"SAM3 tracker {name} must have shape {expected_shape}, got {shape}")

    max_pointer_shape = tuple(int(dim) for dim in max_object_pointers_to_use.shape)
    if max_pointer_shape != (1,):
        raise ValueError(
            f"SAM3 max_object_pointers_to_use must have shape (1,), got {max_pointer_shape}"
        )


def _modulo_temporal_indices(
    network: trt.INetworkDefinition,
    temporal_offsets: trt.ITensor,
) -> trt.ITensor:
    rank = len(tuple(temporal_offsets.shape))
    scalar_shape = (1,) * rank
    one = add_constant(
        network,
        scalar_shape,
        np.ones(scalar_shape, dtype=np.int32),
        dtype=np.int32,
    )
    slots = add_constant(
        network,
        scalar_shape,
        np.full(scalar_shape, _TEMPORAL_MEMORY_SLOTS, dtype=np.int32),
        dtype=np.int32,
    )
    shifted = network.add_elementwise(
        temporal_offsets, one, trt.ElementWiseOperation.SUB
    ).get_output(0)
    quotient = network.add_elementwise(
        shifted, slots, trt.ElementWiseOperation.FLOOR_DIV
    ).get_output(0)
    multiple = network.add_elementwise(quotient, slots, trt.ElementWiseOperation.PROD).get_output(0)
    return network.add_elementwise(shifted, multiple, trt.ElementWiseOperation.SUB).get_output(0)


def _prepare_spatial_memory(
    network: trt.INetworkDefinition,
    memory_features: trt.ITensor,
    memory_position: trt.ITensor,
    memory_temporal_offsets: trt.ITensor,
    weights: Mapping[str, np.ndarray],
    batch_size: int,
) -> tuple[trt.ITensor, trt.ITensor]:
    temporal = _weight(weights, _TEMPORAL_MEMORY_KEY)
    if temporal.shape != (_TEMPORAL_MEMORY_SLOTS, 1, 1, _MEMORY_SIZE):
        raise ValueError(
            "SAM3 temporal memory encoding must have shape "
            f"{(_TEMPORAL_MEMORY_SLOTS, 1, 1, _MEMORY_SIZE)}, got {temporal.shape}"
        )
    temporal_table = add_constant(
        network,
        (_TEMPORAL_MEMORY_SLOTS, 1, _MEMORY_SIZE),
        temporal.reshape(_TEMPORAL_MEMORY_SLOTS, 1, _MEMORY_SIZE),
        dtype=np.float32,
    )
    temporal_indices = _modulo_temporal_indices(network, memory_temporal_offsets)
    temporal_position = network.add_gather(temporal_table, temporal_indices, axis=0).get_output(0)

    position_by_frame = memory_position
    spatial_values = _shuffle(network, memory_features, (batch_size, -1, _MEMORY_SIZE))

    position_by_frame = _fp32_sum(
        network,
        position_by_frame,
        temporal_position,
    )
    spatial_position = _shuffle(network, position_by_frame, (batch_size, -1, _MEMORY_SIZE))
    return spatial_values, spatial_position


def _prepare_pointer_memory(
    network: trt.INetworkDefinition,
    object_pointers: trt.ITensor,
    object_pointer_temporal_offsets: trt.ITensor,
    max_object_pointers_to_use: trt.ITensor,
    weights: Mapping[str, np.ndarray],
    batch_size: int,
) -> tuple[trt.ITensor, trt.ITensor]:
    offsets = network.add_cast(object_pointer_temporal_offsets, trt.float32).get_output(0)
    offsets = _shuffle(network, offsets, (batch_size, -1, 1))
    max_pointers = network.add_cast(max_object_pointers_to_use, trt.float32).get_output(0)
    max_pointers = _shuffle(network, max_pointers, (1, 1, 1))
    one = add_constant(
        network,
        (1, 1, 1),
        np.ones((1, 1, 1), dtype=np.float32),
        dtype=np.float32,
    )
    denominator = network.add_elementwise(
        max_pointers, one, trt.ElementWiseOperation.SUB
    ).get_output(0)
    normalized_offsets = network.add_elementwise(
        offsets, denominator, trt.ElementWiseOperation.DIV
    ).get_output(0)

    sine_width = _HIDDEN_SIZE // 2
    frequency_indices = np.arange(sine_width, dtype=np.int32)
    frequencies = np.power(
        np.float32(10000.0),
        2.0 * (frequency_indices // 2).astype(np.float32) / float(sine_width),
    ).astype(np.float32)
    frequency = add_constant(
        network,
        (1, 1, sine_width),
        frequencies.reshape(1, 1, sine_width),
        dtype=np.float32,
    )
    angles = network.add_elementwise(
        normalized_offsets, frequency, trt.ElementWiseOperation.DIV
    ).get_output(0)
    sine = network.add_unary(angles, trt.UnaryOperation.SIN).get_output(0)
    cosine = network.add_unary(angles, trt.UnaryOperation.COS).get_output(0)
    position = network.add_concatenation([sine, cosine])
    position.axis = 2
    projected_position = _bf16_linear(
        network,
        position.get_output(0),
        weights,
        _POINTER_PROJECTION_PREFIX,
        _HIDDEN_SIZE,
        _MEMORY_SIZE,
    )
    pointer_values = _shuffle(
        network,
        object_pointers,
        (batch_size, -1, _POINTER_TOKENS, _MEMORY_SIZE),
    )
    pointer_values = _shuffle(network, pointer_values, (batch_size, -1, _MEMORY_SIZE))

    expanded_position = _shuffle(network, projected_position, (batch_size, -1, 1, _MEMORY_SIZE))
    repeated_position = network.add_concatenation([expanded_position] * _POINTER_TOKENS)
    repeated_position.axis = 2
    pointer_position = _shuffle(
        network,
        repeated_position.get_output(0),
        (batch_size, -1, _MEMORY_SIZE),
    )
    return pointer_values, pointer_position


def _prepare_memory(
    network: trt.INetworkDefinition,
    memory_features: trt.ITensor,
    memory_position: trt.ITensor,
    memory_temporal_offsets: trt.ITensor,
    object_pointers: trt.ITensor,
    object_pointer_temporal_offsets: trt.ITensor,
    max_object_pointers_to_use: trt.ITensor,
    weights: Mapping[str, np.ndarray],
    batch_size: int,
) -> _PreparedMemory:
    spatial_values, spatial_position = _prepare_spatial_memory(
        network,
        memory_features,
        memory_position,
        memory_temporal_offsets,
        weights,
        batch_size,
    )
    pointer_values, pointer_position = _prepare_pointer_memory(
        network,
        object_pointers,
        object_pointer_temporal_offsets,
        max_object_pointers_to_use,
        weights,
        batch_size,
    )
    return _PreparedMemory(
        spatial_values=spatial_values,
        spatial_position=spatial_position,
        pointer_values=pointer_values,
        pointer_position=pointer_position,
    )


@lru_cache(maxsize=1)
def _axial_rope_arrays() -> tuple[np.ndarray, np.ndarray]:
    frequencies = np.power(
        np.float32(_ROPE_THETA),
        -np.arange(0, _HEAD_DIM, 4, dtype=np.float32) / float(_HEAD_DIM),
    )
    positions = np.arange(_SPATIAL_TOKENS, dtype=np.int32)
    x_positions = (positions % _WIDTH).astype(np.float32)
    y_positions = (positions // _WIDTH).astype(np.float32)
    x_angles = np.outer(x_positions, frequencies)
    y_angles = np.outer(y_positions, frequencies)
    angles = np.repeat(np.concatenate((x_angles, y_angles), axis=1), 2, axis=1)
    return (
        np.ascontiguousarray(np.cos(angles), dtype=np.float32),
        np.ascontiguousarray(np.sin(angles), dtype=np.float32),
    )


def _add_rope_constants(network: trt.INetworkDefinition) -> _RopeConstants:
    cosine, sine = _axial_rope_arrays()
    cosine_tensor = add_constant(
        network,
        (1, 1, _SPATIAL_TOKENS, _HEAD_DIM),
        cosine.reshape(1, 1, _SPATIAL_TOKENS, _HEAD_DIM),
        dtype=np.float32,
    )
    sine_tensor = add_constant(
        network,
        (1, 1, _SPATIAL_TOKENS, _HEAD_DIM),
        sine.reshape(1, 1, _SPATIAL_TOKENS, _HEAD_DIM),
        dtype=np.float32,
    )
    indices = np.arange(_HEAD_DIM, dtype=np.int32).reshape(-1, 2)[:, ::-1].reshape(-1)
    rotated_indices = add_constant(network, (_HEAD_DIM,), indices, dtype=np.int32)
    signs = np.tile(np.array([-1.0, 1.0], dtype=np.float32), _HEAD_DIM // 2)
    rotated_sign = add_constant(
        network,
        (1, 1, 1, _HEAD_DIM),
        signs.reshape(1, 1, 1, _HEAD_DIM),
        dtype=np.float32,
    )
    return _RopeConstants(
        cosine=cosine_tensor,
        sine=sine_tensor,
        rotated_indices=rotated_indices,
        rotated_sign=rotated_sign,
    )


def _apply_axial_rope(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    constants: _RopeConstants,
) -> trt.ITensor:
    # TorchInductor promotes the BF16 Q/K projections for complex RoPE and
    # casts the rotated tensors back only at the FlashAttention boundary.
    inp = _cast(network, inp, trt.float32)
    rotated = network.add_gather(inp, constants.rotated_indices, axis=3).get_output(0)
    rotated = network.add_elementwise(
        rotated, constants.rotated_sign, trt.ElementWiseOperation.PROD
    ).get_output(0)
    cosine = network.add_elementwise(
        inp, constants.cosine, trt.ElementWiseOperation.PROD
    ).get_output(0)
    sine = network.add_elementwise(
        rotated, constants.sine, trt.ElementWiseOperation.PROD
    ).get_output(0)
    return network.add_elementwise(cosine, sine, trt.ElementWiseOperation.SUM).get_output(0)


def _attention_context(
    network: trt.INetworkDefinition,
    query: trt.ITensor,
    key: trt.ITensor,
    value: trt.ITensor,
    batch_size: int,
) -> trt.ITensor:
    # L4's TensorRT fused-attention tactic is faster and more accurate for
    # this head shape in FP16, while the surrounding GEMMs/residual stream
    # retain Meta's BF16 contract.
    query = _cast(network, query, trt.float16)
    key = _cast(network, key, trt.float16)
    value = _cast(network, value, trt.float16)
    context = add_attention_core(network, query, key, value)
    return _shuffle(network, context, (batch_size, _SPATIAL_TOKENS, _HIDDEN_SIZE))


def _self_attention(
    network: trt.INetworkDefinition,
    queries: trt.ITensor,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    rope: _RopeConstants,
    batch_size: int,
) -> trt.ITensor:
    query = _bf16_linear(network, queries, weights, f"{prefix}.q_proj", _HIDDEN_SIZE, _HIDDEN_SIZE)
    key = _bf16_linear(network, queries, weights, f"{prefix}.k_proj", _HIDDEN_SIZE, _HIDDEN_SIZE)
    value = _bf16_linear(network, queries, weights, f"{prefix}.v_proj", _HIDDEN_SIZE, _HIDDEN_SIZE)
    query = _shuffle(network, query, (batch_size, _NUM_HEADS, _SPATIAL_TOKENS, _HEAD_DIM))
    key = _shuffle(network, key, (batch_size, _NUM_HEADS, _SPATIAL_TOKENS, _HEAD_DIM))
    value = _shuffle(network, value, (batch_size, _NUM_HEADS, _SPATIAL_TOKENS, _HEAD_DIM))
    query = _apply_axial_rope(network, query, rope)
    key = _apply_axial_rope(network, key, rope)
    context = _attention_context(
        network,
        query,
        key,
        value,
        batch_size,
    )
    projected = _bf16_linear(
        network, context, weights, f"{prefix}.o_proj", _HIDDEN_SIZE, _HIDDEN_SIZE
    )
    return projected


def _cross_attention(
    network: trt.INetworkDefinition,
    queries: trt.ITensor,
    memory: _PreparedMemory,
    weights: Mapping[str, np.ndarray],
    prefix: str,
    rope: _RopeConstants,
    batch_size: int,
) -> trt.ITensor:
    query = _bf16_linear(network, queries, weights, f"{prefix}.q_proj", _HIDDEN_SIZE, _HIDDEN_SIZE)

    spatial_key_input = _fp32_sum(
        network,
        memory.spatial_values,
        memory.spatial_position,
    )
    pointer_key_input = _bf16_sum(
        network,
        memory.pointer_values,
        memory.pointer_position,
    )
    spatial_key = _bf16_linear(
        network,
        spatial_key_input,
        weights,
        f"{prefix}.k_proj",
        _MEMORY_SIZE,
        _HIDDEN_SIZE,
    )
    pointer_key = _bf16_linear(
        network,
        pointer_key_input,
        weights,
        f"{prefix}.k_proj",
        _MEMORY_SIZE,
        _HIDDEN_SIZE,
    )

    values = network.add_concatenation([memory.spatial_values, memory.pointer_values])
    values.axis = 1
    value = _bf16_linear(
        network,
        values.get_output(0),
        weights,
        f"{prefix}.v_proj",
        _MEMORY_SIZE,
        _HIDDEN_SIZE,
    )

    query = _shuffle(network, query, (batch_size, _NUM_HEADS, _SPATIAL_TOKENS, _HEAD_DIM))
    query = _apply_axial_rope(network, query, rope)
    spatial_key = _shuffle(
        network,
        spatial_key,
        (batch_size, -1, _SPATIAL_TOKENS, _HEAD_DIM),
    )
    spatial_key = _apply_axial_rope(network, spatial_key, rope)
    spatial_key = _cast(network, spatial_key, trt.bfloat16)
    spatial_key = _shuffle(network, spatial_key, (batch_size, _NUM_HEADS, -1, _HEAD_DIM))
    pointer_key = _shuffle(network, pointer_key, (batch_size, _NUM_HEADS, -1, _HEAD_DIM))
    key = network.add_concatenation([spatial_key, pointer_key])
    key.axis = 2
    value = _shuffle(network, value, (batch_size, _NUM_HEADS, -1, _HEAD_DIM))

    context = _attention_context(
        network,
        query,
        key.get_output(0),
        value,
        batch_size,
    )
    projected = _bf16_linear(
        network, context, weights, f"{prefix}.o_proj", _HIDDEN_SIZE, _HIDDEN_SIZE
    )
    return projected


def _feed_forward(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    weights: Mapping[str, np.ndarray],
    prefix: str,
) -> trt.ITensor:
    hidden = _bf16_linear(
        network, inp, weights, f"{prefix}.linear1", _HIDDEN_SIZE, _FEED_FORWARD_SIZE
    )
    hidden = network.add_activation(hidden, trt.ActivationType.RELU).get_output(0)
    output = _bf16_linear(
        network, hidden, weights, f"{prefix}.linear2", _FEED_FORWARD_SIZE, _HIDDEN_SIZE
    )
    return output


def _features_to_tokens(
    network: trt.INetworkDefinition,
    features: trt.ITensor,
    batch_size: int,
) -> trt.ITensor:
    features = _expand_batch(network, features, batch_size)
    features = _transpose(network, features, (0, 2, 3, 1))
    return _shuffle(network, features, (batch_size, _SPATIAL_TOKENS, _HIDDEN_SIZE))


def _tokens_to_features(
    network: trt.INetworkDefinition,
    tokens: trt.ITensor,
    batch_size: int,
) -> trt.ITensor:
    features = _shuffle(network, tokens, (batch_size, _HEIGHT, _WIDTH, _HIDDEN_SIZE))
    return _transpose(network, features, (0, 3, 1, 2))


def add_tracker_recurrent_conditioning(
    network: trt.INetworkDefinition,
    current_features: trt.ITensor,
    current_position: trt.ITensor,
    memory_features: trt.ITensor,
    memory_position: trt.ITensor,
    memory_temporal_offsets: trt.ITensor,
    object_pointers: trt.ITensor,
    object_pointer_temporal_offsets: trt.ITensor,
    max_object_pointers_to_use: trt.ITensor,
    weights: Mapping[str, np.ndarray],
    *,
    batch_size: int,
) -> trt.ITensor:
    """Add SAM3's recurrent memory attention and return ``[B, 256, 72, 72]``.

    Both fixed-batch graphs use the precision schedule emitted by Meta's
    compiled tracker: projection and feed-forward GEMMs plus native TensorRT
    learned GEMMs and the recurrent residual stream run in BF16, fused
    attention uses the measured L4 FP16 tactic, and spatial memory-position
    adds, RoPE math, LayerNorm, and the engine output boundary remain FP32.

    Dynamic input layouts match the existing tracker engine contract:

    * batch 1 spatial memory is ``[1, M, 5184, 64]`` and pointers are
      ``[1, P, 256]``;
    * batch 2 spatial memory is ``[2, M, 5184, 64]`` and pointers are
      ``[2, P, 256]``.
    """

    _validate_inputs(
        current_features,
        current_position,
        memory_features,
        memory_position,
        memory_temporal_offsets,
        object_pointers,
        object_pointer_temporal_offsets,
        max_object_pointers_to_use,
        batch_size,
    )
    memory = _prepare_memory(
        network,
        memory_features,
        memory_position,
        memory_temporal_offsets,
        object_pointers,
        object_pointer_temporal_offsets,
        max_object_pointers_to_use,
        weights,
        batch_size,
    )
    rope = _add_rope_constants(network)

    output = _features_to_tokens(network, current_features, batch_size)
    position = _features_to_tokens(network, current_position, batch_size)
    position_scale = add_constant(
        network,
        (1, 1, 1),
        np.full((1, 1, 1), _POSITION_SCALE, dtype=np.float32),
        dtype=np.float32,
    )
    output = _cast(network, output, trt.bfloat16)
    position = _cast(network, position, trt.bfloat16)
    position_scale = _cast(network, position_scale, trt.bfloat16)
    position = network.add_elementwise(
        position, position_scale, trt.ElementWiseOperation.PROD
    ).get_output(0)
    output = _bf16_sum(network, output, position)

    for layer_index in range(_NUM_LAYERS):
        prefix = f"{_MEMORY_ATTENTION_PREFIX}.layers.{layer_index}"
        normalized = _layer_norm(network, output, weights, f"{prefix}.layer_norm1")
        self_attention = _self_attention(
            network,
            normalized,
            weights,
            f"{prefix}.self_attn",
            rope,
            batch_size,
        )
        output = _bf16_sum(network, output, self_attention)

        normalized = _layer_norm(network, output, weights, f"{prefix}.layer_norm2")
        cross_attention = _cross_attention(
            network,
            normalized,
            memory,
            weights,
            f"{prefix}.cross_attn_image",
            rope,
            batch_size,
        )
        output = _bf16_sum(network, output, cross_attention)

        normalized = _layer_norm(network, output, weights, f"{prefix}.layer_norm3")
        feed_forward = _feed_forward(network, normalized, weights, prefix)
        output = _bf16_sum(network, output, feed_forward)

    output = _layer_norm(network, output, weights, f"{_MEMORY_ATTENTION_PREFIX}.layer_norm")
    return _tokens_to_features(network, output, batch_size)


__all__ = ["add_tracker_recurrent_conditioning"]
