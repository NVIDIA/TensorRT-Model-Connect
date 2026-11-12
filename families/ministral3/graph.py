# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT graph of the bidirectional Ministral 3 encoder.

One engine maps the token IDs of one sequence, ``input_ids [S]``, to the final
RMS-normalized hidden states, ``hidden_states [S, hidden]`` (float32). Every
token attends to every token (no mask); positions are ``0 .. S - 1``.

Rounding points follow the Transformers module at the engine precision:
RMSNorm normalizes in float32, rounds to the working type and then multiplies
by the stored weight; the query scale and SiLU are evaluated in float32 and
rounded once; rotary tables are rounded to the working type before use.
"""

from __future__ import annotations

import math
import sys

import ml_dtypes
import numpy as np
import tensorrt as trt

from . import rope as rope_tables
from .config import ModelConfig

_PRECISIONS = {"fp32": trt.float32, "bf16": trt.bfloat16}


class _Constants:
    """Creates TensorRT constants and keeps their host buffers alive until the build ends."""

    def __init__(self, network: trt.INetworkDefinition, dtype: trt.DataType) -> None:
        self.network = network
        self.dtype = dtype
        self._keep: list[np.ndarray] = []

    def add(
        self, values: np.ndarray, shape: tuple[int, ...] | None = None, *, dtype=None
    ) -> trt.ITensor:
        dtype = self.dtype if dtype is None else dtype
        if dtype == trt.bfloat16:
            array = np.ascontiguousarray(np.asarray(values).astype(ml_dtypes.bfloat16))
            weights = trt.Weights(trt.bfloat16, array.ctypes.data, array.size)
        elif dtype == trt.float32:
            array = np.ascontiguousarray(np.asarray(values).astype(np.float32))
            weights = trt.Weights(array)
        elif dtype in (trt.int32, trt.int64):
            integer = np.int32 if dtype == trt.int32 else np.int64
            array = np.ascontiguousarray(np.asarray(values).astype(integer))
            weights = trt.Weights(array)
        else:
            raise ValueError(f"unsupported constant type {dtype}")
        self._keep.append(array)
        layer = self.network.add_constant(tuple(array.shape) if shape is None else shape, weights)
        return layer.get_output(0)


def _cast(network: trt.INetworkDefinition, tensor: trt.ITensor, dtype: trt.DataType) -> trt.ITensor:
    if tensor.dtype == dtype:
        return tensor
    return network.add_cast(tensor, dtype).get_output(0)


def _elementwise(network, left, right, op) -> trt.ITensor:
    return network.add_elementwise(left, right, op).get_output(0)


def _linear(network, constants: _Constants, rows: trt.ITensor, weight: np.ndarray) -> trt.ITensor:
    """``rows [S, in] @ weight.T`` for a checkpoint weight stored as ``[out, in]``."""
    matrix = constants.add(weight)
    return network.add_matrix_multiply(
        rows, trt.MatrixOperation.NONE, matrix, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)


def _rms_norm(network, constants: _Constants, rows: trt.ITensor, weight: np.ndarray, eps: float):
    work = rows.dtype
    x = _cast(network, rows, trt.float32)
    square = _elementwise(network, x, x, trt.ElementWiseOperation.PROD)
    mean = network.add_reduce(square, trt.ReduceOperation.AVG, 1 << 1, keep_dims=True).get_output(0)
    epsilon = constants.add(np.array([[eps]], dtype=np.float32), dtype=trt.float32)
    root = network.add_unary(
        _elementwise(network, mean, epsilon, trt.ElementWiseOperation.SUM), trt.UnaryOperation.SQRT
    ).get_output(0)
    normalized = _cast(network, _elementwise(network, x, root, trt.ElementWiseOperation.DIV), work)
    gamma = constants.add(np.asarray(weight).reshape(1, -1))
    return _elementwise(network, gamma, normalized, trt.ElementWiseOperation.PROD)


def _rows_to_heads(network, rows: trt.ITensor, heads: int, head_dim: int) -> trt.ITensor:
    layer = network.add_shuffle(rows)
    layer.reshape_dims = (1, -1, heads, head_dim)
    layer.second_transpose = trt.Permutation([0, 2, 1, 3])
    return layer.get_output(0)


def _heads_to_rows(network, heads_4d: trt.ITensor, width: int) -> trt.ITensor:
    layer = network.add_shuffle(heads_4d)
    layer.first_transpose = trt.Permutation([0, 2, 1, 3])
    layer.reshape_dims = (-1, width)
    return layer.get_output(0)


def _slice_columns(network, constants: _Constants, rows, sequence, start: int, width: int):
    """Columns ``[start, start + width)`` of a dynamic ``[S, *]`` tensor."""
    width_tensor = constants.add(np.array([width]), dtype=trt.int64)
    size = network.add_concatenation([sequence, width_tensor])
    size.axis = 0
    layer = network.add_slice(rows, start=(0, start), shape=(1, width), stride=(1, 1))
    layer.set_input(2, size.get_output(0))
    return layer.get_output(0)


def _silu(network, gate: trt.ITensor) -> trt.ITensor:
    x = _cast(network, gate, trt.float32)
    sigmoid = network.add_activation(x, trt.ActivationType.SIGMOID).get_output(0)
    return _cast(
        network, _elementwise(network, x, sigmoid, trt.ElementWiseOperation.PROD), gate.dtype
    )


def build_engine(
    config: ModelConfig,
    weights: dict[str, np.ndarray],
    *,
    precision: str,
    max_sequence_length: int,
    opt_sequence_length: int,
    verbose: bool = False,
) -> bytes:
    """Serialize the encoder plan for sequences of 1 to ``max_sequence_length`` tokens."""
    if precision not in _PRECISIONS:
        raise ValueError(f"ministral3 builds fp32 or bf16 engines, got {precision!r}")
    if not 1 <= opt_sequence_length <= max_sequence_length:
        raise ValueError("opt_sequence_length must lie in [1, max_sequence_length]")
    work = _PRECISIONS[precision]
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    if builder_config is None:
        raise RuntimeError("TensorRT could not create a builder config")
    builder_config.builder_optimization_level = 3
    builder_config.clear_flag(trt.BuilderFlag.TF32)
    constants = _Constants(network, work)

    hidden = config.hidden_size
    heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    q_width = config.attention_size
    kv_width = config.kv_attention_size

    input_ids = network.add_input("input_ids", trt.int32, (-1,))
    profile = builder.create_optimization_profile()
    profile.set_shape("input_ids", (1,), (opt_sequence_length,), (max_sequence_length,))
    builder_config.add_optimization_profile(profile)
    sequence = network.add_shape(input_ids).get_output(0)

    # The RotaryEmbedding operator takes its positionIds as INT64.
    all_positions = constants.add(np.arange(max_sequence_length, dtype=np.int64), dtype=trt.int64)
    position_slice = network.add_slice(all_positions, start=(0,), shape=(1,), stride=(1,))
    position_slice.set_input(2, sequence)
    positions = network.add_shuffle(position_slice.get_output(0))
    positions.reshape_dims = (1, -1)
    positions_2d = positions.get_output(0)

    cos, sin = rope_tables.rotary_tables(config.rope, head_dim, max_sequence_length)
    cos_table = constants.add(cos)
    sin_table = constants.add(sin)

    # The attention logit scale 1/sqrt(head_dim) is folded into the per-position
    # query multiplier and applied in float32 before the single rounding.
    softmax_scale = np.float32(1.0 / math.sqrt(head_dim))
    if rope_tables.needs_query_scale(config.rope, max_sequence_length):
        scales = rope_tables.query_scales(config.rope, max_sequence_length)
        if work == trt.bfloat16:
            scales = scales.astype(ml_dtypes.bfloat16).astype(np.float32)
        factors = constants.add(scales * softmax_scale, dtype=trt.float32)
        factor_slice = network.add_slice(factors, start=(0,), shape=(1,), stride=(1,))
        factor_slice.set_input(2, sequence)
        factor_4d = network.add_shuffle(factor_slice.get_output(0))
        factor_4d.reshape_dims = (1, 1, -1, 1)
        query_factor = factor_4d.get_output(0)
    else:
        query_factor = constants.add(
            np.full((1, 1, 1, 1), softmax_scale, dtype=np.float32), dtype=trt.float32
        )

    embedding = constants.add(weights["embed_tokens.weight"])
    state = network.add_gather(embedding, input_ids, 0).get_output(0)

    for layer in range(config.num_hidden_layers):
        prefix = f"layers.{layer}"
        normed = _rms_norm(
            network,
            constants,
            state,
            weights[f"{prefix}.input_layernorm.weight"],
            config.rms_norm_eps,
        )
        qkv_weight = np.concatenate(
            [weights[f"{prefix}.self_attn.{name}_proj.weight"] for name in ("q", "k", "v")], axis=0
        )
        qkv = _linear(network, constants, normed, qkv_weight)
        q_rows = _slice_columns(network, constants, qkv, sequence, 0, q_width)
        k_rows = _slice_columns(network, constants, qkv, sequence, q_width, kv_width)
        v_rows = _slice_columns(network, constants, qkv, sequence, q_width + kv_width, kv_width)

        rotated = []
        for rows, count in ((q_rows, heads), (k_rows, kv_heads)):
            rotary = network.add_rotary_embedding(
                _rows_to_heads(network, rows, count, head_dim),
                cos_table,
                sin_table,
                False,
                head_dim,
            )
            rotary.set_input(3, positions_2d)
            rotated.append(rotary.get_output(0))
        query, key = rotated
        query = _cast(
            network,
            _elementwise(
                network,
                _cast(network, query, trt.float32),
                query_factor,
                trt.ElementWiseOperation.PROD,
            ),
            work,
        )
        value = _rows_to_heads(network, v_rows, kv_heads, head_dim)
        attention = network.add_attention(
            query, key, value, trt.AttentionNormalizationOp.SOFTMAX, False
        )
        attention.decomposable = True
        context = _heads_to_rows(network, attention.get_output(0), q_width)
        attended = _linear(
            network, constants, context, weights[f"{prefix}.self_attn.o_proj.weight"]
        )
        state = _elementwise(network, state, attended, trt.ElementWiseOperation.SUM)

        normed = _rms_norm(
            network,
            constants,
            state,
            weights[f"{prefix}.post_attention_layernorm.weight"],
            config.rms_norm_eps,
        )
        gate_up_weight = np.concatenate(
            [weights[f"{prefix}.mlp.{name}_proj.weight"] for name in ("gate", "up")], axis=0
        )
        gate_up = _linear(network, constants, normed, gate_up_weight)
        width = config.intermediate_size
        gate = _slice_columns(network, constants, gate_up, sequence, 0, width)
        up = _slice_columns(network, constants, gate_up, sequence, width, width)
        product = _elementwise(network, _silu(network, gate), up, trt.ElementWiseOperation.PROD)
        down = _linear(network, constants, product, weights[f"{prefix}.mlp.down_proj.weight"])
        state = _elementwise(network, state, down, trt.ElementWiseOperation.SUM)

    final = _rms_norm(network, constants, state, weights["norm.weight"], config.rms_norm_eps)
    output = _cast(network, final, trt.float32)
    output.name = "hidden_states"
    network.mark_output(output)

    if verbose:
        print(
            f"[trtmc build] ministral3 encoder: {config.num_hidden_layers} layers, hidden={hidden}, "
            f"sequence 1..{max_sequence_length} (opt {opt_sequence_length}), precision={precision}",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the ministral3 encoder")
    return bytes(plan)
