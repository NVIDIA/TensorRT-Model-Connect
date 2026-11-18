# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT decoder graph of ``ApertusForCausalLM`` with a fixed-capacity native KV cache.

One engine carries two optimization profiles over the same weights: profile 0
prefills up to ``PREFILL_CHUNK_TOKENS`` tokens per launch, profile 1 decodes one
token. Tensor contract (``Sq`` = tokens in the launch, ``C`` = cache capacity):

  inputs   token_id, position_id         int32 [Sq]
           cache_write_indices           int32 [1]   first cache row written
           key_value_lengths             int32 [1]   valid rows after the write
           key_positions                 int32 [R]   0..R-1, R >= valid rows
           cache_k_<i>, cache_v_<i>      [1, Hkv, C, D] in the build precision
  outputs  logits                        float32 [1, vocab], last position only
           present_k_<i>, present_v_<i>  alias cache_k_<i>, cache_v_<i>

The numerics follow the Hugging Face reference: RMSNorm statistics in FP32
with the scale applied in the build precision, RoPE cos/sin computed in FP32
and rounded to the build precision, attention scores and softmax in FP32, and
the xIELU activation evaluated in FP32.
"""

from __future__ import annotations

import math
import sys

import ml_dtypes
import numpy as np
import tensorrt as trt

from .config import ApertusConfig
from .rope import rope_tables
from .weights import ApertusWeights, LayerWeights

PREFILL_CHUNK_TOKENS = 64
# Tactic-selection point for the active key rows; any count up to the capacity is valid.
ACTIVE_KEY_ROWS_OPTIMUM = 256
# xIELU uses expm1(m) for m <= eps < 0. TensorRT has no expm1 and exp(m) - 1
# cancels near zero, so a degree-8 Taylor polynomial (relative error < 2e-8)
# covers (-0.5, 0] and exp(m) - 1 the rest.
_EXPM1_SERIES_LIMIT = -0.5
_EXPM1_SERIES_DEGREE = 8
_MASKED_SCORE = -1.0e30


class _Network:
    """Network-building helpers; keeps every host constant alive until serialization."""

    def __init__(self, network: trt.INetworkDefinition, dtype: trt.DataType) -> None:
        self.network = network
        self.dtype = dtype
        self.storage = (
            np.dtype(ml_dtypes.bfloat16) if dtype == trt.bfloat16 else np.dtype(np.float32)
        )
        self._arrays: list[np.ndarray] = []

    def constant(self, values: np.ndarray, shape: tuple[int, ...] | None = None) -> trt.ITensor:
        # np.ascontiguousarray would turn a 0-d shape input into shape (1,); copy only if needed.
        array = np.asarray(values)
        if shape is not None:
            array = array.reshape(shape)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        self._arrays.append(array)
        if array.dtype == np.dtype(ml_dtypes.bfloat16):
            weights = trt.Weights(trt.bfloat16, array.ctypes.data, array.size)
        else:
            weights = trt.Weights(array)
        layer = self.network.add_constant(array.shape, weights)
        return layer.get_output(0)

    def scalar(
        self, value: float, rank: int, dtype: np.dtype = np.dtype(np.float32)
    ) -> trt.ITensor:
        return self.constant(np.full((1,) * rank, value, dtype=dtype))

    def cast(self, tensor: trt.ITensor, dtype: trt.DataType) -> trt.ITensor:
        if tensor.dtype == dtype:
            return tensor
        return self.network.add_cast(tensor, dtype).get_output(0)

    def binary(self, a: trt.ITensor, b: trt.ITensor, op: trt.ElementWiseOperation) -> trt.ITensor:
        return self.network.add_elementwise(a, b, op).get_output(0)

    def mul(self, a: trt.ITensor, b: trt.ITensor) -> trt.ITensor:
        return self.binary(a, b, trt.ElementWiseOperation.PROD)

    def add(self, a: trt.ITensor, b: trt.ITensor) -> trt.ITensor:
        return self.binary(a, b, trt.ElementWiseOperation.SUM)

    def reshape(self, tensor: trt.ITensor, dims: tuple[int, ...]) -> trt.ITensor:
        layer = self.network.add_shuffle(tensor)
        layer.reshape_dims = dims
        return layer.get_output(0)

    def linear(self, tensor: trt.ITensor, weight: np.ndarray) -> trt.ITensor:
        """``tensor @ weight.T`` with a Hugging Face ``[out, in]`` weight."""
        rhs = self.constant(weight.astype(self.storage, copy=False))
        return self.network.add_matrix_multiply(
            tensor, trt.MatrixOperation.NONE, rhs, trt.MatrixOperation.TRANSPOSE
        ).get_output(0)

    def rms_norm(
        self, tensor: trt.ITensor, weight: np.ndarray, eps: float, axis: int
    ) -> trt.ITensor:
        """``weight * (x * rsqrt(mean(x^2) + eps)).to(dtype)`` with FP32 statistics."""
        rank = len(tensor.shape)
        x = self.cast(tensor, trt.float32)
        mean = self.network.add_reduce(
            self.mul(x, x), trt.ReduceOperation.AVG, 1 << axis, keep_dims=True
        ).get_output(0)
        root = self.network.add_unary(
            self.add(mean, self.scalar(eps, rank)), trt.UnaryOperation.SQRT
        )
        inverse = self.network.add_unary(root.get_output(0), trt.UnaryOperation.RECIP).get_output(0)
        normalized = self.cast(self.mul(x, inverse), self.dtype)
        gamma = self.constant(weight.astype(self.storage, copy=False), (1,) * (rank - 1) + (-1,))
        return self.mul(normalized, gamma)


def _expm1_nonpositive(net: _Network, m: trt.ITensor) -> trt.ITensor:
    """Accurate ``expm1(m)`` for FP32 ``m <= 0``."""
    series = net.scalar(1.0, 2)
    for degree in range(_EXPM1_SERIES_DEGREE, 1, -1):
        series = net.add(
            net.scalar(1.0, 2), net.mul(net.mul(m, net.scalar(1.0 / degree, 2)), series)
        )
    series = net.mul(m, series)
    exp = net.network.add_unary(m, trt.UnaryOperation.EXP).get_output(0)
    direct = net.binary(exp, net.scalar(1.0, 2), trt.ElementWiseOperation.SUB)
    near_zero = net.binary(m, net.scalar(_EXPM1_SERIES_LIMIT, 2), trt.ElementWiseOperation.GREATER)
    return net.network.add_select(near_zero, series, direct).get_output(0)


def _xielu(net: _Network, tensor: trt.ITensor, layer: LayerWeights) -> trt.ITensor:
    """``where(x > 0, a_p*x*x + b*x, (expm1(min(x, eps)) - x)*a_n + b*x)`` in FP32."""
    x = net.cast(tensor, trt.float32)
    beta_x = net.mul(net.scalar(layer.xielu_beta, 2), x)
    positive = net.add(net.mul(net.mul(net.scalar(layer.xielu_alpha_p, 2), x), x), beta_x)
    clamped = net.binary(x, net.scalar(layer.xielu_eps, 2), trt.ElementWiseOperation.MIN)
    shifted = net.binary(_expm1_nonpositive(net, clamped), x, trt.ElementWiseOperation.SUB)
    negative = net.add(net.mul(shifted, net.scalar(layer.xielu_alpha_n, 2)), beta_x)
    is_positive = net.binary(x, net.scalar(0.0, 2), trt.ElementWiseOperation.GREATER)
    return net.cast(
        net.network.add_select(is_positive, positive, negative).get_output(0), net.dtype
    )


def _rows_to_heads(net: _Network, tensor: trt.ITensor, heads: int, head_dim: int) -> trt.ITensor:
    """``[Sq, heads * D]`` -> ``[1, heads, Sq, D]``."""
    layer = net.network.add_shuffle(tensor)
    layer.reshape_dims = (1, -1, heads, head_dim)
    layer.second_transpose = (0, 2, 1, 3)
    return layer.get_output(0)


def _heads_to_rows(net: _Network, tensor: trt.ITensor, width: int) -> trt.ITensor:
    """``[1, heads, Sq, D]`` -> ``[Sq, heads * D]``."""
    layer = net.network.add_shuffle(tensor)
    layer.first_transpose = (0, 2, 1, 3)
    layer.reshape_dims = (-1, width)
    return layer.get_output(0)


def _active_rows(net: _Network, cache: trt.ITensor, key_positions: trt.ITensor) -> trt.ITensor:
    """The first ``len(key_positions)`` rows of a ``[1, Hkv, C, D]`` cache."""
    shape = net.network.add_shape(cache).get_output(0)
    leading = net.network.add_slice(shape, start=(0,), shape=(2,), stride=(1,)).get_output(0)
    trailing = net.network.add_slice(shape, start=(3,), shape=(1,), stride=(1,)).get_output(0)
    rows = net.network.add_shape(key_positions).get_output(0)
    size = net.network.add_concatenation([leading, rows, trailing])
    size.axis = 0
    layer = net.network.add_slice(
        cache, start=(0, 0, 0, 0), shape=(1, 1, 1, 1), stride=(1, 1, 1, 1)
    )
    layer.set_input(2, size.get_output(0))
    return layer.get_output(0)


def _attention_masks(
    net: _Network,
    token_id: trt.ITensor,
    cache_write_indices: trt.ITensor,
    key_value_lengths: trt.ITensor,
    key_positions: trt.ITensor,
) -> tuple[trt.ITensor, trt.ITensor]:
    """BOOL causal mask ``[1, 1, 1, Sq, R]`` and active-row mask ``[1, 1, 1, R, 1]``."""
    network = net.network
    query_count = network.add_shape(token_id).get_output(0)
    query_iota = network.add_fill((1,), trt.FillOperation.LINSPACE, trt.int32)
    query_iota.set_input(0, query_count)
    query_iota.set_input(1, net.constant(np.array(0, dtype=np.int32)))
    query_iota.set_input(2, net.constant(np.array([1], dtype=np.int32)))
    # Query i sits at cache row write_index + i and may read rows <= that row.
    limits = net.add(
        net.add(query_iota.get_output(0), cache_write_indices),
        net.constant(np.array([1], dtype=np.int32)),
    )
    limits = net.reshape(limits, (1, 1, 1, -1, 1))
    keys = net.reshape(key_positions, (1, 1, 1, 1, -1))
    causal = net.binary(keys, limits, trt.ElementWiseOperation.LESS)
    active = net.binary(
        keys, net.reshape(key_value_lengths, (1, 1, 1, 1, 1)), trt.ElementWiseOperation.LESS
    )
    mask = net.binary(causal, active, trt.ElementWiseOperation.AND)
    rows = net.binary(
        net.reshape(key_positions, (1, 1, 1, -1, 1)),
        net.reshape(key_value_lengths, (1, 1, 1, 1, 1)),
        trt.ElementWiseOperation.LESS,
    )
    return mask, rows


def _attention(
    net: _Network,
    config: ApertusConfig,
    query: trt.ITensor,
    keys: trt.ITensor,
    values: trt.ITensor,
    masks: tuple[trt.ITensor, trt.ITensor],
) -> trt.ITensor:
    """Grouped-query attention of ``[1, H, Sq, D]`` queries over ``[1, Hkv, R, D]`` cache rows."""
    kv_heads, head_dim = config.num_key_value_heads, config.head_dim
    groups = config.num_attention_heads // kv_heads
    mask, active_rows = masks
    zero = net.cast(net.scalar(0.0, 5), net.dtype)
    # Rows past the valid length may hold stale or never-written data; zero them so
    # that masked rows cannot carry NaN into the probability-weighted sum.
    keys = net.network.add_select(
        active_rows, net.reshape(keys, (1, kv_heads, 1, -1, head_dim)), zero
    ).get_output(0)
    values = net.network.add_select(
        active_rows, net.reshape(values, (1, kv_heads, 1, -1, head_dim)), zero
    ).get_output(0)
    grouped = net.reshape(query, (1, kv_heads, groups, -1, head_dim))
    scores = net.network.add_matrix_multiply(
        net.cast(grouped, trt.float32),
        trt.MatrixOperation.NONE,
        net.cast(keys, trt.float32),
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    scores = net.mul(scores, net.scalar(1.0 / math.sqrt(head_dim), 5))
    scores = net.network.add_select(mask, scores, net.scalar(_MASKED_SCORE, 5)).get_output(0)
    softmax = net.network.add_softmax(scores)
    softmax.axes = 1 << 4
    probabilities = net.cast(softmax.get_output(0), net.dtype)
    context = net.network.add_matrix_multiply(
        probabilities, trt.MatrixOperation.NONE, values, trt.MatrixOperation.NONE
    ).get_output(0)
    return net.reshape(context, (1, config.num_attention_heads, -1, head_dim))


def _rope(
    net: _Network,
    tensor: trt.ITensor,
    cos: trt.ITensor,
    sin: trt.ITensor,
    positions: trt.ITensor,
    head_dim: int,
) -> trt.ITensor:
    layer = net.network.add_rotary_embedding(tensor, cos, sin, False, head_dim)
    layer.set_input(3, positions)
    return layer.get_output(0)


def _decoder_layer(
    net: _Network,
    config: ApertusConfig,
    layer: LayerWeights,
    hidden: trt.ITensor,
    caches: tuple[trt.ITensor, trt.ITensor],
    io: dict[str, trt.ITensor],
) -> tuple[trt.ITensor, trt.ITensor, trt.ITensor]:
    eps, head_dim = config.rms_norm_eps, config.head_dim
    heads, kv_heads = config.num_attention_heads, config.num_key_value_heads
    normed = net.rms_norm(hidden, layer.attention_norm, eps, axis=1)
    query = _rows_to_heads(net, net.linear(normed, layer.q_proj), heads, head_dim)
    key = _rows_to_heads(net, net.linear(normed, layer.k_proj), kv_heads, head_dim)
    value = _rows_to_heads(net, net.linear(normed, layer.v_proj), kv_heads, head_dim)
    query = _rope(
        net,
        net.rms_norm(query, layer.q_norm, eps, axis=3),
        io["rope_cos"],
        io["rope_sin"],
        io["rope_positions"],
        head_dim,
    )
    key = _rope(
        net,
        net.rms_norm(key, layer.k_norm, eps, axis=3),
        io["rope_cos"],
        io["rope_sin"],
        io["rope_positions"],
        head_dim,
    )
    updates = []
    for cache, rows in zip(caches, (key, value)):
        update = net.network.add_kv_cache_update(
            cache, rows, io["cache_write_indices"], trt.KVCacheMode.LINEAR
        )
        if update is None:
            raise RuntimeError("TensorRT could not create the Apertus KV-cache update")
        updates.append(update.get_output(0))
    active = [_active_rows(net, cache, io["key_positions"]) for cache in updates]
    context = _attention(net, config, query, active[0], active[1], io["masks"])
    hidden = net.add(
        hidden, net.linear(_heads_to_rows(net, context, config.attention_size), layer.o_proj)
    )
    normed = net.rms_norm(hidden, layer.feedforward_norm, eps, axis=1)
    activated = _xielu(net, net.linear(normed, layer.up_proj), layer)
    hidden = net.add(hidden, net.linear(activated, layer.down_proj))
    return hidden, updates[0], updates[1]


def _last_row(net: _Network, tensor: trt.ITensor, width: int) -> trt.ITensor:
    shape = net.network.add_shape(tensor).get_output(0)
    start = net.binary(
        shape, net.constant(np.array([1, width], dtype=np.int64)), trt.ElementWiseOperation.SUB
    )
    layer = net.network.add_slice(tensor, start=(0, 0), shape=(1, width), stride=(1, 1))
    layer.set_input(1, start)
    return layer.get_output(0)


def build_engine(
    config: ApertusConfig,
    weights: ApertusWeights,
    *,
    precision: str,
    max_cache_length: int,
    verbose: bool,
) -> bytes:
    """Serialize the dual-profile prefill / decode engine."""
    if precision not in ("bf16", "fp32"):
        raise ValueError("Apertus engines are built in bf16 or fp32")
    if not 1 <= max_cache_length <= config.max_position_embeddings:
        raise ValueError("Apertus cache capacity must be within max_position_embeddings")
    dtype = trt.bfloat16 if precision == "bf16" else trt.float32
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    builder_config.builder_optimization_level = 3
    if precision == "fp32":
        # FP32 engines are the exact-parity build: no TF32 rounding of FP32 GEMM inputs.
        builder_config.clear_flag(trt.BuilderFlag.TF32)
    net = _Network(network, dtype)

    token_id = network.add_input("token_id", trt.int32, (-1,))
    position_id = network.add_input("position_id", trt.int32, (-1,))
    cache_write_indices = network.add_input("cache_write_indices", trt.int32, (1,))
    key_value_lengths = network.add_input("key_value_lengths", trt.int32, (1,))
    key_positions = network.add_input("key_positions", trt.int32, (-1,))
    cache_shape = (1, config.num_key_value_heads, max_cache_length, config.head_dim)
    caches = [
        (
            network.add_input(f"cache_k_{index}", dtype, cache_shape),
            network.add_input(f"cache_v_{index}", dtype, cache_shape),
        )
        for index in range(config.num_hidden_layers)
    ]
    chunk = min(PREFILL_CHUNK_TOKENS, max_cache_length)
    for minimum, optimum, maximum in ((1, chunk, chunk), (1, 1, 1)):
        profile = builder.create_optimization_profile()
        for name in ("token_id", "position_id"):
            profile.set_shape(name, (minimum,), (optimum,), (maximum,))
        profile.set_shape(
            "key_positions",
            (1,),
            (min(ACTIVE_KEY_ROWS_OPTIMUM, max_cache_length),),
            (max_cache_length,),
        )
        builder_config.add_optimization_profile(profile)

    cos, sin = rope_tables(config.head_dim, config.rope, max_cache_length)
    io = {
        "cache_write_indices": cache_write_indices,
        "key_positions": key_positions,
        "rope_cos": net.constant(cos.astype(net.storage)),
        "rope_sin": net.constant(sin.astype(net.storage)),
        "rope_positions": net.reshape(position_id, (1, -1)),
        "masks": _attention_masks(
            net, token_id, cache_write_indices, key_value_lengths, key_positions
        ),
    }
    embedding = net.constant(weights.embedding.astype(net.storage, copy=False))
    hidden = network.add_gather(embedding, token_id, 0).get_output(0)
    presents = []
    for layer, layer_caches in zip(weights.layers, caches):
        hidden, present_k, present_v = _decoder_layer(net, config, layer, hidden, layer_caches, io)
        presents.append((present_k, present_v))
    hidden = net.rms_norm(
        _last_row(net, hidden, config.hidden_size), weights.final_norm, config.rms_norm_eps, axis=1
    )
    logits = net.cast(net.linear(hidden, weights.lm_head), trt.float32)
    logits.name = "logits"
    network.mark_output(logits)
    for index, (present_k, present_v) in enumerate(presents):
        present_k.name = f"present_k_{index}"
        present_v.name = f"present_v_{index}"
        network.mark_output(present_k)
        network.mark_output(present_v)

    if verbose:
        print(
            f"[trtmc build] apertus {precision} engine: layers={config.num_hidden_layers} "
            f"hidden={config.hidden_size} cache={max_cache_length} prefill_chunk={chunk}",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("Apertus engine build failed")
    return bytes(plan)
