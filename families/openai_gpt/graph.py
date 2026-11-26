# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT decoder graph of ``OpenAIGPTLMHeadModel`` with a fixed-capacity native KV cache.

One FP32 engine carries two optimization profiles over the same weights: profile 0
prefills a prefix of up to the cache capacity in one launch, profile 1 decodes one
token. Tensor contract (``Sq`` = tokens in the launch, ``C`` = cache capacity):

  inputs   token_id, position_id         int32 [Sq]
           cache_write_indices           int32 [1]   first cache row written
           key_value_lengths             int32 [1]   valid rows after the write
           key_positions                 int32 [R]   0..R-1, valid rows <= R <= C
           cache_k_<i>, cache_v_<i>      float32 [1, n_head, C, head_dim]
  outputs  logits                        float32 [1, vocab], last position only
           present_k_<i>, present_v_<i>  alias cache_k_<i>, cache_v_<i>

The arithmetic follows the Hugging Face reference in float32: learned token plus
position embeddings, post-norm blocks ``n = ln_1(x + attn(x))``, ``h = ln_2(n +
mlp(n))``, scores scaled by ``1 / sqrt(head_dim)`` before the causal mask, the
``gelu_new`` tanh approximation in the reference's order of operations, no final
norm, and an LM head that reuses the token embedding. TF32 is disabled, so every
matrix product keeps float32 inputs.
"""

from __future__ import annotations

import math
import sys

import numpy as np
import tensorrt as trt

from .config import OpenAIGPTConfig
from .weights import LayerWeights, OpenAIGPTWeights

# Tactic-selection points; any length up to the capacity is valid.
PREFILL_TOKENS_OPTIMUM = 64
ACTIVE_KEY_ROWS_OPTIMUM = 256
# A masked score; its probability is exactly zero after the softmax.
_MASKED_SCORE = -1.0e30
_GELU_SCALE = math.sqrt(2.0 / math.pi)
_GELU_CUBIC = 0.044715


class _Network:
    """Float32 network-building helpers; keeps every host constant alive until serialization."""

    def __init__(self, network: trt.INetworkDefinition) -> None:
        self.network = network
        self._arrays: list[np.ndarray] = []

    def constant(self, values: np.ndarray, shape: tuple[int, ...] | None = None) -> trt.ITensor:
        array = np.asarray(values)
        if shape is not None:
            array = array.reshape(shape)
        if not array.flags.c_contiguous:
            array = np.ascontiguousarray(array)
        self._arrays.append(array)
        return self.network.add_constant(array.shape, trt.Weights(array)).get_output(0)

    def scalar(self, value: float, rank: int) -> trt.ITensor:
        return self.constant(np.full((1,) * rank, value, dtype=np.float32))

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

    def conv1d(self, tensor: trt.ITensor, weight: np.ndarray, bias: np.ndarray) -> trt.ITensor:
        """Hugging Face ``Conv1D``: ``tensor @ weight + bias`` with an ``[in, out]`` weight."""
        product = self.network.add_matrix_multiply(
            tensor, trt.MatrixOperation.NONE, self.constant(weight), trt.MatrixOperation.NONE
        ).get_output(0)
        return self.add(product, self.constant(bias, (1, -1)))

    def layer_norm(
        self, tensor: trt.ITensor, weight: np.ndarray, bias: np.ndarray, eps: float
    ) -> trt.ITensor:
        """``nn.LayerNorm`` over the last axis of ``[Sq, n_embd]`` (biased variance)."""
        layer = self.network.add_normalization(
            tensor, self.constant(weight, (1, -1)), self.constant(bias, (1, -1)), 1 << 1
        )
        layer.epsilon = eps
        return layer.get_output(0)


def _gelu_new(net: _Network, x: trt.ITensor) -> trt.ITensor:
    """``0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x**3)))``, as transformers orders it."""
    cube = net.mul(net.mul(x, x), x)
    inner = net.add(x, net.mul(net.scalar(_GELU_CUBIC, 2), cube))
    tanh = net.network.add_activation(
        net.mul(net.scalar(_GELU_SCALE, 2), inner), trt.ActivationType.TANH
    ).get_output(0)
    return net.mul(net.mul(net.scalar(0.5, 2), x), net.add(net.scalar(1.0, 2), tanh))


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


def _columns(net: _Network, tensor: trt.ITensor, start: int, width: int) -> trt.ITensor:
    """Columns ``[start, start + width)`` of a ``[Sq, N]`` tensor."""
    rows = net.network.add_slice(
        net.network.add_shape(tensor).get_output(0), start=(0,), shape=(1,), stride=(1,)
    ).get_output(0)
    size = net.network.add_concatenation([rows, net.constant(np.array([width], dtype=np.int64))])
    size.axis = 0
    layer = net.network.add_slice(tensor, start=(0, start), shape=(1, width), stride=(1, 1))
    layer.set_input(2, size.get_output(0))
    return layer.get_output(0)


def _active_rows(net: _Network, cache: trt.ITensor, key_positions: trt.ITensor) -> trt.ITensor:
    """The first ``len(key_positions)`` rows of a ``[1, heads, C, D]`` cache."""
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
    """BOOL causal mask ``[1, 1, Sq, R]`` and valid-row mask ``[1, 1, R, 1]``."""
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
    keys = net.reshape(key_positions, (1, 1, 1, -1))
    causal = net.binary(keys, net.reshape(limits, (1, 1, -1, 1)), trt.ElementWiseOperation.LESS)
    valid = net.reshape(key_value_lengths, (1, 1, 1, 1))
    mask = net.binary(
        causal, net.binary(keys, valid, trt.ElementWiseOperation.LESS), trt.ElementWiseOperation.AND
    )
    rows = net.binary(
        net.reshape(key_positions, (1, 1, -1, 1)), valid, trt.ElementWiseOperation.LESS
    )
    return mask, rows


def _attention(
    net: _Network,
    config: OpenAIGPTConfig,
    query: trt.ITensor,
    keys: trt.ITensor,
    values: trt.ITensor,
    masks: tuple[trt.ITensor, trt.ITensor],
) -> trt.ITensor:
    """Causal attention of ``[1, H, Sq, D]`` queries over ``[1, H, R, D]`` cache rows."""
    mask, valid_rows = masks
    zero = net.scalar(0.0, 4)
    # Rows past the valid length may hold stale or never-written data; zero them so
    # that a masked row cannot carry NaN into the probability-weighted sum.
    keys = net.network.add_select(valid_rows, keys, zero).get_output(0)
    values = net.network.add_select(valid_rows, values, zero).get_output(0)
    scores = net.network.add_matrix_multiply(
        query, trt.MatrixOperation.NONE, keys, trt.MatrixOperation.TRANSPOSE
    ).get_output(0)
    # The reference divides the scores by sqrt(head_dim) before masking; so does this graph.
    scores = net.binary(
        scores, net.scalar(math.sqrt(config.head_dim), 4), trt.ElementWiseOperation.DIV
    )
    scores = net.network.add_select(mask, scores, net.scalar(_MASKED_SCORE, 4)).get_output(0)
    softmax = net.network.add_softmax(scores)
    softmax.axes = 1 << 3
    return net.network.add_matrix_multiply(
        softmax.get_output(0), trt.MatrixOperation.NONE, values, trt.MatrixOperation.NONE
    ).get_output(0)


def _block(
    net: _Network,
    config: OpenAIGPTConfig,
    layer: LayerWeights,
    hidden: trt.ITensor,
    caches: tuple[trt.ITensor, trt.ITensor],
    io: dict,
) -> tuple[trt.ITensor, trt.ITensor, trt.ITensor]:
    embd, heads, head_dim = config.n_embd, config.n_head, config.head_dim
    qkv = net.conv1d(hidden, layer.attn_weight, layer.attn_bias)
    query, key, value = (
        _rows_to_heads(net, _columns(net, qkv, index * embd, embd), heads, head_dim)
        for index in range(3)
    )
    updates = []
    for cache, rows in zip(caches, (key, value)):
        update = net.network.add_kv_cache_update(
            cache, rows, io["cache_write_indices"], trt.KVCacheMode.LINEAR
        )
        if update is None:
            raise RuntimeError("TensorRT could not create the OpenAI GPT KV-cache update")
        updates.append(update.get_output(0))
    active = [_active_rows(net, cache, io["key_positions"]) for cache in updates]
    context = _attention(net, config, query, active[0], active[1], io["masks"])
    attended = net.conv1d(
        _heads_to_rows(net, context, embd), layer.attn_proj_weight, layer.attn_proj_bias
    )
    eps = config.layer_norm_epsilon
    normed = net.layer_norm(net.add(hidden, attended), layer.ln_1_weight, layer.ln_1_bias, eps)
    inner = _gelu_new(net, net.conv1d(normed, layer.fc_weight, layer.fc_bias))
    mlp = net.conv1d(inner, layer.mlp_proj_weight, layer.mlp_proj_bias)
    hidden = net.layer_norm(net.add(normed, mlp), layer.ln_2_weight, layer.ln_2_bias, eps)
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
    config: OpenAIGPTConfig,
    weights: OpenAIGPTWeights,
    *,
    max_cache_length: int,
    verbose: bool,
) -> bytes:
    """Serialize the float32 prefill / decode engine for a cache of ``max_cache_length`` rows."""
    if not 1 <= max_cache_length <= config.n_positions:
        raise ValueError("OpenAI GPT cache capacity must be within n_positions")
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    builder_config = builder.create_builder_config()
    builder_config.builder_optimization_level = 3
    # The checkpoint and the reference are float32; TF32 would round every matrix product.
    builder_config.clear_flag(trt.BuilderFlag.TF32)
    net = _Network(network)

    token_id = network.add_input("token_id", trt.int32, (-1,))
    position_id = network.add_input("position_id", trt.int32, (-1,))
    cache_write_indices = network.add_input("cache_write_indices", trt.int32, (1,))
    key_value_lengths = network.add_input("key_value_lengths", trt.int32, (1,))
    key_positions = network.add_input("key_positions", trt.int32, (-1,))
    cache_shape = (1, config.n_head, max_cache_length, config.head_dim)
    caches = [
        (
            network.add_input(f"cache_k_{index}", trt.float32, cache_shape),
            network.add_input(f"cache_v_{index}", trt.float32, cache_shape),
        )
        for index in range(config.n_layer)
    ]
    prefill_optimum = min(PREFILL_TOKENS_OPTIMUM, max_cache_length)
    rows_optimum = min(ACTIVE_KEY_ROWS_OPTIMUM, max_cache_length)
    for minimum, optimum, maximum in ((1, prefill_optimum, max_cache_length), (1, 1, 1)):
        profile = builder.create_optimization_profile()
        for name in ("token_id", "position_id"):
            profile.set_shape(name, (minimum,), (optimum,), (maximum,))
        profile.set_shape("key_positions", (1,), (rows_optimum,), (max_cache_length,))
        builder_config.add_optimization_profile(profile)

    io = {
        "cache_write_indices": cache_write_indices,
        "key_positions": key_positions,
        "masks": _attention_masks(
            net, token_id, cache_write_indices, key_value_lengths, key_positions
        ),
    }
    # One constant serves the embedding lookup and the tied LM head.
    tokens_embed = net.constant(weights.tokens_embed)
    hidden = net.add(
        network.add_gather(tokens_embed, token_id, 0).get_output(0),
        network.add_gather(net.constant(weights.positions_embed), position_id, 0).get_output(0),
    )
    presents = []
    for layer, layer_caches in zip(weights.layers, caches):
        hidden, present_k, present_v = _block(net, config, layer, hidden, layer_caches, io)
        presents.append((present_k, present_v))
    logits = network.add_matrix_multiply(
        _last_row(net, hidden, config.n_embd),
        trt.MatrixOperation.NONE,
        tokens_embed,
        trt.MatrixOperation.TRANSPOSE,
    ).get_output(0)
    logits.name = "logits"
    network.mark_output(logits)
    for index, (present_k, present_v) in enumerate(presents):
        present_k.name = f"present_k_{index}"
        present_v.name = f"present_v_{index}"
        network.mark_output(present_k)
        network.mark_output(present_v)

    if verbose:
        print(
            f"[trtmc build] openai_gpt fp32 engine: layers={config.n_layer} "
            f"n_embd={config.n_embd} cache={max_cache_length}",
            file=sys.stderr,
        )
    plan = builder.build_serialized_network(network, builder_config)
    if plan is None:
        raise RuntimeError("OpenAI GPT engine build failed")
    return bytes(plan)
