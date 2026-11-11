# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Laguna TensorRT decoder graph.

One strongly typed network serves a multi-row prefill profile and a one-row
decode profile. Tensor contract (``rows`` is runtime-dynamic):

* inputs: ``token_id`` / ``position_id`` int32 ``[rows]``; ``attention_mask``
  float32 ``[rows, max_cache + rows]`` (additive; the runtime hides stale cache
  rows and future rows); ``cache_k_{i}`` / ``cache_v_{i}`` ``[max_cache,
  kv_heads * head_dim]`` where cache row ``c`` holds position ``c``;
* outputs: ``logits`` float32 ``[1, vocab]`` for the last row and
  ``present_k_{i}`` / ``present_v_{i}`` ``[rows, kv_heads * head_dim]``.

The layer math follows the reference ``LagunaForCausalLM``: per-layer query
head counts, per-head q/k RMSNorm, per-layer-type RoPE (YaRN on the first
``rotary_dim`` channels of full-attention layers), a sliding window on
sliding-attention layers, a softplus per-head output gate, a dense SwiGLU on
``dense`` layers and a sigmoid-routed MoE with a selection bias, renormalised
top-k weights, a routed scaling factor and an ungated shared expert on
``sparse`` layers. Every routed expert is evaluated on every row with batched
matmuls; unselected experts receive weight zero.
"""

from __future__ import annotations

import math

import numpy as np
import tensorrt as trt

from .checkpoint import BF16, LagunaWeights
from .config import SLIDING_ATTENTION, LagunaConfig
from .rope import rope_tables


# Prefill chunk bounds of the multi-row profile.
PREFILL_OPT_ROWS = 64
PREFILL_MAX_ROWS = 128
# Additive penalty for keys outside the sliding window; the same order of
# magnitude as the runtime's masked score, representable in bf16.
_WINDOW_PENALTY = -1.0e4

_NONE = trt.MatrixOperation.NONE
_TRANSPOSE = trt.MatrixOperation.TRANSPOSE
_EW = trt.ElementWiseOperation


def layer_tensor_name(stem: str, layer: int) -> str:
    return f"{stem}_{layer}"


class _Graph:
    """Network plus the host arrays its raw-pointer constants reference."""

    def __init__(self, network: trt.INetworkDefinition, work: trt.DataType) -> None:
        self.net = network
        self.work = work
        self._keep: list[np.ndarray] = []

    def const(self, array: np.ndarray, shape: tuple[int, ...] | None = None) -> trt.ITensor:
        array = np.ascontiguousarray(array)
        shape = tuple(array.shape) if shape is None else shape
        if array.dtype == BF16:
            # TensorRT keeps only the pointer; the array must outlive the build.
            self._keep.append(array)
            weights = trt.Weights(trt.bfloat16, array.ctypes.data, array.size)
        else:
            weights = trt.Weights(array)
        return self.net.add_constant(shape, weights).get_output(0)

    def scalar(self, value: float, rank: int, dtype: trt.DataType) -> trt.ITensor:
        tensor = self.const(np.full((1,) * rank, value, dtype=np.float32))
        return self.cast(tensor, dtype)

    def cast(self, tensor: trt.ITensor, dtype: trt.DataType) -> trt.ITensor:
        if tensor.dtype == dtype:
            return tensor
        return self.net.add_cast(tensor, dtype).get_output(0)

    def ew(self, a: trt.ITensor, b: trt.ITensor, op: trt.ElementWiseOperation) -> trt.ITensor:
        return self.net.add_elementwise(a, b, op).get_output(0)

    def unary(self, a: trt.ITensor, op: trt.UnaryOperation) -> trt.ITensor:
        return self.net.add_unary(a, op).get_output(0)

    def reshape(self, tensor: trt.ITensor, dims: tuple[int, ...]) -> trt.ITensor:
        layer = self.net.add_shuffle(tensor)
        layer.reshape_dims = dims
        return layer.get_output(0)

    def linear(self, x: trt.ITensor, weight: np.ndarray) -> trt.ITensor:
        """``x @ weight.T`` for a ``[out, in]`` checkpoint weight."""
        rank = len(x.shape)
        w = self.const(weight, (1,) * (rank - 2) + tuple(weight.shape))
        return self.net.add_matrix_multiply(x, _NONE, w, _TRANSPOSE).get_output(0)

    def rms_norm(self, x: trt.ITensor, gamma: np.ndarray, eps: float, axis: int) -> trt.ITensor:
        """``gamma * (x / sqrt(mean(x^2) + eps)).to(dtype)`` with FP32 statistics."""
        rank = len(x.shape)
        xf = self.cast(x, trt.float32)
        mean = self.net.add_reduce(
            self.ew(xf, xf, _EW.PROD), trt.ReduceOperation.AVG, 1 << axis, keep_dims=True
        ).get_output(0)
        denom = self.unary(
            self.ew(mean, self.scalar(eps, rank, trt.float32), _EW.SUM), trt.UnaryOperation.SQRT
        )
        normalized = self.cast(self.ew(xf, denom, _EW.DIV), x.dtype)
        shape = [1] * rank
        shape[axis] = gamma.shape[0]
        return self.ew(normalized, self.const(gamma, tuple(shape)), _EW.PROD)

    def silu(self, x: trt.ITensor) -> trt.ITensor:
        sigmoid = self.net.add_activation(x, trt.ActivationType.SIGMOID).get_output(0)
        return self.ew(x, sigmoid, _EW.PROD)

    def softplus_f32(self, x: trt.ITensor) -> trt.ITensor:
        """``max(x, 0) + log(1 + exp(-|x|))`` in FP32 (overflow-free softplus)."""
        rank = len(x.shape)
        zero = self.scalar(0.0, rank, trt.float32)
        one = self.scalar(1.0, rank, trt.float32)
        negative_abs = self.unary(self.unary(x, trt.UnaryOperation.ABS), trt.UnaryOperation.NEG)
        tail = self.unary(
            self.ew(one, self.unary(negative_abs, trt.UnaryOperation.EXP), _EW.SUM),
            trt.UnaryOperation.LOG,
        )
        return self.ew(self.ew(x, zero, _EW.MAX), tail, _EW.SUM)


def _rows_to_heads(g: _Graph, x: trt.ITensor, heads: int, head_dim: int) -> trt.ITensor:
    """``[rows, heads * head_dim]`` -> ``[1, heads, rows, head_dim]``."""
    split = g.net.add_shuffle(x)
    split.reshape_dims = (-1, heads, head_dim)
    split.second_transpose = trt.Permutation([1, 0, 2])
    return g.reshape(split.get_output(0), (1, heads, -1, head_dim))


def _heads_to_rows(g: _Graph, x: trt.ITensor, width: int) -> trt.ITensor:
    """``[1, heads, rows, head_dim]`` -> ``[rows, heads * head_dim]``."""
    merge = g.net.add_shuffle(x)
    merge.first_transpose = trt.Permutation([0, 2, 1, 3])
    merge.reshape_dims = (-1, width)
    return merge.get_output(0)


def _rope(
    g: _Graph,
    x: trt.ITensor,
    heads: int,
    head_dim: int,
    tables: tuple[trt.ITensor, trt.ITensor],
    position_rows: trt.ITensor,
    rotary_dim: int,
) -> trt.ITensor:
    """Rotate-half RoPE on the first ``rotary_dim`` channels of every head."""
    layer = g.net.add_rotary_embedding(
        _rows_to_heads(g, x, heads, head_dim), tables[0], tables[1], False, rotary_dim
    )
    layer.set_input(3, position_rows)
    return _heads_to_rows(g, layer.get_output(0), heads * head_dim)


def _mask_4d(g: _Graph, mask: trt.ITensor) -> trt.ITensor:
    """``[rows, keys]`` -> ``[1, 1, rows, keys]``."""
    shape = g.net.add_shape(mask).get_output(0)
    ones = g.const(np.array([1, 1], dtype=np.int64))
    target = g.net.add_concatenation([ones, shape])
    target.axis = 0
    layer = g.net.add_shuffle(mask)
    layer.set_input(1, target.get_output(0))
    return layer.get_output(0)


def _sliding_window_mask(
    g: _Graph,
    mask: trt.ITensor,
    position_id: trt.ITensor,
    max_cache: int,
    window: int,
) -> trt.ITensor:
    """Hide keys more than ``window - 1`` positions behind each query row.

    Key column ``c < max_cache`` is cache row ``c`` (position ``c``); key column
    ``max_cache + j`` is new row ``j`` at ``position_id[j]``. Query row ``i``
    keeps keys at positions ``>= position_id[i] - (window - 1)``.
    """
    positions = g.cast(position_id, trt.float32)
    cache_positions = g.const(np.arange(max_cache, dtype=np.float32))
    keys = g.net.add_concatenation([cache_positions, positions])
    keys.axis = 0
    key_row = g.reshape(keys.get_output(0), (1, -1))
    query_column = g.reshape(positions, (-1, 1))
    oldest = g.ew(query_column, g.scalar(float(window - 1), 2, trt.float32), _EW.SUB)
    outside = g.ew(key_row, oldest, _EW.LESS)
    penalty = g.net.add_select(
        outside, g.scalar(_WINDOW_PENALTY, 2, trt.float32), g.scalar(0.0, 2, trt.float32)
    ).get_output(0)
    return g.ew(mask, g.cast(penalty, mask.dtype), _EW.SUM)


def _attention(
    g: _Graph,
    cfg: LagunaConfig,
    w: dict[str, np.ndarray],
    p: str,
    layer: int,
    normed: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    mask: trt.ITensor,
    position_rows: trt.ITensor,
    tables: dict[str, tuple[trt.ITensor, trt.ITensor]],
) -> tuple[trt.ITensor, trt.ITensor, trt.ITensor]:
    heads = cfg.heads_per_layer[layer]
    kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    width = heads * head_dim
    kind = cfg.layer_types[layer]

    q = g.linear(normed, w[f"{p}.q_proj"])
    k = g.linear(normed, w[f"{p}.k_proj"])
    v = g.linear(normed, w[f"{p}.v_proj"])
    q = g.reshape(
        g.rms_norm(g.reshape(q, (-1, heads, head_dim)), w[f"{p}.q_norm"], cfg.rms_norm_eps, 2),
        (-1, width),
    )
    k = g.reshape(
        g.rms_norm(g.reshape(k, (-1, kv_heads, head_dim)), w[f"{p}.k_norm"], cfg.rms_norm_eps, 2),
        (-1, cfg.kv_width),
    )
    rotary_dim = cfg.rope[kind].rotary_dim
    q = _rope(g, q, heads, head_dim, tables[kind], position_rows, rotary_dim)
    k = _rope(g, k, kv_heads, head_dim, tables[kind], position_rows, rotary_dim)

    keys = g.net.add_concatenation([cache_k, k])
    keys.axis = 0
    values = g.net.add_concatenation([cache_v, v])
    values.axis = 0
    # IAttention does not scale scores; scale Q in FP32 and round once.
    q_scaled = g.cast(
        g.ew(
            g.cast(_rows_to_heads(g, q, heads, head_dim), trt.float32),
            g.scalar(1.0 / math.sqrt(head_dim), 4, trt.float32),
            _EW.PROD,
        ),
        g.work,
    )
    attention = g.net.add_attention(
        q_scaled,
        _rows_to_heads(g, keys.get_output(0), kv_heads, head_dim),
        _rows_to_heads(g, values.get_output(0), kv_heads, head_dim),
        trt.AttentionNormalizationOp.SOFTMAX,
        False,
    )
    attention.decomposable = True
    attention.mask = _mask_4d(g, mask)
    context = _heads_to_rows(g, attention.get_output(0), width)

    # Softplus per-head output gate, computed in FP32 from the attention input.
    gate = g.softplus_f32(g.cast(g.linear(normed, w[f"{p}.g_proj"]), trt.float32))
    gate = g.reshape(g.cast(gate, g.work), (-1, heads, 1))
    gated = g.reshape(g.ew(g.reshape(context, (-1, heads, head_dim)), gate, _EW.PROD), (-1, width))
    return g.linear(gated, w[f"{p}.o_proj"]), k, v


def _swiglu(g: _Graph, x: trt.ITensor, gate: np.ndarray, up: np.ndarray, down: np.ndarray):
    hidden = g.ew(g.silu(g.linear(x, gate)), g.linear(x, up), _EW.PROD)
    return g.linear(hidden, down)


def _moe(
    g: _Graph, cfg: LagunaConfig, w: dict[str, np.ndarray], p: str, x: trt.ITensor
) -> trt.ITensor:
    experts = cfg.num_experts
    top_k = cfg.num_experts_per_tok

    # Router in FP32: sigmoid scores, top-k chosen on score + selection bias,
    # weights = chosen scores renormalised to sum to one.
    logits = g.linear(g.cast(x, trt.float32), w[f"{p}.router"])
    if cfg.router_logit_softcapping > 0.0:
        cap = g.scalar(cfg.router_logit_softcapping, 2, trt.float32)
        logits = g.ew(
            g.net.add_activation(g.ew(logits, cap, _EW.DIV), trt.ActivationType.TANH).get_output(0),
            cap,
            _EW.PROD,
        )
    scores = g.net.add_activation(logits, trt.ActivationType.SIGMOID).get_output(0)
    selection = g.ew(scores, g.const(w[f"{p}.router_bias"], (1, experts)), _EW.SUM)
    chosen = g.net.add_topk(selection, trt.TopKOperation.MAX, top_k, 1 << 1).get_output(1)
    picked = g.net.add_gather_v2(scores, chosen, trt.GatherMode.ELEMENT)
    picked.axis = 1
    weights = picked.get_output(0)
    total = g.net.add_reduce(weights, trt.ReduceOperation.SUM, 1 << 1, keep_dims=True)
    weights = g.cast(g.ew(weights, total.get_output(0), _EW.DIV), g.work)

    # Dense [rows, experts] routing weights: one_hot(chosen) * weights summed over k.
    expert_ids = g.const(np.arange(experts, dtype=np.int32).reshape(1, 1, experts))
    one_hot = g.cast(g.ew(g.reshape(chosen, (0, 0, 1)), expert_ids, _EW.EQUAL), g.work)
    dense = g.net.add_reduce(
        g.ew(one_hot, g.reshape(weights, (0, 0, 1)), _EW.PROD),
        trt.ReduceOperation.SUM,
        1 << 1,
        keep_dims=False,
    ).get_output(0)

    # All experts on all rows: [experts, rows, hidden].
    rows = g.ew(
        g.reshape(x, (1, -1, cfg.hidden_size)),
        g.cast(g.const(np.ones((experts, 1, 1), dtype=np.float32)), g.work),
        _EW.PROD,
    )
    gate = g.net.add_matrix_multiply(
        rows, _NONE, g.const(w[f"{p}.experts.gate_proj"]), _TRANSPOSE
    ).get_output(0)
    up = g.net.add_matrix_multiply(
        rows, _NONE, g.const(w[f"{p}.experts.up_proj"]), _TRANSPOSE
    ).get_output(0)
    expert_out = g.net.add_matrix_multiply(
        g.ew(g.silu(gate), up, _EW.PROD), _NONE, g.const(w[f"{p}.experts.down_proj"]), _TRANSPOSE
    ).get_output(0)
    dense_t = g.net.add_shuffle(dense)
    dense_t.first_transpose = trt.Permutation([1, 0])
    dense_t.reshape_dims = (0, 0, 1)
    routed = g.net.add_reduce(
        g.ew(expert_out, dense_t.get_output(0), _EW.PROD),
        trt.ReduceOperation.SUM,
        1 << 0,
        keep_dims=False,
    ).get_output(0)
    routed = g.ew(routed, g.scalar(cfg.routed_scaling_factor, 2, g.work), _EW.PROD)
    shared = _swiglu(
        g, x, w[f"{p}.shared.gate_proj"], w[f"{p}.shared.up_proj"], w[f"{p}.shared.down_proj"]
    )
    return g.ew(routed, shared, _EW.SUM)


def _last_row(g: _Graph, rows: trt.ITensor, width: int) -> trt.ITensor:
    shape = g.net.add_shape(rows).get_output(0)
    one_row = g.const(np.array([1, width], dtype=np.int64))
    start = g.ew(shape, one_row, _EW.SUB)
    layer = g.net.add_slice(rows, start=(0, 0), shape=(1, width), stride=(1, 1))
    layer.set_input(1, start)
    return layer.get_output(0)


def define_network(
    network: trt.INetworkDefinition,
    cfg: LagunaConfig,
    weights: LagunaWeights,
    max_cache: int,
) -> _Graph:
    """Populate ``network``; the returned graph must live until the engine is built."""
    work = trt.bfloat16 if weights.storage == BF16 else trt.float32
    g = _Graph(network, work)
    w = weights.tensors
    kv_width = cfg.kv_width

    token_id = network.add_input("token_id", trt.int32, (-1,))
    position_id = network.add_input("position_id", trt.int32, (-1,))
    mask = g.cast(network.add_input("attention_mask", trt.float32, (-1, -1)), work)
    position_rows = g.reshape(position_id, (1, -1))
    caches = [
        (
            network.add_input(layer_tensor_name("cache_k", i), work, (max_cache, kv_width)),
            network.add_input(layer_tensor_name("cache_v", i), work, (max_cache, kv_width)),
        )
        for i in range(cfg.num_hidden_layers)
    ]

    tables = {}
    for kind, spec in cfg.rope.items():
        cos, sin = rope_tables(spec, max_cache + 1)
        tables[kind] = (g.cast(g.const(cos), work), g.cast(g.const(sin), work))
    window_mask = None
    if SLIDING_ATTENTION in cfg.layer_types:
        window_mask = _sliding_window_mask(g, mask, position_id, max_cache, cfg.sliding_window)

    hidden = network.add_gather(g.const(w["embed"]), token_id, 0).get_output(0)
    for layer in range(cfg.num_hidden_layers):
        p = f"layer.{layer}"
        layer_mask = window_mask if cfg.layer_types[layer] == SLIDING_ATTENTION else mask
        normed = g.rms_norm(hidden, w[f"{p}.input_norm"], cfg.rms_norm_eps, 1)
        attn, present_k, present_v = _attention(
            g, cfg, w, p, layer, normed, *caches[layer], layer_mask, position_rows, tables
        )
        hidden = g.ew(hidden, attn, _EW.SUM)
        normed = g.rms_norm(hidden, w[f"{p}.post_attn_norm"], cfg.rms_norm_eps, 1)
        if cfg.mlp_layer_types[layer] == "dense":
            mlp = _swiglu(
                g, normed, w[f"{p}.mlp.gate_proj"], w[f"{p}.mlp.up_proj"], w[f"{p}.mlp.down_proj"]
            )
        else:
            mlp = _moe(g, cfg, w, p, normed)
        hidden = g.ew(hidden, mlp, _EW.SUM)
        present_k.name = layer_tensor_name("present_k", layer)
        present_v.name = layer_tensor_name("present_v", layer)
        network.mark_output(present_k)
        network.mark_output(present_v)

    hidden = g.rms_norm(_last_row(g, hidden, cfg.hidden_size), w["final_norm"], cfg.rms_norm_eps, 1)
    logits = g.cast(g.linear(hidden, w["lm_head"]), trt.float32)
    logits.name = "logits"
    network.mark_output(logits)
    return g


def add_profiles(builder: trt.Builder, config: trt.IBuilderConfig, max_cache: int) -> int:
    """Profile 0 prefills ``1..chunk`` rows; profile 1 decodes one row."""
    chunk = min(PREFILL_MAX_ROWS, max_cache)
    optimum = min(PREFILL_OPT_ROWS, chunk)
    for opt_rows, max_rows in ((optimum, chunk), (1, 1)):
        profile = builder.create_optimization_profile()
        for name in ("token_id", "position_id"):
            profile.set_shape(name, (1,), (opt_rows,), (max_rows,))
        profile.set_shape(
            "attention_mask",
            (1, max_cache + 1),
            (opt_rows, max_cache + opt_rows),
            (max_rows, max_cache + max_rows),
        )
        config.add_optimization_profile(profile)
    return chunk
