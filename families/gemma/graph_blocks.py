# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Composable architectural building blocks for TRT engine construction.

Layer 2 in the three-layer builder stack:

    graph_ops.py        Layer 1: Atomic TRT operations (tensor-in/tensor-out)
        |
    graph_blocks.py     Layer 2: Composable blocks (weight-aware)  <- THIS FILE
        |
    builders / plugins  Layer 3: Full engine assembly

Each block composes multiple graph_ops into a reusable sub-structure
(full attention block, SwiGLU MLP, GELU MLP, norm dispatch). Functions
accept a ``weights`` dict + ``prefix`` string to resolve weight names.

Blocks do NOT apply residual connections. Callers compose the residual
pattern, which is what varies across architectures (sequential vs parallel
residual, DeepStack injection, MoE routing, etc.).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from . import graph_ops


if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from typing import Any as QuantContext


# ---------------------------------------------------------------------------
# Precision boundary helpers (used by standard_decoder_builder, not inside
# blocks themselves).
# ---------------------------------------------------------------------------

def make_matmul_fn(network, dtype, quant_ctx):
    """Create a matmul callable that routes through quant_ctx if present.

    Returns a function: (lhs, lhs_w, rhs_w, rhs_weights, weight_name) -> ITensor
    """
    if quant_ctx is None:
        def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
            return graph_ops.add_matmul_rhs_constant(
                network, lhs, lhs_w, rhs_w, rhs_weights, dtype=dtype)
        return matmul
    else:
        def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
            return quant_ctx.maybe_quantized_matmul(
                network, lhs, lhs_w, rhs_w, rhs_weights, weight_name,
                dtype=dtype)
        return matmul


_make_matmul_fn = make_matmul_fn


def infer_kv_attention_size(
    weights: dict,
    *,
    prefix: str = "layer.0",
    num_kv_heads: int,
    head_dim: int,
) -> int:
    """Validate and return the compact K/V row width."""
    expected = int(num_kv_heads * head_dim)
    explicit = weights.get("_kv_attention_size")
    if explicit is not None and int(explicit) != expected:
        raise ValueError(
            f"Compact K/V cache width must be num_kv_heads * head_dim "
            f"({expected}), got _kv_attention_size={int(explicit)}")
    w_k = weights.get(f"{prefix}.w_k")
    if isinstance(w_k, np.ndarray) and w_k.ndim == 2:
        actual = int(w_k.shape[1])
        if actual != expected:
            raise ValueError(
                f"{prefix}.w_k must use compact K/V width {expected}, "
                f"got {actual}")
    return expected


def apply_norm(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    hidden_size: int,
    gamma: np.ndarray,
    beta: np.ndarray | None,
    eps_tensor: trt.ITensor,
    norm_type: str,
    dtype: np.dtype = np.float32,
    eps: float | None = None,
) -> trt.ITensor:
    """Dispatch to RMSNorm or LayerNorm based on norm_type."""
    if norm_type == "layernorm":
        if beta is None:
            beta = np.zeros(hidden_size, dtype=np.float32)
        if eps is not None:
            return graph_ops.add_layer_norm_native(
                network, inp, hidden_size, gamma, beta, eps, dtype=dtype)
        # Native INormalizationLayer requires a build-time scalar epsilon.
        # Some callers only pass epsilon as an ITensor, so keep the manual
        # explicit graph until those builders thread the scalar too.
        return graph_ops.add_layer_norm(
            network, inp, hidden_size, gamma, beta, eps_tensor, dtype=dtype)
    else:
        return graph_ops.add_rms_norm(
            network, inp, hidden_size, gamma, eps_tensor, dtype=dtype)


def add_attention_block(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    position_id: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    attention_size: int,
    num_heads: int,
    head_dim: int,
    max_cache_length: int,
    eps_tensor: trt.ITensor,
    kv_attention_size: int | None = None,
    num_kv_heads: int | None = None,
    attention_scale: float | None = None,
    eps: float | None = None,
    norm_type: str = "rmsnorm",
    position_type: str = "rope",
    alibi_slopes_tensor: trt.ITensor | None = None,
    alibi_indices_tensor: trt.ITensor | None = None,
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    layer_prefix: str = "",
    # TRT 10 native API tensors.
    cos_half_tensor: trt.ITensor | None = None,
    sin_half_tensor: trt.ITensor | None = None,
    rotary_embedding_dim: int = 0,
    interleaved_rope: bool = False,
) -> dict[str, trt.ITensor]:
    """Pre-norm -> QKV -> RoPE -> cache concat -> attention -> output proj.

    Returns {"normed": ..., "attn_out": ..., "present_k": ..., "present_v": ...}.
    Does NOT apply residual -- callers compose the residual pattern.

    This function uses TRT 10 native APIs for the basic transformer primitives:
      - IRotaryEmbeddingLayer for RoPE
      - IAttention for scaled dot-product attention
    ALiBi is represented as a per-head additive attention mask and still uses
    native IAttention.
    """
    matmul = _make_matmul_fn(network, dtype, quant_ctx)
    attention_window = max_cache_length + 1
    if num_kv_heads is None:
        num_kv_heads = num_heads
    if kv_attention_size is None:
        kv_attention_size = num_kv_heads * head_dim

    # Weight name for quant scale lookup — use layer_prefix if provided,
    # otherwise fall back to the weights-dict prefix.
    _lp = layer_prefix or prefix

    # Pre-attention norm
    normed = apply_norm(
        network, hidden, hidden_size,
        weights[f"{prefix}.input_norm"],
        weights.get(f"{prefix}.input_norm_beta"),
        eps_tensor, norm_type, dtype=dtype, eps=eps)

    # QKV projections
    q = matmul(normed, hidden_size, attention_size,
               weights[f"{prefix}.w_q"], f"{_lp}.w_q")
    k = matmul(normed, hidden_size, kv_attention_size,
               weights[f"{prefix}.w_k"], f"{_lp}.w_k")
    v = matmul(normed, hidden_size, kv_attention_size,
               weights[f"{prefix}.w_v"], f"{_lp}.w_v")

    # Optional QKV biases
    q_bias = weights.get(f"{prefix}.q_bias")
    if q_bias is not None:
        q = graph_ops.add_bias_sum(network, q, attention_size, q_bias, dtype=dtype)
    k_bias = weights.get(f"{prefix}.k_bias")
    if k_bias is not None:
        k = graph_ops.add_bias_sum(network, k, kv_attention_size, k_bias, dtype=dtype)
    v_bias = weights.get(f"{prefix}.v_bias")
    if v_bias is not None:
        v = graph_ops.add_bias_sum(network, v, kv_attention_size, v_bias, dtype=dtype)

    # Optional per-head q/k norm
    q_norm = weights.get(f"{prefix}.q_norm")
    if q_norm is not None:
        q = graph_ops.add_rms_norm_per_head(
            network, q, num_heads, head_dim, q_norm, eps_tensor, dtype=dtype)
    k_norm = weights.get(f"{prefix}.k_norm")
    if k_norm is not None:
        k = graph_ops.add_rms_norm_per_head(
            network, k, num_kv_heads, head_dim, k_norm, eps_tensor, dtype=dtype)

    # ------------------------------------------------------------------ #
    # RoPE via native IRotaryEmbeddingLayer                              #
    # ------------------------------------------------------------------ #

    if position_type == "rope":
        if cos_half_tensor is None or sin_half_tensor is None:
            raise ValueError(
                "RoPE attention requires half-dimension cos/sin tensors for "
                "TRT native IRotaryEmbeddingLayer")
        rope_dim = rotary_embedding_dim or head_dim
        rope_dim = graph_ops.validate_native_rope_dim(rope_dim)
        q = graph_ops.add_apply_rope_native(
            network, q, num_heads, head_dim,
            cos_half_tensor, sin_half_tensor, position_id,
            rope_dim, interleaved_rope)
        k = graph_ops.add_apply_rope_native(
            network, k, num_kv_heads, head_dim,
            cos_half_tensor, sin_half_tensor, position_id,
            rope_dim, interleaved_rope)

    # Save present K/V (before concatenation, this is the raw projection output)
    present_k = k
    present_v = v

    # Reshape current K, V for concatenation
    k_reshape = network.add_shuffle(k)
    k_reshape.reshape_dims = (1, kv_attention_size)
    v_reshape = network.add_shuffle(v)
    v_reshape.reshape_dims = (1, kv_attention_size)

    # Concatenate with cache
    all_k = network.add_concatenation(
        [cache_k, k_reshape.get_output(0)])
    all_k.axis = 0
    all_v = network.add_concatenation(
        [cache_v, v_reshape.get_output(0)])
    all_v.axis = 0

    # ------------------------------------------------------------------ #
    # Attention core — native IAttention                                  #
    # ------------------------------------------------------------------ #
    kv_seq = attention_window
    if alibi_slopes_tensor is not None:
        if alibi_indices_tensor is None:
            raise ValueError("ALiBi attention requires cache position indices")
        mask_4d = graph_ops.add_alibi_mask_4d(
            network,
            attention_mask,
            position_id,
            alibi_slopes_tensor,
            alibi_indices_tensor,
            num_heads,
        )
    else:
        mask_4d = graph_ops.add_2d_mask_to_4d(network, attention_mask)

    context = graph_ops.add_attention_from_rows(
        network,
        q,
        all_k.get_output(0),
        all_v.get_output(0),
        num_heads=num_heads,
        head_dim=head_dim,
        num_kv_heads=num_kv_heads,
        q_seq=1,
        kv_seq=kv_seq,
        causal=False,
        mask=mask_4d,
        scale=attention_scale,
    )

    # Output projection
    attn_out = matmul(context,
                      attention_size, hidden_size,
                      weights[f"{prefix}.w_o"], f"{_lp}.w_o")

    # Optional output projection bias
    o_bias = weights.get(f"{prefix}.o_bias")
    if o_bias is not None:
        attn_out = graph_ops.add_bias_sum(network, attn_out, hidden_size, o_bias, dtype=dtype)

    return {
        "normed": normed,
        "attn_out": attn_out,
        "present_k": present_k,
        "present_v": present_v,
    }


def add_swiglu_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    mlp_size: int,
    activation: str = "silu",
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    layer_prefix: str = "",
) -> trt.ITensor:
    """Gate/up/down MLP with a checkpoint-selected gate activation."""
    matmul = _make_matmul_fn(network, dtype, quant_ctx)
    _lp = layer_prefix or prefix

    gate = matmul(inp, hidden_size, mlp_size,
                  weights[f"{prefix}.w_gate"], f"{_lp}.w_gate")
    up = matmul(inp, hidden_size, mlp_size,
                weights[f"{prefix}.w_up"], f"{_lp}.w_up")

    activated = graph_ops.add_activation(
        network, gate, activation, dtype=dtype)
    gated = network.add_elementwise(
        activated, up, trt.ElementWiseOperation.PROD)

    mlp_out = matmul(gated.get_output(0), mlp_size, hidden_size,
                     weights[f"{prefix}.w_down"], f"{_lp}.w_down")
    return mlp_out


def add_gelu_fc_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    mlp_size: int,
    activation: str = "gelu_new",
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    layer_prefix: str = "",
) -> trt.ITensor:
    """fc1 -> activation -> fc2 MLP. Returns output tensor."""
    matmul = _make_matmul_fn(network, dtype, quant_ctx)
    _lp = layer_prefix or prefix

    fc1 = matmul(inp, hidden_size, mlp_size,
                 weights[f"{prefix}.w_fc1"], f"{_lp}.w_fc1")
    fc1_bias = weights.get(f"{prefix}.fc1_bias")
    if fc1_bias is not None:
        fc1 = graph_ops.add_bias_sum(network, fc1, mlp_size, fc1_bias, dtype=dtype)

    activated = graph_ops.add_activation(network, fc1, activation, dtype=dtype)

    fc2 = matmul(activated, mlp_size, hidden_size,
                 weights[f"{prefix}.w_fc2"], f"{_lp}.w_fc2")
    fc2_bias = weights.get(f"{prefix}.fc2_bias")
    if fc2_bias is not None:
        fc2 = graph_ops.add_bias_sum(network, fc2, hidden_size, fc2_bias, dtype=dtype)

    return fc2



def _gemma_raw(config) -> dict:
    """The decoder's own config fields.

    Gemma 3 at 4B and above nests them under ``text_config`` beside a vision
    config; the 1B and Gemma 2 state them at the top level. Nested values win,
    matching how families/gemma/config.py merges them.
    """
    raw = getattr(config, "raw", {}) or {}
    nested = raw.get("text_config")
    if isinstance(nested, dict):
        return {**raw, **nested}
    return raw


_GEMMA3_MODEL_TYPES = frozenset({"gemma3", "gemma3_text"})
_GEMMA3_SLIDING_PATTERN = 6


def gemma3_attention_schedule(config, num_layers: int) -> dict:
    """Which layers attend locally, and what rope base each one uses.

    Gemma 3 interleaves local and global attention: with
    ``sliding_window_pattern: 6``, five layers in six attend only to the last
    ``sliding_window`` tokens and every sixth attends to everything. The two
    kinds also use different rope bases - ``rope_local_base_freq`` (10000) for
    the local layers against ``rope_theta`` (1000000) for the global ones.

    Gemma and Gemma 2 declare no pattern, so every layer comes back global and
    the caller builds exactly the graph it built before.
    """
    raw = _gemma_raw(config)
    window = raw.get("sliding_window")
    local_base = raw.get("rope_local_base_freq")
    # A second rope base is what distinguishes Gemma 3 from Gemma 2 here.
    # Gemma 2 also interleaves windows but rotates every layer on one base, so
    # without this key the caller must keep building a single-base graph.
    #
    # For Gemma 3 that fallback is never right: it silently rebuilds all layers
    # as global attention on one base, which builds cleanly and generates
    # fluent, wrong text. google/gemma-3-12b-it states neither key and relies on
    # the transformers defaults, so refuse rather than guess - the model layer
    # is responsible for filling these in before the graph is built.
    if str(getattr(config, "model_type", "")).lower() in _GEMMA3_MODEL_TYPES and not (
        window and local_base
    ):
        raise ValueError(
            "Gemma 3 needs both sliding_window and rope_local_base_freq to place "
            f"local attention; got sliding_window={window!r} "
            f"rope_local_base_freq={local_base!r}"
        )
    if not window or not local_base:
        return {
            "is_local": [False] * num_layers,
            "window": None,
            "local_theta": float(config.rope_theta),
        }
    # An explicit layer_types list is the checkpoint stating the schedule
    # outright, so it wins over anything derived. gemma-3-270m-it ships one.
    layer_types = raw.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        if len(layer_types) != num_layers:
            raise ValueError(
                f"Gemma layer_types lists {len(layer_types)} entries for "
                f"{num_layers} layers"
            )
        unknown = sorted({str(kind) for kind in layer_types} - {"sliding_attention", "full_attention"})
        if unknown:
            raise ValueError(f"Gemma layer_types has unsupported entries: {unknown}")
        return {
            "is_local": [str(kind) == "sliding_attention" for kind in layer_types],
            "window": int(window),
            "local_theta": float(local_base),
        }
    # Otherwise derive it. google/gemma-3-1b-it states sliding_window_pattern;
    # google/gemma-3-270m-it omits it and relies on the transformers default of
    # 6. Treating the absent key as "no schedule" would build every layer
    # global and be wrong without failing.
    declared = raw.get("sliding_window_pattern")
    pattern = _GEMMA3_SLIDING_PATTERN if declared is None else int(declared)
    if pattern < 1:
        raise ValueError(f"Gemma sliding_window_pattern must be positive, got {pattern}")
    # transformers: is_sliding = bool((layer_idx + 1) % sliding_window_pattern),
    # so every pattern-th layer is the global one.
    return {
        "is_local": [bool((index + 1) % pattern) for index in range(num_layers)],
        "window": int(window),
        "local_theta": float(local_base),
    }


def gemma_attention_scale(config, head_dim: int) -> float:
    """Gemma scales queries by a declared constant, not always by head_dim.

    ``query_pre_attn_scalar`` matches ``head_dim`` on some widths and not on
    others, so it has to be read rather than inferred.
    """
    raw = _gemma_raw(config)
    scalar = raw.get("query_pre_attn_scalar")
    if scalar is None:
        scalar = head_dim
    scalar = float(scalar)
    if scalar <= 0.0:
        raise ValueError(f"Gemma query_pre_attn_scalar must be positive, got {scalar}")
    return 1.0 / (scalar**0.5)


def gemma_global_rope_position_scale(config) -> float:
    """Linear rope scaling for the global layers, as a position multiplier.

    Gemma 3 at 4B and above declares ``rope_scaling`` of type ``linear`` with a
    factor. transformers resolves that per attention type: the sliding layers
    stay unscaled on ``rope_local_base_freq`` and only the full-attention layers
    are scaled. A factor of f divides the position by f, so the multiplier is
    ``1 / f``. Checkpoints without the key, including gemma-3-1b-it, come back
    1.0 and build the table they built before.
    """
    raw = _gemma_raw(config)
    scaling = raw.get("rope_scaling")
    if not isinstance(scaling, dict):
        return 1.0
    kind = str(scaling.get("rope_type") or scaling.get("type") or "").lower()
    if kind in ("", "default"):
        return 1.0
    if kind != "linear":
        raise NotImplementedError(
            f"gemma supports only linear rope scaling, got rope_type={kind!r}"
        )
    factor = float(scaling.get("factor", 1.0))
    if factor <= 0.0:
        raise ValueError(f"gemma rope_scaling factor must be positive, got {factor}")
    return 1.0 / factor
