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
from tensorrt_model_connect import trt_compat

from . import graph_ops

trt = trt_compat.get_trt()

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict
    from ...quantization.context import QuantContext


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
                network, lhs, lhs_w, rhs_w, rhs_weights, dtype=dtype
            )

        return matmul
    else:

        def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
            return quant_ctx.maybe_quantized_matmul(
                network, lhs, lhs_w, rhs_w, rhs_weights, weight_name, dtype=dtype
            )

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
            f"({expected}), got _kv_attention_size={int(explicit)}"
        )
    w_k = weights.get(f"{prefix}.w_k")
    if isinstance(w_k, np.ndarray) and w_k.ndim == 2:
        actual = int(w_k.shape[1])
        if actual != expected:
            raise ValueError(f"{prefix}.w_k must use compact K/V width {expected}, got {actual}")
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
                network, inp, hidden_size, gamma, beta, eps, dtype=dtype
            )
        # Native INormalizationLayer requires a build-time scalar epsilon.
        # Some callers only pass epsilon as an ITensor, so keep the manual
        # shared fallback until those builders thread the scalar too.
        return graph_ops.add_layer_norm(
            network, inp, hidden_size, gamma, beta, eps_tensor, dtype=dtype
        )
    else:
        return graph_ops.add_rms_norm(network, inp, hidden_size, gamma, eps_tensor, dtype=dtype)


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
    alibi_slopes_tensor: trt.ITensor | None = None,
    alibi_indices_tensor: trt.ITensor | None = None,
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    layer_prefix: str = "",
    cos_half_tensor: trt.ITensor | None = None,
    sin_half_tensor: trt.ITensor | None = None,
    rotary_embedding_dim: int = 0,
    interleaved_rope: bool = False,
    ffi_attention_kernel: str | None = None,
    dynamic_kv_cache: bool = False,
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
    _lp = layer_prefix or prefix
    normed = apply_norm(
        network,
        hidden,
        hidden_size,
        weights[f"{prefix}.input_norm"],
        weights.get(f"{prefix}.input_norm_beta"),
        eps_tensor,
        "layernorm",
        dtype=dtype,
        eps=eps,
    )
    q = matmul(normed, hidden_size, attention_size, weights[f"{prefix}.w_q"], f"{_lp}.w_q")
    k = matmul(normed, hidden_size, kv_attention_size, weights[f"{prefix}.w_k"], f"{_lp}.w_k")
    v = matmul(normed, hidden_size, kv_attention_size, weights[f"{prefix}.w_v"], f"{_lp}.w_v")
    q_bias = weights.get(f"{prefix}.q_bias")
    if q_bias is not None:
        q = graph_ops.add_bias_sum(network, q, attention_size, q_bias, dtype=dtype)
    k_bias = weights.get(f"{prefix}.k_bias")
    if k_bias is not None:
        k = graph_ops.add_bias_sum(network, k, kv_attention_size, k_bias, dtype=dtype)
    v_bias = weights.get(f"{prefix}.v_bias")
    if v_bias is not None:
        v = graph_ops.add_bias_sum(network, v, kv_attention_size, v_bias, dtype=dtype)
    q_norm = weights.get(f"{prefix}.q_norm")
    if q_norm is not None:
        q = graph_ops.add_rms_norm_per_head(
            network, q, num_heads, head_dim, q_norm, eps_tensor, dtype=dtype
        )
    k_norm = weights.get(f"{prefix}.k_norm")
    if k_norm is not None:
        k = graph_ops.add_rms_norm_per_head(
            network, k, num_kv_heads, head_dim, k_norm, eps_tensor, dtype=dtype
        )
    use_native_attention = ffi_attention_kernel is None
    present_k = k
    present_v = v
    k_reshape = network.add_shuffle(k)
    k_reshape.reshape_dims = (1, kv_attention_size)
    v_reshape = network.add_shuffle(v)
    v_reshape.reshape_dims = (1, kv_attention_size)
    all_k = network.add_concatenation([cache_k, k_reshape.get_output(0)])
    all_k.axis = 0
    all_v = network.add_concatenation([cache_v, v_reshape.get_output(0)])
    all_v.axis = 0
    if use_native_attention:
        kv_seq = None if dynamic_kv_cache else attention_window
        if alibi_slopes_tensor is not None:
            if alibi_indices_tensor is None:
                raise ValueError("ALiBi attention requires cache position indices")
            if dynamic_kv_cache:
                raise ValueError("dynamic_kv_cache is not supported for ALiBi attention")
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
            fp32_accumulation=dtype != np.float32,
        )
    elif ffi_attention_kernel is not None:
        if num_kv_heads != num_heads:
            raise ValueError(
                "FFI decoder attention requires num_kv_heads == num_heads; use TRT native attention for compact GQA/MQA KV cache"
            )
        context = graph_ops.add_decoder_attention_ffi(
            network,
            q,
            all_k.get_output(0),
            all_v.get_output(0),
            kernel_name=ffi_attention_kernel,
            num_heads=num_heads,
            head_dim=head_dim,
            attention_window=attention_window,
        )
    attn_out = matmul(context, attention_size, hidden_size, weights[f"{prefix}.w_o"], f"{_lp}.w_o")
    o_bias = weights.get(f"{prefix}.o_bias")
    if o_bias is not None:
        attn_out = graph_ops.add_bias_sum(network, attn_out, hidden_size, o_bias, dtype=dtype)
    return {"normed": normed, "attn_out": attn_out, "present_k": present_k, "present_v": present_v}


def add_gelu_fc_mlp(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    mlp_size: int,
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    layer_prefix: str = "",
) -> trt.ITensor:
    """fc1 -> activation -> fc2 MLP. Returns output tensor."""
    matmul = _make_matmul_fn(network, dtype, quant_ctx)
    _lp = layer_prefix or prefix
    fc1 = matmul(inp, hidden_size, mlp_size, weights[f"{prefix}.w_fc1"], f"{_lp}.w_fc1")
    fc1_bias = weights.get(f"{prefix}.fc1_bias")
    if fc1_bias is not None:
        fc1 = graph_ops.add_bias_sum(network, fc1, mlp_size, fc1_bias, dtype=dtype)
    activated = graph_ops.add_activation(network, fc1, "gelu_new", dtype=dtype)
    fc2 = matmul(activated, mlp_size, hidden_size, weights[f"{prefix}.w_fc2"], f"{_lp}.w_fc2")
    fc2_bias = weights.get(f"{prefix}.fc2_bias")
    if fc2_bias is not None:
        fc2 = graph_ops.add_bias_sum(network, fc2, hidden_size, fc2_bias, dtype=dtype)
    return fc2
