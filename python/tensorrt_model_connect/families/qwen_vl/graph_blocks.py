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
    from .lora import DynamicLoraConfig
    from ...quantization.context import QuantContext


# ---------------------------------------------------------------------------
# Precision boundary helpers (used by standard_decoder_builder, not inside
# blocks themselves).
# ---------------------------------------------------------------------------

def make_matmul_fn(network, dtype, quant_ctx, lora_config=None):
    """Create a matmul callable that routes through quant_ctx if present.

    Returns a function: (lhs, lhs_w, rhs_w, rhs_weights, weight_name) -> ITensor
    """
    def matmul(lhs, lhs_w, rhs_w, rhs_weights, weight_name):
        if quant_ctx is None:
            base = graph_ops.add_matmul_rhs_constant(
                network, lhs, lhs_w, rhs_w, rhs_weights, dtype=dtype)
        else:
            base = quant_ctx.maybe_quantized_matmul(
                network, lhs, lhs_w, rhs_w, rhs_weights, weight_name,
                dtype=dtype)

        if lora_config is None or not lora_config.targets_weight(weight_name):
            return base

        a_name, b_name = lora_config.input_names(weight_name)
        lora_a = network.add_input(
            a_name, lhs.dtype, (lhs_w, lora_config.max_rank))
        lora_b = network.add_input(
            b_name, lhs.dtype, (lora_config.max_rank, rhs_w))
        if lora_a is None or lora_b is None:
            raise RuntimeError(
                f"Failed to add dynamic LoRA inputs for projection {weight_name}")
        delta = graph_ops.add_lora_delta(network, lhs, lora_a, lora_b)
        return network.add_elementwise(
            base, delta, trt.ElementWiseOperation.SUM).get_output(0)

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


def cast_to_dtype(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
    target_dtype: trt.DataType,
) -> trt.ITensor:
    """Cast tensor to target dtype (no-op if already matching)."""
    if tensor.dtype == target_dtype:
        return tensor
    return network.add_cast(tensor, target_dtype).get_output(0)


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
        # shared fallback until those builders thread the scalar too.
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
    *,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    attention_size: int,
    num_heads: int,
    head_dim: int,
    eps_tensor: trt.ITensor,
    cos_half_tensor: trt.ITensor,
    sin_half_tensor: trt.ITensor,
    cache_write_indices: trt.ITensor,
    key_value_lengths: trt.ITensor,
    kv_attention_size: int | None = None,
    num_kv_heads: int | None = None,
    attention_scale: float | None = None,
    eps: float | None = None,
    norm_type: str = "rmsnorm",
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    layer_prefix: str = "",
    rotary_embedding_dim: int = 0,
    sequence_length: int | None = 1,
    lora_config: DynamicLoraConfig | None = None,
    recipe_instance: str | None = None,
) -> dict[str, trt.ITensor]:
    """Build native Qwen-VL attention with runtime-owned full-capacity KV.

    Returns {"normed": ..., "attn_out": ..., "present_k": ..., "present_v": ...}.
    Does NOT apply residual -- callers compose the residual pattern.
    """
    matmul = _make_matmul_fn(network, dtype, quant_ctx, lora_config)
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
            network, q, num_heads, head_dim, q_norm, eps_tensor, dtype=dtype,
            sequence_length=sequence_length)
    k_norm = weights.get(f"{prefix}.k_norm")
    if k_norm is not None:
        k = graph_ops.add_rms_norm_per_head(
            network, k, num_kv_heads, head_dim, k_norm, eps_tensor, dtype=dtype,
            sequence_length=sequence_length)

    rope_dim = graph_ops.validate_native_rope_dim(
        rotary_embedding_dim or head_dim)
    q = graph_ops.add_apply_rope_native_sequence(
        network, q, num_heads, head_dim,
        cos_half_tensor, sin_half_tensor, rope_dim,
        interleaved=False, sequence_length=sequence_length)
    k = graph_ops.add_apply_rope_native_sequence(
        network, k, num_kv_heads, head_dim,
        cos_half_tensor, sin_half_tensor, rope_dim,
        interleaved=False, sequence_length=sequence_length)

    native_attention = graph_ops.add_native_kv_cache_attention_from_rows(
        network,
        q,
        k,
        v,
        cache_k,
        cache_v,
        cache_write_indices,
        key_value_lengths,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_seq=sequence_length,
        scale=attention_scale,
        tag=f"{prefix}.attn",
        recipe_instance=recipe_instance,
    )
    context = native_attention["context"]
    present_k = native_attention["present_k"]
    present_v = native_attention["present_v"]

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
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    layer_prefix: str = "",
    lora_config: DynamicLoraConfig | None = None,
) -> trt.ITensor:
    """Gate/up/down SwiGLU MLP. Returns output tensor."""
    matmul = _make_matmul_fn(network, dtype, quant_ctx, lora_config)
    _lp = layer_prefix or prefix

    gate = matmul(inp, hidden_size, mlp_size,
                  weights[f"{prefix}.w_gate"], f"{_lp}.w_gate")
    up = matmul(inp, hidden_size, mlp_size,
                weights[f"{prefix}.w_up"], f"{_lp}.w_up")

    sigmoid = network.add_activation(gate, trt.ActivationType.SIGMOID)
    swish = network.add_elementwise(
        gate, sigmoid.get_output(0), trt.ElementWiseOperation.PROD)
    gated = network.add_elementwise(
        swish.get_output(0), up, trt.ElementWiseOperation.PROD)

    mlp_out = matmul(gated.get_output(0), mlp_size, hidden_size,
                     weights[f"{prefix}.w_down"], f"{_lp}.w_down")
    return mlp_out
