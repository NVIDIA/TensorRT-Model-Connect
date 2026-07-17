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


def cast_to_fp32(
    network: trt.INetworkDefinition,
    tensor: trt.ITensor,
) -> trt.ITensor:
    """Cast tensor to FP32 for numerically sensitive ops."""
    if tensor.dtype == trt.float32:
        return tensor
    return network.add_cast(tensor, trt.float32).get_output(0)


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
    ffi_attention_kernel: str | None = None,
    dynamic_kv_cache: bool = False,
    sequence_length: int | None = 1,
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
            network, q, num_heads, head_dim, q_norm, eps_tensor, dtype=dtype,
            sequence_length=sequence_length)
    k_norm = weights.get(f"{prefix}.k_norm")
    if k_norm is not None:
        k = graph_ops.add_rms_norm_per_head(
            network, k, num_kv_heads, head_dim, k_norm, eps_tensor, dtype=dtype,
            sequence_length=sequence_length)

    # ------------------------------------------------------------------ #
    # RoPE via native IRotaryEmbeddingLayer                              #
    # ------------------------------------------------------------------ #
    use_native_attention = ffi_attention_kernel is None

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
            rope_dim, interleaved_rope, sequence_length=sequence_length)
        k = graph_ops.add_apply_rope_native(
            network, k, num_kv_heads, head_dim,
            cos_half_tensor, sin_half_tensor, position_id,
            rope_dim, interleaved_rope, sequence_length=sequence_length)

    # Save present K/V (before concatenation, this is the raw projection output)
    present_k = k
    present_v = v

    # Reshape current K, V for concatenation
    current_k = k
    current_v = v
    if sequence_length is not None:
        k_reshape = network.add_shuffle(k)
        k_reshape.reshape_dims = (sequence_length, kv_attention_size)
        v_reshape = network.add_shuffle(v)
        v_reshape.reshape_dims = (sequence_length, kv_attention_size)
        current_k = k_reshape.get_output(0)
        current_v = v_reshape.get_output(0)

    # Concatenate with cache
    all_k = network.add_concatenation(
        [cache_k, current_k])
    all_k.axis = 0
    all_v = network.add_concatenation(
        [cache_v, current_v])
    all_v.axis = 0

    # ------------------------------------------------------------------ #
    # Attention core — native IAttention or FFI kernel                    #
    # ------------------------------------------------------------------ #
    if use_native_attention:
        kv_seq = None if dynamic_kv_cache or sequence_length is None else attention_window
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
            q_seq=sequence_length,
            kv_seq=kv_seq,
            causal=False,
            mask=mask_4d,
            scale=attention_scale,
            explicit_attention=bool(weights.get("_explicit_attention", False)),
        )
    elif ffi_attention_kernel is not None:
        if num_kv_heads != num_heads:
            raise ValueError(
                "FFI decoder attention requires num_kv_heads == num_heads; "
                "use TRT native attention for compact GQA/MQA KV cache")
        # Fused attention kernel via TVM-FFI plugin
        context = graph_ops.add_decoder_attention_ffi(
            network, q, all_k.get_output(0), all_v.get_output(0),
            kernel_name=ffi_attention_kernel,
            num_heads=num_heads, head_dim=head_dim,
            attention_window=attention_window)

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
) -> trt.ITensor:
    """Gate/up/down SwiGLU MLP. Returns output tensor."""
    matmul = _make_matmul_fn(network, dtype, quant_ctx)
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


# ---------------------------------------------------------------------------
# Diffusion building blocks
# ---------------------------------------------------------------------------

def add_gated_mlp(
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
    """Gated MLP: activation(fc1(x)) * fc1_gate(x), then fc2.

    Used by T5 encoder (gated GELU) and DiT FFN. Two parallel projections
    where one is gated by activation.

    Weight keys: {prefix}.w_fc1, {prefix}.w_fc1_gate, {prefix}.w_fc2
    Optional: {prefix}.fc1_bias, {prefix}.fc1_gate_bias, {prefix}.fc2_bias
    """
    matmul = _make_matmul_fn(network, dtype, quant_ctx)
    _lp = layer_prefix or prefix

    # Two parallel projections
    fc1 = matmul(inp, hidden_size, mlp_size,
                 weights[f"{prefix}.w_fc1"], f"{_lp}.w_fc1")
    fc1_bias = weights.get(f"{prefix}.fc1_bias")
    if fc1_bias is not None:
        fc1 = graph_ops.add_bias_sum(network, fc1, mlp_size, fc1_bias, dtype=dtype)

    fc1_gate = matmul(inp, hidden_size, mlp_size,
                      weights[f"{prefix}.w_fc1_gate"], f"{_lp}.w_fc1_gate")
    fc1_gate_bias = weights.get(f"{prefix}.fc1_gate_bias")
    if fc1_gate_bias is not None:
        fc1_gate = graph_ops.add_bias_sum(
            network, fc1_gate, mlp_size, fc1_gate_bias, dtype=dtype)

    # Gate: activation(fc1) * fc1_gate
    activated = graph_ops.add_activation(network, fc1, activation, dtype=dtype)
    gated = network.add_elementwise(
        activated, fc1_gate, trt.ElementWiseOperation.PROD)

    # Output projection
    fc2 = matmul(gated.get_output(0), mlp_size, hidden_size,
                 weights[f"{prefix}.w_fc2"], f"{_lp}.w_fc2")
    fc2_bias = weights.get(f"{prefix}.fc2_bias")
    if fc2_bias is not None:
        fc2 = graph_ops.add_bias_sum(network, fc2, hidden_size, fc2_bias, dtype=dtype)

    return fc2


def add_dit_block(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    context: trt.ITensor,
    adaln_params: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    context_dim: int,
    num_heads: int,
    q_seq_len: int,
    kv_seq_len: int,
    mlp_size: int,
    eps: float = 1e-5,
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    layer_prefix: str = "",
) -> trt.ITensor:
    """DiT block: AdaLN self-attention + cross-attention + AdaLN FFN.

    adaln_params: [1, 6 * hidden_size] — from timestep MLP, split into
        (scale1, shift1, gate1, scale2, shift2, gate2) for self-attn and FFN.

    Weight keys expected:
        {prefix}.self_attn.w_q/w_k/w_v/w_o
        {prefix}.cross_attn.w_q/w_k/w_v/w_o
        {prefix}.ffn.w_fc1/w_fc1_gate/w_fc2
        {prefix}.norm_cross.gamma  (for cross-attn norm)

    Returns updated hidden state.
    """
    # Split AdaLN params: 6 chunks of hidden_size
    # [scale1, shift1, gate1, scale2, shift2, gate2]
    chunks = []
    for i in range(6):
        s = network.add_slice(
            adaln_params,
            start=(0, i * hidden_size),
            shape=(1, hidden_size),
            stride=(1, 1),
        )
        chunks.append(s.get_output(0))
    scale1, shift1, gate1, scale2, shift2, gate2 = chunks

    # --- Self-attention with AdaLN ---
    normed = graph_ops.add_adaptive_layernorm(
        network, hidden, scale1, shift1, hidden_size, eps, dtype=dtype)

    self_attn_out = graph_ops.add_self_attention_block(
        network, normed,
        w_q=weights[f"{prefix}.self_attn.w_q"],
        w_k=weights[f"{prefix}.self_attn.w_k"],
        w_v=weights[f"{prefix}.self_attn.w_v"],
        w_o=weights[f"{prefix}.self_attn.w_o"],
        hidden_size=hidden_size,
        num_heads=num_heads,
        seq_length=q_seq_len,
        dtype=dtype,
    )

    # Gate and residual
    gated_self_attn = network.add_elementwise(
        self_attn_out, gate1, trt.ElementWiseOperation.PROD)
    hidden = network.add_elementwise(
        hidden, gated_self_attn.get_output(0),
        trt.ElementWiseOperation.SUM).get_output(0)

    # --- Cross-attention (no AdaLN, uses standard LayerNorm) ---
    cross_norm_gamma = weights.get(f"{prefix}.norm_cross.gamma")
    if cross_norm_gamma is not None:
        cross_normed = graph_ops.add_layer_norm_native(
            network, hidden, hidden_size,
            cross_norm_gamma,
            weights.get(f"{prefix}.norm_cross.beta",
                        np.zeros(hidden_size, dtype=np.float32)),
            eps, dtype=dtype)
    else:
        cross_normed = hidden

    cross_attn_out = graph_ops.add_cross_attention(
        network, cross_normed, context,
        w_q=weights[f"{prefix}.cross_attn.w_q"],
        w_k=weights[f"{prefix}.cross_attn.w_k"],
        w_v=weights[f"{prefix}.cross_attn.w_v"],
        w_o=weights[f"{prefix}.cross_attn.w_o"],
        hidden_size=hidden_size,
        context_dim=context_dim,
        num_heads=num_heads,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        dtype=dtype,
    )

    hidden = network.add_elementwise(
        hidden, cross_attn_out,
        trt.ElementWiseOperation.SUM).get_output(0)

    # --- FFN with AdaLN ---
    ffn_normed = graph_ops.add_adaptive_layernorm(
        network, hidden, scale2, shift2, hidden_size, eps, dtype=dtype)

    ffn_out = add_gated_mlp(
        network, ffn_normed,
        weights=weights,
        prefix=f"{prefix}.ffn",
        hidden_size=hidden_size,
        mlp_size=mlp_size,
        activation="silu",
        dtype=dtype,
        quant_ctx=quant_ctx,
        layer_prefix=layer_prefix,
    )

    # Gate and residual
    gated_ffn = network.add_elementwise(
        ffn_out, gate2, trt.ElementWiseOperation.PROD)
    hidden = network.add_elementwise(
        hidden, gated_ffn.get_output(0),
        trt.ElementWiseOperation.SUM).get_output(0)

    return hidden


def add_vae_resblock_3d(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    cache_in1: trt.ITensor,
    cache_in2: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    in_channels: int,
    out_channels: int,
    norm_type: str = "group_norm",
    num_groups: int = 32,
    temporal_kernel: int = 3,
    eps: float = 1e-6,
    dtype: np.dtype = np.float32,
) -> tuple[trt.ITensor, trt.ITensor, trt.ITensor]:
    """3D VAE residual block with causal temporal convolutions.

    Input: [B, C_in, T, H, W] (T >= 1)
    cache_in1, cache_in2: temporal caches for the two causal convs

    Args:
        norm_type: "group_norm" uses GroupNorm with weight/bias keys,
                   "l2_channel_norm" uses L2 channel norm with gamma key.

    Returns: (output, updated_cache1, updated_cache2)

    Structure: Norm -> SiLU -> CausalConv3D -> Norm -> SiLU -> CausalConv3D + shortcut
    """
    def _apply_vae_norm(x, channels, norm_idx):
        if norm_type == "l2_channel_norm":
            return graph_ops.add_l2_channel_norm(
                network, x, channels,
                weights[f"{prefix}.norm{norm_idx}.gamma"], eps,
                dtype=dtype)
        else:
            return graph_ops.add_group_norm(
                network, x, channels, num_groups,
                weights[f"{prefix}.norm{norm_idx}.weight"],
                weights[f"{prefix}.norm{norm_idx}.bias"], eps,
                dtype=dtype)

    # First Norm + SiLU + CausalConv3D
    normed1 = _apply_vae_norm(inp, in_channels, 1)
    act1 = graph_ops.add_silu(network, normed1)
    conv1_out, cache_out1 = graph_ops.add_causal_conv3d(
        network, act1, cache_in1,
        weight=weights[f"{prefix}.conv1.weight"],
        bias=weights.get(f"{prefix}.conv1.bias"),
        out_channels=out_channels,
        kernel_size=(temporal_kernel, 3, 3),
        padding_hw=(1, 1),
        dtype=dtype,
    )

    # Second Norm + SiLU + CausalConv3D
    normed2 = _apply_vae_norm(conv1_out, out_channels, 2)
    act2 = graph_ops.add_silu(network, normed2)
    conv2_out, cache_out2 = graph_ops.add_causal_conv3d(
        network, act2, cache_in2,
        weight=weights[f"{prefix}.conv2.weight"],
        bias=weights.get(f"{prefix}.conv2.bias"),
        out_channels=out_channels,
        kernel_size=(temporal_kernel, 3, 3),
        padding_hw=(1, 1),
        dtype=dtype,
    )

    # Shortcut (1x1 conv if channel mismatch)
    # Weight key differs: l2_channel_norm models use "conv_shortcut", group_norm use "shortcut"
    if in_channels != out_channels:
        sc_key = f"{prefix}.conv_shortcut" if norm_type == "l2_channel_norm" else f"{prefix}.shortcut"
        shortcut = graph_ops.add_conv3d_as_conv2d(
            network, inp,
            weight=weights[f"{sc_key}.weight"],
            bias=weights.get(f"{sc_key}.bias"),
            out_channels=out_channels,
            kernel_size=(1, 1, 1),
            dtype=dtype,
        )
    else:
        shortcut = inp

    # Residual connection
    out = network.add_elementwise(
        conv2_out, shortcut, trt.ElementWiseOperation.SUM)

    return out.get_output(0), cache_out1, cache_out2


def add_vae_spatial_attention(
    network: trt.INetworkDefinition,
    inp: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    channels: int,
    norm_type: str = "l2_channel_norm",
    num_groups: int = 32,
    eps: float = 1e-6,
    dtype: np.dtype = np.float32,
    quant_ctx: QuantContext | None = None,
    layer_prefix: str = "",
) -> trt.ITensor:
    """VAE mid-block spatial self-attention with configurable norm.

    Single-head attention over spatial positions (H*W) per frame.

    Input: [B, C, T, H, W]
    Weight keys:
        {prefix}.norm.gamma           [C, 1, 1, 1]  (l2_channel_norm)
        {prefix}.norm.weight/.bias    [C]            (group_norm)
        {prefix}.to_qkv.weight        [3C, C, 1, 1, 1]
        {prefix}.to_qkv.bias          [3C]
        {prefix}.proj.weight           [C, C, 1, 1, 1]
        {prefix}.proj.bias             [C]

    Output: [B, C, T, H, W] (residual connection applied)
    """
    matmul = _make_matmul_fn(network, dtype, quant_ctx)
    _lp = layer_prefix or prefix

    b, c, t, h, w = inp.shape
    bt = b * t
    hw = h * w
    attn_scale = 1.0 / np.sqrt(max(c, 1))

    identity = inp

    # Configurable norm
    if norm_type == "l2_channel_norm":
        normed = graph_ops.add_l2_channel_norm(
            network, inp, channels,
            weights[f"{prefix}.norm.gamma"], eps, dtype=dtype)
    else:
        normed = graph_ops.add_group_norm(
            network, inp, channels, num_groups,
            weights[f"{prefix}.norm.weight"],
            weights[f"{prefix}.norm.bias"], eps, dtype=dtype)

    # Reshape [B, C, T, H, W] -> [B*T*H*W, C]  (2D for matmul compat)
    flatten = network.add_shuffle(normed)
    flatten.first_transpose = trt.Permutation([0, 2, 3, 4, 1])  # [B,T,H,W,C]
    flatten.reshape_dims = (bt * hw, c)

    # QKV projection: [BT*HW, C] @ [C, 3C] -> [BT*HW, 3C]
    qkv_w = weights[f"{prefix}.to_qkv.weight"]
    qkv_w_2d = qkv_w.reshape(3 * c, c).T.copy()
    qkv = matmul(flatten.get_output(0), c, 3 * c, qkv_w_2d,
                 f"{_lp}.to_qkv.weight")
    qkv_bias = weights.get(f"{prefix}.to_qkv.bias")
    if qkv_bias is not None:
        qkv = graph_ops.add_bias_sum(network, qkv, 3 * c, qkv_bias, dtype=dtype)

    # Reshape to [BT, HW, 3C] then split Q, K, V
    qkv_3d = network.add_shuffle(qkv)
    qkv_3d.reshape_dims = (bt, hw, 3 * c)

    q_slice = network.add_slice(
        qkv_3d.get_output(0),
        start=(0, 0, 0), shape=(bt, hw, c), stride=(1, 1, 1))
    k_slice = network.add_slice(
        qkv_3d.get_output(0),
        start=(0, 0, c), shape=(bt, hw, c), stride=(1, 1, 1))
    v_slice = network.add_slice(
        qkv_3d.get_output(0),
        start=(0, 0, 2 * c), shape=(bt, hw, c), stride=(1, 1, 1))

    q = q_slice.get_output(0)  # [BT, HW, C]
    k = k_slice.get_output(0)
    v = v_slice.get_output(0)

    # Native attention over spatial positions for each B*T frame.
    q_4d = network.add_shuffle(q)
    q_4d.reshape_dims = (bt, 1, hw, c)
    k_4d = network.add_shuffle(k)
    k_4d.reshape_dims = (bt, 1, hw, c)
    v_4d = network.add_shuffle(v)
    v_4d.reshape_dims = (bt, 1, hw, c)
    context = graph_ops.add_attention_core(
        network,
        q_4d.get_output(0),
        k_4d.get_output(0),
        v_4d.get_output(0),
        scale=attn_scale,
    )

    # Flatten context to 2D for output projection: [BT*HW, C]
    ctx_flat = network.add_shuffle(context)
    ctx_flat.reshape_dims = (bt * hw, c)

    # Output projection: [BT*HW, C] @ [C, C] -> [BT*HW, C]
    proj_w = weights[f"{prefix}.proj.weight"]
    proj_w_2d = proj_w.reshape(c, c).T.copy()
    proj_out = matmul(ctx_flat.get_output(0), c, c, proj_w_2d,
                      f"{_lp}.proj.weight")
    proj_bias = weights.get(f"{prefix}.proj.bias")
    if proj_bias is not None:
        proj_out = graph_ops.add_bias_sum(network, proj_out, c, proj_bias, dtype=dtype)

    # Reshape back to [B, C, T, H, W]
    reshape_out = network.add_shuffle(proj_out)
    reshape_out.reshape_dims = (b, t, h, w, c)
    reshape_out.second_transpose = trt.Permutation([0, 4, 1, 2, 3])  # [B,C,T,H,W]

    # Residual
    result = network.add_elementwise(
        reshape_out.get_output(0), identity, trt.ElementWiseOperation.SUM)

    return result.get_output(0)
