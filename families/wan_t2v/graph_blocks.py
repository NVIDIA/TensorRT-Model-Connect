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
                network, x, channels, weights[f"{prefix}.norm{norm_idx}.gamma"], eps, dtype=dtype
            )
        else:
            return graph_ops.add_group_norm(
                network,
                x,
                channels,
                num_groups,
                weights[f"{prefix}.norm{norm_idx}.weight"],
                weights[f"{prefix}.norm{norm_idx}.bias"],
                eps,
                dtype=dtype,
            )

    # First Norm + SiLU + CausalConv3D
    normed1 = _apply_vae_norm(inp, in_channels, 1)
    act1 = graph_ops.add_silu(network, normed1)
    conv1_out, cache_out1 = graph_ops.add_causal_conv3d(
        network,
        act1,
        cache_in1,
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
        network,
        act2,
        cache_in2,
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
        sc_key = (
            f"{prefix}.conv_shortcut" if norm_type == "l2_channel_norm" else f"{prefix}.shortcut"
        )
        shortcut = graph_ops.add_conv3d_as_conv2d(
            network,
            inp,
            weight=weights[f"{sc_key}.weight"],
            bias=weights.get(f"{sc_key}.bias"),
            out_channels=out_channels,
            kernel_size=(1, 1, 1),
            dtype=dtype,
        )
    else:
        shortcut = inp

    # Residual connection
    out = network.add_elementwise(conv2_out, shortcut, trt.ElementWiseOperation.SUM)

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
    quant_ctx=None,
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
            network, inp, channels, weights[f"{prefix}.norm.gamma"], eps, dtype=dtype
        )
    else:
        normed = graph_ops.add_group_norm(
            network,
            inp,
            channels,
            num_groups,
            weights[f"{prefix}.norm.weight"],
            weights[f"{prefix}.norm.bias"],
            eps,
            dtype=dtype,
        )

    # Reshape [B, C, T, H, W] -> [B*T*H*W, C]  (2D for matmul compat)
    flatten = network.add_shuffle(normed)
    flatten.first_transpose = trt.Permutation([0, 2, 3, 4, 1])  # [B,T,H,W,C]
    flatten.reshape_dims = (bt * hw, c)

    # QKV projection: [BT*HW, C] @ [C, 3C] -> [BT*HW, 3C]
    qkv_w = weights[f"{prefix}.to_qkv.weight"]
    qkv_w_2d = qkv_w.reshape(3 * c, c).T.copy()
    qkv = matmul(flatten.get_output(0), c, 3 * c, qkv_w_2d, f"{_lp}.to_qkv.weight")
    qkv_bias = weights.get(f"{prefix}.to_qkv.bias")
    if qkv_bias is not None:
        qkv = graph_ops.add_bias_sum(network, qkv, 3 * c, qkv_bias, dtype=dtype)

    # Reshape to [BT, HW, 3C] then split Q, K, V
    qkv_3d = network.add_shuffle(qkv)
    qkv_3d.reshape_dims = (bt, hw, 3 * c)

    q_slice = network.add_slice(
        qkv_3d.get_output(0), start=(0, 0, 0), shape=(bt, hw, c), stride=(1, 1, 1)
    )
    k_slice = network.add_slice(
        qkv_3d.get_output(0), start=(0, 0, c), shape=(bt, hw, c), stride=(1, 1, 1)
    )
    v_slice = network.add_slice(
        qkv_3d.get_output(0), start=(0, 0, 2 * c), shape=(bt, hw, c), stride=(1, 1, 1)
    )

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
    proj_out = matmul(ctx_flat.get_output(0), c, c, proj_w_2d, f"{_lp}.proj.weight")
    proj_bias = weights.get(f"{prefix}.proj.bias")
    if proj_bias is not None:
        proj_out = graph_ops.add_bias_sum(network, proj_out, c, proj_bias, dtype=dtype)

    # Reshape back to [B, C, T, H, W]
    reshape_out = network.add_shuffle(proj_out)
    reshape_out.reshape_dims = (b, t, h, w, c)
    reshape_out.second_transpose = trt.Permutation([0, 4, 1, 2, 3])  # [B,C,T,H,W]

    # Residual
    result = network.add_elementwise(
        reshape_out.get_output(0), identity, trt.ElementWiseOperation.SUM
    )

    return result.get_output(0)
