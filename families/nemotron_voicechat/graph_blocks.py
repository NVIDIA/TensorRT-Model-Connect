# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VoiceChat thinker graph blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import tensorrt as trt

from . import graph_ops

if TYPE_CHECKING:
    from .checkpoint_mapper import WeightDict


def infer_kv_attention_size(
    weights: dict,
    *,
    prefix: str = "layer.0",
    num_kv_heads: int,
    head_dim: int,
) -> int:
    """Validate and return the compact K/V row width."""
    expected = int(num_kv_heads * head_dim)
    w_k = weights.get(f"{prefix}.w_k")
    if isinstance(w_k, np.ndarray) and w_k.ndim == 2:
        actual = int(w_k.shape[1])
        if actual != expected:
            raise ValueError(f"{prefix}.w_k must use compact K/V width {expected}, got {actual}")
    return expected


def add_attention_block(
    network: trt.INetworkDefinition,
    hidden: trt.ITensor,
    cache_k: trt.ITensor,
    cache_v: trt.ITensor,
    attention_mask: trt.ITensor,
    *,
    weights: WeightDict,
    prefix: str,
    hidden_size: int,
    attention_size: int,
    kv_attention_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    max_cache_length: int,
    eps_tensor: trt.ITensor,
    dtype: np.dtype = np.float32,
) -> dict[str, trt.ITensor]:
    """Add the pinned Nemotron-H RMSNorm/GQA attention block."""
    normed = graph_ops.add_rms_norm(
        network,
        hidden,
        hidden_size,
        weights[f"{prefix}.input_norm"],
        eps_tensor,
        dtype=dtype,
    )
    q = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, attention_size, weights[f"{prefix}.w_q"], dtype=dtype
    )
    k = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, kv_attention_size, weights[f"{prefix}.w_k"], dtype=dtype
    )
    v = graph_ops.add_matmul_rhs_constant(
        network, normed, hidden_size, kv_attention_size, weights[f"{prefix}.w_v"], dtype=dtype
    )

    k_row = network.add_shuffle(k)
    k_row.reshape_dims = (1, kv_attention_size)
    v_row = network.add_shuffle(v)
    v_row.reshape_dims = (1, kv_attention_size)
    all_k = network.add_concatenation([cache_k, k_row.get_output(0)])
    all_k.axis = 0
    all_v = network.add_concatenation([cache_v, v_row.get_output(0)])
    all_v.axis = 0

    context = graph_ops.add_attention_from_rows(
        network,
        q,
        all_k.get_output(0),
        all_v.get_output(0),
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_seq=1,
        kv_seq=max_cache_length + 1,
        mask=graph_ops.add_2d_mask_to_4d(network, attention_mask),
    )
    attn_out = graph_ops.add_matmul_rhs_constant(
        network, context, attention_size, hidden_size, weights[f"{prefix}.w_o"], dtype=dtype
    )
    return {"attn_out": attn_out, "present_k": k, "present_v": v}
