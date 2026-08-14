# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Packed-attention compatibility for Lance reference environments.

Lance imports FlashAttention's packed, dense, rotary, and padding helpers while
loading its official image-reference modules. Platforms without a qualified
FlashAttention package can explicitly select this PyTorch implementation. It is
never selected by CPU architecture inference.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as functional
from torch.nn.attention.bias import causal_lower_right


def _sequence_offsets(
    cumulative: torch.Tensor,
    *,
    total: int,
    label: str,
) -> list[int]:
    offsets = [int(value) for value in cumulative.detach().cpu().tolist()]
    if len(offsets) < 2 or offsets[0] != 0 or offsets[-1] != total:
        raise ValueError(
            f"{label} must start at zero and end at packed length {total}: "
            f"{offsets}"
        )
    if any(end < start for start, end in zip(offsets, offsets[1:])):
        raise ValueError(f"{label} must be non-decreasing: {offsets}")
    return offsets


def _attention_mask(query_length: int, key_length: int, causal: bool):
    if not causal:
        return None
    return causal_lower_right(query_length, key_length)


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: Sequence[int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
) -> torch.Tensor:
    """Evaluate FlashAttention's dense batch contract with PyTorch SDPA."""
    del deterministic
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("dense q, k, and v tensors must have rank 4")
    if k.shape[:3] != v.shape[:3]:
        raise ValueError("dense k and v tensors must have matching shapes")
    if q.shape[0] != k.shape[0] or q.shape[-1] != k.shape[-1]:
        raise ValueError("dense q and k batch and head dimensions must match")
    if tuple(window_size) != (-1, -1):
        raise NotImplementedError("L4T SDPA fallback does not support local windows")
    if softcap or alibi_slopes is not None:
        raise NotImplementedError(
            "L4T SDPA fallback supports only Lance's dense attention contract"
        )
    if return_attn_probs:
        raise NotImplementedError("L4T SDPA fallback does not return attention probabilities")

    query = q.transpose(1, 2)
    key = k.transpose(1, 2)
    value = v.transpose(1, 2)
    output = functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=_attention_mask(query.shape[-2], key.shape[-2], causal),
        dropout_p=dropout_p,
        scale=softmax_scale,
        enable_gqa=query.shape[1] != key.shape[1],
    )
    return output.transpose(1, 2)


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: Sequence[int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate FlashAttention's packed varlen contract with PyTorch SDPA."""
    del max_seqlen_q, max_seqlen_k, deterministic
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("packed q, k, and v tensors must have rank 3")
    if k.shape[:2] != v.shape[:2]:
        raise ValueError("packed k and v tensors must have matching token and head counts")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("packed q and k head dimensions must match")
    if tuple(window_size) != (-1, -1):
        raise NotImplementedError("L4T SDPA fallback does not support local windows")
    if softcap or alibi_slopes is not None or block_table is not None:
        raise NotImplementedError(
            "L4T SDPA fallback supports only Lance's dense attention contract"
        )
    if return_attn_probs:
        raise NotImplementedError("L4T SDPA fallback does not return attention probabilities")

    query_offsets = _sequence_offsets(
        cu_seqlens_q,
        total=q.shape[0],
        label="cu_seqlens_q",
    )
    key_offsets = _sequence_offsets(
        cu_seqlens_k,
        total=k.shape[0],
        label="cu_seqlens_k",
    )
    if len(query_offsets) != len(key_offsets):
        raise ValueError("packed q and k must contain the same number of sequences")

    outputs: list[torch.Tensor] = []
    for q_start, q_end, k_start, k_end in zip(
        query_offsets,
        query_offsets[1:],
        key_offsets,
        key_offsets[1:],
    ):
        query = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        key = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
        value = v[k_start:k_end].transpose(0, 1).unsqueeze(0)
        mask = _attention_mask(q_end - q_start, k_end - k_start, causal)
        output = functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=dropout_p,
            scale=softmax_scale,
            enable_gqa=query.shape[1] != key.shape[1],
        )
        outputs.append(output.squeeze(0).transpose(0, 1))
    return (
        torch.cat(outputs, dim=0)
        if outputs
        else q.new_empty((0, q.shape[1], v.shape[-1]))
    )


__all__ = ["flash_attn_func", "flash_attn_varlen_func"]
