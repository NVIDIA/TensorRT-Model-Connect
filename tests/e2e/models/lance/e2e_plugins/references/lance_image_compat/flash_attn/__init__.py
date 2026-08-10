# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safe PyTorch fallback for Lance's image-only variable-length attention.

The pinned upstream image reference imports ``flash_attn_varlen_func``
unconditionally. The compiled ``flash-attn`` distribution also ships training
checkpoint helpers affected by CVE-2026-31253, so the reference exposes only
the inference primitive it needs and delegates that operation to PyTorch SDPA.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def flash_attn_varlen_func(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor | None = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the subset of packed attention used by Lance image inference."""
    del max_seqlen_q, max_seqlen_k, deterministic
    if dropout_p != 0.0:
        raise ValueError("Lance's image reference supports inference dropout only")
    if window_size != (-1, -1):
        raise ValueError("Lance's image reference does not request windowed attention")
    if softcap != 0.0 or alibi_slopes is not None or block_table is not None:
        raise ValueError("Lance's image reference received unsupported attention options")
    if return_attn_probs:
        raise ValueError("Lance's image reference does not expose attention probabilities")
    if query.ndim != 3 or key.ndim != 3 or value.ndim != 3:
        raise ValueError("packed attention inputs must have shape [tokens, heads, dim]")

    q_offsets = cu_seqlens_q.detach().cpu().tolist()
    k_offsets = cu_seqlens_k.detach().cpu().tolist()
    if len(q_offsets) != len(k_offsets) or len(q_offsets) < 2:
        raise ValueError("packed attention offsets must describe matching batches")
    if q_offsets[0] != 0 or k_offsets[0] != 0:
        raise ValueError("packed attention offsets must start at zero")
    if any(not isinstance(offset, int) for offset in (*q_offsets, *k_offsets)):
        raise ValueError("packed attention offsets must be integers")

    outputs: list[torch.Tensor] = []
    for index in range(len(q_offsets) - 1):
        q_start, q_end = q_offsets[index : index + 2]
        k_start, k_end = k_offsets[index : index + 2]
        q_length = q_end - q_start
        k_length = k_end - k_start
        if q_length < 0 or k_length < 0:
            raise ValueError("packed attention offsets must be nondecreasing")
        if causal and q_length != k_length:
            raise ValueError("causal packed attention requires equal query/key lengths")

        q_slice = query[q_start:q_end].transpose(0, 1)
        k_slice = key[k_start:k_end].transpose(0, 1)
        v_slice = value[k_start:k_end].transpose(0, 1)
        output = F.scaled_dot_product_attention(
            q_slice,
            k_slice,
            v_slice,
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale,
        )
        outputs.append(output.transpose(0, 1))

    if q_offsets[-1] != query.shape[0] or k_offsets[-1] != key.shape[0]:
        raise ValueError("packed attention offsets must cover every input token")
    return torch.cat(outputs, dim=0)


__all__ = ["flash_attn_varlen_func"]
