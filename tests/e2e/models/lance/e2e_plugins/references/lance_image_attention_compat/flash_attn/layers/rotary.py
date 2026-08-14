# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rotary embedding helper required by Lance's Qwen2.5-VL vision encoder."""

from __future__ import annotations

import torch


def apply_rotary_emb(
    values: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
    seqlen_offsets: int = 0,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
) -> torch.Tensor:
    """Apply FlashAttention's rotary contract for dense Lance vision inputs."""
    del max_seqlen
    if cu_seqlens is not None:
        raise NotImplementedError("L4T rotary fallback does not support packed offsets")
    if not isinstance(seqlen_offsets, int):
        raise NotImplementedError("L4T rotary fallback requires one integer sequence offset")
    if values.ndim not in (3, 4):
        raise ValueError("rotary input must have rank 3 or 4")

    sequence_axis = 1 if values.ndim == 4 else 0
    sequence_length = values.shape[sequence_axis]
    cos = cos[seqlen_offsets : seqlen_offsets + sequence_length]
    sin = sin[seqlen_offsets : seqlen_offsets + sequence_length]
    if conjugate:
        sin = -sin
    if values.ndim == 4:
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    else:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    if interleaved:
        first = values[..., 0::2]
        second = values[..., 1::2]
        rotated = torch.stack(
            (first * cos - second * sin, first * sin + second * cos),
            dim=-1,
        ).flatten(-2)
    else:
        first, second = values.chunk(2, dim=-1)
        rotated = torch.cat(
            (first * cos - second * sin, first * sin + second * cos),
            dim=-1,
        )
    if inplace:
        values.copy_(rotated)
        return values
    return rotated


__all__ = ["apply_rotary_emb"]
