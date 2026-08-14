# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Padding helpers required by Transformers' FlashAttention integration."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def index_first_axis(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Select entries from the first axis without changing trailing dimensions."""
    return values.index_select(0, indices.to(device=values.device, dtype=torch.long))


def unpad_input(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Pack valid batch tokens and return FlashAttention-compatible indices."""
    if hidden_states.ndim < 2 or attention_mask.shape != hidden_states.shape[:2]:
        raise ValueError("attention_mask must match hidden_states batch and sequence axes")
    valid = attention_mask.to(dtype=torch.bool)
    lengths = valid.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(valid.flatten(), as_tuple=False).flatten()
    flattened = hidden_states.reshape(-1, *hidden_states.shape[2:])
    unpadded = index_first_axis(flattened, indices)
    cumulative = functional.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))
    maximum = int(lengths.max().item()) if lengths.numel() else 0
    return unpadded, indices, cumulative, maximum


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    sequence_length: int,
) -> torch.Tensor:
    """Restore packed token states to a zero-padded batch tensor."""
    output = hidden_states.new_zeros(
        (batch_size * sequence_length, *hidden_states.shape[1:])
    )
    output.index_copy_(
        0,
        indices.to(device=hidden_states.device, dtype=torch.long),
        hidden_states,
    )
    return output.reshape(batch_size, sequence_length, *hidden_states.shape[1:])


__all__ = ["index_first_axis", "pad_input", "unpad_input"]
