# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration fields consumed by the Marian family implementation."""

from __future__ import annotations

from typing import Any, Protocol


class ModelConfig(Protocol):
    """Marian's view of the repository's parsed model configuration."""

    model_type: str
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    rms_norm_eps: float
    raw: dict[str, Any]
