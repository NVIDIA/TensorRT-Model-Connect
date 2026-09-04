# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve GPT-Neo's Hugging Face local/global attention contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def resolve_attention_layer_types(
    raw_config: Mapping[str, Any],
    *,
    num_layers: int,
) -> tuple[str, ...]:
    """Return one validated attention type for every decoder layer."""

    configured = raw_config.get("attention_layers")
    if configured is not None:
        layer_types = list(configured)
    else:
        layer_types = []
        for entry in raw_config.get("attention_types") or []:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError(
                    "GPT-Neo attention_types entries must be [pattern, repeats]"
                )
            pattern, repeats = entry
            if not isinstance(pattern, (list, tuple)):
                raise ValueError("GPT-Neo attention pattern must be a sequence")
            layer_types.extend(list(pattern) * int(repeats))

    if not layer_types:
        layer_types = ["global"] * num_layers
    if len(layer_types) != num_layers:
        raise ValueError(
            "GPT-Neo attention pattern length does not match num_hidden_layers: "
            f"{len(layer_types)} != {num_layers}"
        )

    normalized = tuple(str(layer_type).lower() for layer_type in layer_types)
    unsupported = sorted(set(normalized) - {"global", "local"})
    if unsupported:
        raise ValueError(
            f"Unsupported GPT-Neo attention layer types: {unsupported}"
        )
    return normalized


def resolve_local_attention_window(
    raw_config: Mapping[str, Any],
    layer_types: tuple[str, ...],
) -> int:
    """Return and validate the local attention window."""

    window = int(raw_config.get("window_size") or 0)
    if "local" in layer_types and window <= 0:
        raise ValueError(
            "GPT-Neo local attention layers require a positive window_size"
        )
    return window
