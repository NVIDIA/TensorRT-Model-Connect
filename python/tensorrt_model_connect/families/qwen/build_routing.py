# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight build routing for the Qwen native-KV prototype."""

from __future__ import annotations


def is_qwen3_06b_native_kv_prototype(config: object) -> bool:
    """Match the exact Qwen3-0.6B architecture qualified by this prototype."""
    return (
        str(getattr(config, "model_type", "")).lower() == "qwen3"
        and int(getattr(config, "hidden_size", 0)) == 1024
        and int(getattr(config, "intermediate_size", 0)) == 3072
        and int(getattr(config, "num_hidden_layers", 0)) == 28
        and int(getattr(config, "num_attention_heads", 0)) == 16
        and int(getattr(config, "num_key_value_heads", 0)) == 8
        and int(getattr(config, "head_dim", 0)) == 128
        and int(getattr(config, "max_position_embeddings", 0)) == 40960
    )


def prefer_native_default(
    config: object,
    *,
    explicit_public_options: frozenset[str],
) -> bool:
    """Keep model-only Qwen3-0.6B builds on the native full-context path."""
    deployment_options = {"precision", "max_cache_length"}
    return (
        not deployment_options.intersection(explicit_public_options)
        and is_qwen3_06b_native_kv_prototype(config)
    )
