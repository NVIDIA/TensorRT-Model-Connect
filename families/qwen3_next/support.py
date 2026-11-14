# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact metadata ownership for Qwen3-Next language checkpoints."""

from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("qwen3_next",),
    architectures=("Qwen3NextForCausalLM",),
    tasks=("text_generation",),
    default_task="text_generation",
    default_precision="bf16",
)
