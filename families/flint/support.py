# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact Flint checkpoint ownership."""

from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("flint",),
    architectures=("FlintForCausalLM",),
    tasks=("text_generation",),
    default_task="text_generation",
    default_precision="bf16",
)
