# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact checkpoint identity of the Apertus causal language models."""

from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("apertus",),
    architectures=("ApertusForCausalLM",),
    tasks=("text_continuation",),
    default_task="text_continuation",
    default_precision="bf16",
)
