# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cohere 2 parallel decoder, including Tiny Aya Global."""

from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("cohere2",),
    architectures=("Cohere2ForCausalLM",),
    tasks=("text_continuation",),
    default_task="text_continuation",
    default_precision="fp32",
)
