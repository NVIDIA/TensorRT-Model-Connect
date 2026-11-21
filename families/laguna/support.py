# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Laguna owns checkpoint identity and the available primary bundle task."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("laguna",),
    architectures=("LagunaForCausalLM",),
    tasks=("text_continuation",),
    default_task="text_continuation",
    default_precision="bf16",
)
