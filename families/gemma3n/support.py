# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact ownership of Gemma 3n multimodal checkpoints."""

from tensorrt_model_connect.model_support import family_support

SUPPORTED_TASKS = ("images_text_to_text",)

describe = family_support(
    model_types=("gemma3n",),
    architectures=("Gemma3nForConditionalGeneration",),
    tasks=SUPPORTED_TASKS,
    default_task="images_text_to_text",
    default_precision="bf16",
)
