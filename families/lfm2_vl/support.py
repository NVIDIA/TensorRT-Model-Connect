# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Family-owned support for Liquid vision-language checkpoints."""

from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("lfm2_vl",),
    tasks=("images_text_to_text", "text_conversation", "vision_language_generation"),
    default_task="images_text_to_text",
)
