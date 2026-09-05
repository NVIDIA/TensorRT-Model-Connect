# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for qwen_vl."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("qwen2_5_vl", "qwen2_vl", "qwen3_vl", "qwen_vl", "qwenvl"),
    tasks=("vision_language_generation",),
    default_task="vision_language_generation",
)
