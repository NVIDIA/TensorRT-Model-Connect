# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact ownership of the Cosmos3-Edge reasoning checkpoint."""

from tensorrt_model_connect.model_support import family_support

SUPPORTED_TASKS = (
    "text_generation", "vision_language_generation", "text_continuation",
    "images_text_to_text", "video_text_to_text", "image_video_text_to_text",
)

describe = family_support(
    model_types=("cosmos3_edge",),
    tasks=SUPPORTED_TASKS,
    default_task="vision_language_generation",
)
