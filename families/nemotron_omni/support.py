# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact checkpoint identity and complete native Omni task ownership."""

from tensorrt_model_connect.model_support import family_support

SUPPORTED_TASKS = (
    "text_generation",
    "vision_language_generation",
    "text_continuation",
    "images_text_to_text",
    "video_text_to_text",
    "audio_text_to_text",
    "image_audio_text_to_text",
    "audio_video_text_to_text",
)

describe = family_support(
    model_types=("NemotronH_Nano_Omni_Reasoning_V3",),
    tasks=SUPPORTED_TASKS,
    default_task="vision_language_generation",
    default_precision="fp16",
)
