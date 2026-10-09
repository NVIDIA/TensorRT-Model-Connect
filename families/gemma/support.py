# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gemma owns checkpoint identity and the available primary bundle tasks."""

from tensorrt_model_connect.model_support import family_support

_native = family_support(
    model_types=("gemma", "gemma2", "gemma3", "gemma3_text"),
    tasks=("text_generation",),
    default_task="text_generation",
)


def describe(metadata):
    """Keep earlier native generations distinct from Gemma4 media contracts."""
    if metadata.config.get("model_type") not in {"gemma4", "gemma4_unified"}:
        return _native(metadata)
    tasks = ["text_generation", "text_continuation"]
    vision = isinstance(metadata.config.get("vision_config"), dict)
    audio = isinstance(metadata.config.get("audio_config"), dict)
    if vision:
        tasks.append("images_text_to_text")
    if audio:
        tasks.append("audio_text_to_text")
    if vision and audio:
        tasks.append("image_audio_text_to_text")
    default = "images_text_to_text" if vision else (
        "audio_text_to_text" if audio else "text_continuation"
    )
    return family_support(
        model_types=("gemma4", "gemma4_unified"), tasks=tuple(tasks), default_task=default,
    )(metadata)
