# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact checkpoint identity and Task support for Granite Speech."""

from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("granite_speech",),
    architectures=("GraniteSpeechForConditionalGeneration",),
    tasks=("speech_transcription",),
    default_task="speech_transcription",
    default_precision="bf16",
)
