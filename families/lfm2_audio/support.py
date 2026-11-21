# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LFM audio checkpoint identity and native task capabilities."""

from tensorrt_model_connect.model_support import family_support

describe = family_support(
    architectures=("Lfm2AudioForConditionalGeneration",),
    tasks=(
        "offline_speech_dialogue",
        "speech_transcription",
        "speech_to_speech_response",
        "text_to_speech",
        "streaming_text_to_speech",
        "audio_text_to_text",
        "text_conversation",
    ),
    default_task="offline_speech_dialogue",
    default_precision="bf16",
)
