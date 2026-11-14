# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact Cohere encoder-decoder ASR checkpoint identity."""

from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("cohere_asr",),
    architectures=("CohereAsrForConditionalGeneration",),
    tasks=("speech_transcription",),
    default_task="speech_transcription",
)
