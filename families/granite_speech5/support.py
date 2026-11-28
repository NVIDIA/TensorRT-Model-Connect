# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact checkpoint identity and Task support for Granite Speech 5 CTC."""

from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("granite_speech5_ctc",),
    architectures=("GraniteSpeech5ForCTC",),
    tasks=("speech_transcription",),
    default_task="speech_transcription",
    default_precision="bf16",
)
