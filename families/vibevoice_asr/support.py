# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tensorrt_model_connect.model_support import family_support

# The released checkpoint keeps the original layout: model_type "vibevoice" is
# shared with the VibeVoice speech-synthesis checkpoints, so only the ASR
# architecture identifies this family.
describe = family_support(
    architectures=("VibeVoiceForASRTraining",),
    tasks=("speech_transcription",),
    default_task="speech_transcription",
    default_precision="bf16",
)
