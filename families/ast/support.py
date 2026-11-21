# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for the Audio Spectrogram Transformer."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("audio-spectrogram-transformer",),
    architectures=("ASTForAudioClassification",),
    tasks=("audio_to_class_scores",),
    default_task="audio_to_class_scores",
)
