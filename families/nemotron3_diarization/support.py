# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for Nemotron 3 Diarization."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("nemotron3_diarization",),
    architectures=("Nemotron3DiarizationForAudioFrameClassification",),
    tasks=("audio_to_timed_labels",),
    default_task="audio_to_timed_labels",
)
