# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for Stable Audio Open (diffusers StableAudioPipeline)."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    pipeline_classes=("StableAudioPipeline",),
    tasks=("text_to_audio",),
    default_task="text_to_audio",
)
