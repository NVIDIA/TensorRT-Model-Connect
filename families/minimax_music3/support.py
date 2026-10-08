# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for MiniMax-Music3."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("minimax_music3", "minimaxmusic3"),
    architectures=("MiniMaxMusic3ForConditionalGeneration",),
    tasks=("audio_generation",),
    default_task="audio_generation",
)
