# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for magpie_tts."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("decoder_ce", "magpie_tts", "magpietts"),
    required_files=("magpie_tts_multilingual_357m.nemo",),
    tasks=("audio_generation",),
    default_task="audio_generation",
)
