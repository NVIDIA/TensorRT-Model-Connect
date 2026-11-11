# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact Phonon checkpoint identity, independent of optional dependencies."""

from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("gradium_phonon_tts",),
    architectures=("GradiumPhononTTSForConditionalGeneration",),
    tasks=("text_to_speech",),
    default_task="text_to_speech",
)
