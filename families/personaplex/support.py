# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for personaplex."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("moshi", "personaplex", "personaplex_7b"),
    tasks=("speech_to_speech",),
    default_task="speech_to_speech",
)
