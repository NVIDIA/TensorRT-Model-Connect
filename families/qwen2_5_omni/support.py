# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text-only Thinker slice; no audio, image or video Task is advertised."""

from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("qwen2_5_omni",),
    tasks=("text_continuation",),
    default_task="text_continuation",
    default_precision="fp32",
)
