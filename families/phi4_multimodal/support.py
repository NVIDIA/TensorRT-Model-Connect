# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for phi4_multimodal."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("phi4mm", "phi4_multimodal", "phi4multimodal"),
    tasks=("vision_language_generation",),
    default_task="vision_language_generation",
)
