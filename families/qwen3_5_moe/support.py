# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned discovery for ordinary full hybrid-MoE checkpoints."""

from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("qwen3_5_moe",),
    tasks=("text_generation", "vision_language_generation"),
    default_task="vision_language_generation",
    default_precision="fp16",
)
