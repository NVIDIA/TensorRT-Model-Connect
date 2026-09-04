# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for deepseek_ocr."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("deepseek_ocr", "deepseek_vl_v2", "deepseekocr"),
    tasks=("vision_language_generation",),
    default_task="vision_language_generation",
)
