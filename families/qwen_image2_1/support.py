# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for qwen_image2_1."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    pipeline_classes=("QwenImage21Pipeline",),
    tasks=("text_to_image",),
    default_task="text_to_image",
    default_precision="bf16",
)
