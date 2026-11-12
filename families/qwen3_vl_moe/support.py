# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact ownership of the routed-expert Qwen3-VL architecture."""

from tensorrt_model_connect.model_support import FamilySupport


def describe(metadata):
    if metadata.model_type != "qwen3_vl_moe" or metadata.architectures != (
        "Qwen3VLMoeForConditionalGeneration",
    ):
        return None
    return FamilySupport(
        tasks=("vision_language_generation",),
        default_task="vision_language_generation",
        default_precision="bf16",
    )
