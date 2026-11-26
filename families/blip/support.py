# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""BLIP captioning ownership excludes BLIP retrieval and VQA architectures."""

from tensorrt_model_connect.model_support import FamilySupport


def describe(metadata):
    if metadata.model_type != "blip" or metadata.architectures != ("BlipForConditionalGeneration",):
        return None
    return FamilySupport(
        tasks=("images_text_to_text",), default_task="images_text_to_text", default_precision="fp32"
    )
