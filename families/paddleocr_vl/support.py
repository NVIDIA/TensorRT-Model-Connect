# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import FamilySupport


def describe(metadata):
    if (
        metadata.model_type == "paddleocr_vl"
        and "PaddleOCRVLForConditionalGeneration" in metadata.architectures
    ):
        return FamilySupport(
            tasks=("images_text_to_text",),
            default_task="images_text_to_text",
            default_precision="fp32",
        )
    return None
