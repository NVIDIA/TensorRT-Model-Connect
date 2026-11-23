# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import FamilySupport


def is_pipeline_class(value):
    return value == "PH3ImageModularPipeline"


def describe(metadata):
    if is_pipeline_class(metadata.pipeline_class):
        return FamilySupport(
            tasks=("text_to_image",), default_task="text_to_image", default_precision="bf16"
        )
    return None
