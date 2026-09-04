# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for qwen_image."""

from tensorrt_model_connect.model_support import ModelMetadata, FamilySupport, family_support


_TASKS = ("image_generation", "image_edit")
_EDIT = family_support(
    model_types=("qwen-image-edit", "qwen_image_edit", "qwenimageedit"),
    pipeline_classes=("QwenImageEditPipeline", "QwenImageEditPlusPipeline"),
    tasks=_TASKS,
    default_task="image_edit",
)
_GENERATE = family_support(
    model_types=("qwen-image", "qwen_image", "qwenimage"),
    pipeline_classes=("QwenImagePipeline",),
    tasks=_TASKS,
    default_task="image_generation",
)


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    return _EDIT(metadata) or _GENERATE(metadata)
