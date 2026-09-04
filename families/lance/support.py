# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for Lance."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata, family_support


_STATIC = family_support(
    model_types=("lance",),
    tasks=("vision_language_generation",),
    default_task="vision_language_generation",
)
_SUPPORT = FamilySupport(
    tasks=("vision_language_generation",),
    default_task="vision_language_generation",
)


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    if support := _STATIC(metadata):
        return support
    config = metadata.config
    if config.get("model_name") == "Lance" and config.get("organization") == (
        "bytedance-research"
    ):
        return _SUPPORT
    return None
