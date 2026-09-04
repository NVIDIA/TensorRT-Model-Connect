# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for T5."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata, family_support


_BY_TYPE = family_support(
    model_types=("t5",),
    tasks=("text_generation",),
    default_task="text_generation",
)
_BY_ARCHITECTURE = family_support(
    architectures=("T5ForConditionalGeneration", "T5Model", "T5EncoderModel"),
    tasks=("text_generation",),
    default_task="text_generation",
)


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    if metadata.architectures:
        return _BY_ARCHITECTURE(metadata)
    return _BY_TYPE(metadata)
