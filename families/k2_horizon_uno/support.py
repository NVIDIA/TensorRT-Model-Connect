# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for K2-Horizon-Uno."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata


_SUPPORT = FamilySupport(tasks=("text_generation",), default_task="text_generation")
_FILES = ("adapter_config.json", "adapter_model.safetensors")


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    """Select the two-file adapter snapshot; build validates its exact recipe."""

    return _SUPPORT if metadata.files == _FILES else None
