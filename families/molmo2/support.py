# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint identity and public task ownership for Molmo2."""

from tensorrt_model_connect.model_support import FamilySupport


def describe(metadata):
    if (
        metadata.model_type == "molmo2"
        and "Molmo2ForConditionalGeneration" in metadata.architectures
    ):
        return FamilySupport(("vision_language_generation",), "vision_language_generation", "bf16")
    return None
