# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recognize the released Laya checkpoint's explicit model identity."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    if metadata.model_type == "laya" and {"rl_agent_config.json", "model.safetensors"} <= set(
        metadata.files
    ):
        return FamilySupport(("structured_decision",), "structured_decision", "bf16")
    return None
