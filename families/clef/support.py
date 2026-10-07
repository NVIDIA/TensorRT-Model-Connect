# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Clef owns the joint schema head, not ordinary Qwen generation."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    if metadata.model_type == "qwen3_5" and {
        "joint_head_config.json",
        "joint_head.safetensors",
        "joint_schema_model.py",
    } <= set(metadata.files):
        return FamilySupport(("structured_decision",), "structured_decision", "bf16")
    return None
