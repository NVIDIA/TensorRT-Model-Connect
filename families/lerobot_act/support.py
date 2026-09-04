# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for LeRobot ACT."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata


_SUPPORT = FamilySupport(tasks=("robot_control",), default_task="robot_control")


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    model_type = metadata.model_type.lower().replace("-", "_")
    checkpoint_type = str(metadata.config.get("type", "")).lower()
    if checkpoint_type == "act" or model_type in {
        "act",
        "act_policy",
        "actpolicy",
        "lerobot_act",
    }:
        return _SUPPORT
    return None
