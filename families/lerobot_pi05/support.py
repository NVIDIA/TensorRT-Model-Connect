# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for LeRobot Pi 0.5."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata


_SUPPORT = FamilySupport(
    tasks=("images_state_text_to_action_chunk",),
    default_task="images_state_text_to_action_chunk",
    default_precision="bf16",
)


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    # LeRobot policy checkpoints identify themselves with ``type`` in config.json.
    # Pi 0, Pi 0-FAST and the other policy types are separate families.
    if str(metadata.config.get("type", "")).lower() == "pi05":
        return _SUPPORT
    return None
