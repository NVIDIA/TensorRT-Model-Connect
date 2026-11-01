# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for LeRobot GR00T N1.7."""

from tensorrt_model_connect.model_support import FamilySupport, ModelMetadata


_SUPPORT = FamilySupport(
    tasks=("images_state_text_to_action_chunk",),
    default_task="images_state_text_to_action_chunk",
    default_precision="bf16",
)


def describe(metadata: ModelMetadata) -> FamilySupport | None:
    # LeRobot policy checkpoints identify themselves with ``type`` in config.json; LeRobot 0.6
    # saves only GR00T N1.7 policies under ``groot``.
    if str(metadata.config.get("type", "")).lower() == "groot":
        return _SUPPORT
    return None
