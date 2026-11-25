# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ownership of the register-token DINOv2 backbone."""

from tensorrt_model_connect.model_support import family_support

_backbone_support = family_support(
    model_types=("dinov2_with_registers",),
    architectures=("Dinov2WithRegistersModel",),
    tasks=("image_to_token_and_pooled_features",),
    default_task="image_to_token_and_pooled_features",
    default_precision="fp32",
)


def describe(metadata):
    if metadata.architectures and any(
        architecture != "Dinov2WithRegistersModel" for architecture in metadata.architectures
    ):
        return None
    return _backbone_support(metadata)
