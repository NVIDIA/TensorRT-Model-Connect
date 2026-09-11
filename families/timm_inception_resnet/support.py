# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint identity and task ownership for timm InceptionResNet."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("timm_inception_resnet", "inception_resnet_v2"),
    architectures=("inception_resnet_v2",),
    tasks=("classification",),
    default_task="classification",
)
