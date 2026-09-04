# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for timm_resnet."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("timm_resnet", "timmresnet", "resnet", "resnet50"),
    architectures=("resnet50", "resnext50_32x4d", "wide_resnet50_2"),
    tasks=("classification",),
    default_task="classification",
)
