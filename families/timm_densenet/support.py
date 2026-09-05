# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for timm DenseNet."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("timm_densenet", "densenet121", "densenet161", "densenet169", "densenet201"),
    architectures=("densenet121", "densenet161", "densenet169", "densenet201"),
    tasks=("classification",),
    default_task="classification",
)
