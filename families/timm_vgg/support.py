# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for timm_vgg."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("timm_vgg",),
    architectures=("vgg11", "vgg13", "vgg16", "vgg19"),
    tasks=("classification",),
    default_task="classification",
)
