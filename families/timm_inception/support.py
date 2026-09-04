# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for timm_inception."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    architectures=("inception_v3",),
    tasks=("classification",),
    default_task="classification",
)
