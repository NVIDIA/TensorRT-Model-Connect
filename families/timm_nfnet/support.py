# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for timm NFNet."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=(
        "timm_nfnet",
        "dm_nfnet_f0",
        "dm_nfnet_f1",
        "dm_nfnet_f2",
        "dm_nfnet_f3",
        "dm_nfnet_f4",
        "dm_nfnet_f5",
        "dm_nfnet_f6",
    ),
    architectures=(
        "dm_nfnet_f0",
        "dm_nfnet_f1",
        "dm_nfnet_f2",
        "dm_nfnet_f3",
        "dm_nfnet_f4",
        "dm_nfnet_f5",
        "dm_nfnet_f6",
    ),
    tasks=("classification",),
    default_task="classification",
)
