# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for timm Res2Net."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=(
        "timm_res2net",
        "res2net50_14w_8s",
        "res2net50_26w_4s",
        "res2net50_26w_6s",
        "res2net50_26w_8s",
        "res2net50_48w_2s",
        "res2net50d",
        "res2net101_26w_4s",
        "res2net101d",
        "res2next50",
    ),
    architectures=(
        "res2net50_14w_8s",
        "res2net50_26w_4s",
        "res2net50_26w_6s",
        "res2net50_26w_8s",
        "res2net50_48w_2s",
        "res2net50d",
        "res2net101_26w_4s",
        "res2net101d",
        "res2next50",
    ),
    tasks=("classification",),
    default_task="classification",
)
