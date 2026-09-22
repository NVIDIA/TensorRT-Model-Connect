# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identify the family-owned HSTU checkpoint format."""

from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("hstu",),
    tasks=("recommendation",),
    default_task="recommendation",
)
