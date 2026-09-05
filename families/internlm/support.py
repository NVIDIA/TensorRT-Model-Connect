# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for internlm."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("internlm", "internlm2"),
    tasks=("text_generation",),
    default_task="text_generation",
)
