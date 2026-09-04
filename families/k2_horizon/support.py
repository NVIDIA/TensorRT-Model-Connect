# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for K2-Horizon."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("k2_horizon",),
    architectures=("K2HorizonForCausalLM",),
    tasks=("text_generation",),
    default_task="text_generation",
)
