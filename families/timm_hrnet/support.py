# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint identity and task ownership for timm HRNet."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("timm_hrnet", "hrnet_w18"),
    architectures=("hrnet_w18",),
    tasks=("classification",),
    default_task="classification",
)
