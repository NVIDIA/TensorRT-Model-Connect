# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint identity and task ownership for timm Swin."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("timm_swin", "swin_tiny_patch4_window7_224"),
    architectures=("swin_tiny_patch4_window7_224",),
    tasks=("classification",),
    default_task="classification",
)
