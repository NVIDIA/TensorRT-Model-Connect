# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint identity and task ownership for timm RepVGG."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("timm_repvgg", "repvgg_a2"),
    architectures=("repvgg_a2",),
    tasks=("classification",),
    default_task="classification",
)
