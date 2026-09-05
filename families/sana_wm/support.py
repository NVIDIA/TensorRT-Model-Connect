# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for sana_wm."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("sana-wm", "sana_wm", "sanamsvideocamctrl_1600m_p1_d20"),
    required_files=(
        "config.yaml",
        "dit/sana_wm_1600m_720p.safetensors",
        "refiner/transformer/config.json",
    ),
    tasks=("world_model_generation",),
    default_task="world_model_generation",
)
