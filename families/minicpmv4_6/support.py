# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact checkpoint identity and task support of the MiniCPM-V 4.6 family."""

from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("minicpmv4_6",),
    architectures=("MiniCPMV4_6ForConditionalGeneration",),
    tasks=("images_text_to_text",),
    default_task="images_text_to_text",
    default_precision="fp16",
)
