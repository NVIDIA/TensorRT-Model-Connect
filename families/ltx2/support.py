# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for LTX-2.5 (``LTX2Pipeline``)."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("ltx2", "ltx-2", "ltx-2.5"),
    pipeline_classes=("LTX2Pipeline",),
    tasks=("text_to_audio_video",),
    default_task="text_to_audio_video",
    default_precision="bf16",
)
