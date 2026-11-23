# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("vjepa2",),
    tasks=("video_to_token_features",),
    default_task="video_to_token_features",
    default_precision="fp32",
)
