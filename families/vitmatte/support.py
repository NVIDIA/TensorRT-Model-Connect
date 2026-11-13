# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("vitmatte",),
    architectures=("VitMatteForImageMatting",),
    tasks=("image_trimap_to_alpha",),
    default_task="image_trimap_to_alpha",
    default_precision="fp32",
)
