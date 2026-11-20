# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for Uni3D."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    model_types=("uni3d",),
    required_files=("uni3d.onnx",),
    tasks=("points_to_embedding",),
    default_task="points_to_embedding",
    default_precision="fp16",
)
