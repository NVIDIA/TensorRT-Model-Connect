# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for Depth Anything V2."""

from tensorrt_model_connect.model_support import family_support


# Depth Anything V2's transformers config.json carries a standard model_type
# and architectures list, unlike YOLOX or the Ultralytics YOLO releases -
# this is the ordinary Hugging Face identity path.
describe = family_support(
    model_types=("depth_anything",),
    architectures=("DepthAnythingForDepthEstimation",),
    tasks=("monocular_geometry",),
    default_task="monocular_geometry",
)
