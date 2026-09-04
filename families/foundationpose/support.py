# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for FoundationPose."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    required_files=("refine_model.onnx", "score_model.onnx"),
    tasks=("pose_hypothesis_refinement",),
    default_task="pose_hypothesis_refinement",
)
