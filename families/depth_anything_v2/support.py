# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for Depth Anything V2."""

from tensorrt_model_connect.model_support import family_support


# Depth Anything V2 requires the standard Hugging Face model_type identity.
# An architectures-only fallback (as families/dinov3 uses for its timm
# mirror) is deliberately not added here: unlike that mirror, no known
# depth_anything checkpoint omits model_type, and model.py's build() rejects
# anything but model_type == "depth_anything" outright, so advertising
# support on architectures alone would let describe() accept a checkpoint
# build() then immediately raises on.
describe = family_support(
    model_types=("depth_anything",),
    tasks=("monocular_geometry",),
    default_task="monocular_geometry",
)
