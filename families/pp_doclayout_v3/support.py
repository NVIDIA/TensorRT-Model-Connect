# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discover PP-DocLayoutV3 without claiming the earlier layout architectures."""

from tensorrt_model_connect.model_support import family_support

describe = family_support(
    model_types=("pp_doclayout_v3",),
    architectures=("PPDocLayoutV3ForObjectDetection",),
    tasks=("document_layout",),
    default_task="document_layout",
)
