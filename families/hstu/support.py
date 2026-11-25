# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from tensorrt_model_connect.model_support import Variant, family_support

describe = family_support(
    model_types=("dlrmv3_hstu",),
    required_files=("dlrmv3_weights.npz",),
    tasks=("structured_ranking",),
    default_task="structured_ranking",
    default_precision="fp32",
    variants=(Variant("fp32"), Variant("bf16")),
)
