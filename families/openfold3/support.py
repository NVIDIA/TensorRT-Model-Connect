# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for OpenFold3."""

from tensorrt_model_connect.model_support import family_support


describe = family_support(
    required_files=(
        "of3-ob-2025-06-30-174k.pt",
        "components.bcif",
        "query.json",
        "openfold3_features.npz",
        "openfold3_structure.json",
    ),
    tasks=("structure_prediction",),
    default_task="structure_prediction",
    default_precision="fp16",
)
