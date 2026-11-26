# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Family-owned model and task support for ESMFold2."""

from tensorrt_model_connect.model_support import Variant, family_support


describe = family_support(
    model_types=("esmfold2",),
    architectures=("EsmFold2Model",),
    tasks=("structure_prediction",),
    default_task="structure_prediction",
    default_precision="bf16",
    variants=(
        Variant(precision="bf16"),
        # The strict-parity build (TF32 off) for debugging against the fp32 reference; no premerge case runs it.
        Variant(precision="fp32", status="experimental"),
    ),
)
